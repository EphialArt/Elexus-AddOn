import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint
from utils import MAX_TOTAL_TOKENS
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.attn_bias import LowerTriangularMask

class MeshEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, data, return_all_vertices=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        if return_all_vertices:
            return x, batch  # Return vertex features AND batch info
        latent = global_mean_pool(x, batch)
        return latent


def batch_vertex_features(vertex_feats, batch):
    B = batch.max().item() + 1
    latent_dim = vertex_feats.size(1)
    verts_per_mesh = []

    for i in range(B):
        verts_per_mesh.append(vertex_feats[batch == i])

    padded_feats = pad_sequence(verts_per_mesh, batch_first=True, padding_value=0)  
    # mask: True where padding, False where real data
    lengths = torch.tensor([v.size(0) for v in verts_per_mesh], device=vertex_feats.device)
    max_len = padded_feats.size(1)
    mask = torch.arange(max_len, device=vertex_feats.device).unsqueeze(0) >= lengths.unsqueeze(1)
    return padded_feats, mask

class XFormerDecoderLayer(nn.Module):
    def __init__(self, dim, nhead, ffn_dim):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        assert dim % nhead == 0, "dim must be divisible by nhead"

        self.qkv = nn.Linear(dim, dim * 3)
        self.context_kv = nn.Linear(dim, dim * 2)  # for keys and values from mesh memory

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.bias = LowerTriangularMask()

    def forward(self, x, context):
        B, T, D = x.shape
        S = context.size(1)

        # Self-attention q/k/v
        qkv = self.qkv(self.norm1(x)).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k_self, v_self = qkv.unbind(2)

        # Project context into k/v and reshape to [B, S, nhead, head_dim]
        context_kv = self.context_kv(context).reshape(B, S, 2, self.nhead, self.head_dim)
        k_ctx, v_ctx = context_kv.unbind(2)

        # Concatenate self-attention and cross-attention
        k_full = torch.cat([k_self, k_ctx], dim=1)  # [B, T+S, nhead, head_dim]
        v_full = torch.cat([v_self, v_ctx], dim=1)

        attn_out = memory_efficient_attention(
            q, k_full, v_full,
            attn_bias=self.bias,
            p=0.0
        )

        attn_out = attn_out.reshape(B, T, D)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TokenDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, hidden_dim, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(MAX_TOTAL_TOKENS, hidden_dim)
        self.float_proj = nn.Linear(1, hidden_dim)
        self.int_proj = nn.Linear(1, hidden_dim)
        self.uuid_proj = nn.Linear(1, hidden_dim)
        self.mesh_feat_proj = nn.Linear(latent_dim, hidden_dim)

        self.layers = nn.ModuleList([
            XFormerDecoderLayer(hidden_dim, nhead, hidden_dim * 4)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.float_head = nn.Linear(hidden_dim, 1)
        self.int_head = nn.Linear(hidden_dim, 270)
        self.uuid_head = nn.Linear(hidden_dim, 1200)

    def forward(self, tokens, floats, ints, uuids, vertex_features, vertex_mask):
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        embedded = self.embedding(tokens) + self.pos_embedding(positions)

        float_mask = (tokens == 310).unsqueeze(-1)
        int_mask = (tokens == 311).unsqueeze(-1)
        uuid_mask = (tokens == 312).unsqueeze(-1)
        float_emb = self.float_proj(floats.unsqueeze(-1))
        int_emb = self.int_proj(ints.unsqueeze(-1).float())
        uuid_emb = self.uuid_proj(uuids.unsqueeze(-1).float())
        embedded = embedded + float_emb * float_mask.float() + int_emb * int_mask.float() + uuid_emb * uuid_mask.float()

        mesh_memory = self.mesh_feat_proj(vertex_features)
        mesh_memory = mesh_memory.masked_fill(vertex_mask.unsqueeze(-1), 0.)

        x = embedded
        for layer in self.layers:
            x = layer(x, mesh_memory) 

        logits = self.fc(x)
        float_preds = self.float_head(x).squeeze(-1)
        int_preds = self.int_head(x)
        uuid_preds = self.uuid_head(x)

        return logits, float_preds, int_preds, uuid_preds

class MeshToCADModel(nn.Module):
    def __init__(self, mesh_in_channels, mesh_hidden, mesh_latent, vocab_size, dec_hidden, dec_layers=2):
        super().__init__()
        self.encoder = MeshEncoder(mesh_in_channels, mesh_hidden, mesh_latent)
        self.decoder = TokenDecoder(mesh_latent, vocab_size, dec_hidden, dec_layers)

    def forward(self, tokens, floats, ints, uuids, mesh_data):
        # Encode mesh: get vertex features + batch info
        vertex_features, batch = self.encoder(mesh_data, return_all_vertices=True)  # [Total_V, latent_dim], [Total_V]
        
        # Batch and pad vertex features per mesh in batch
        vertex_features_batched, vertex_mask = batch_vertex_features(vertex_features, batch)  # [B, S, latent_dim], [B, S]

        # Decode
        logits, float_preds, int_preds, uuid_preds = self.decoder(tokens, floats, ints, uuids, vertex_features_batched, vertex_mask)
        return logits, float_preds, int_preds, uuid_preds