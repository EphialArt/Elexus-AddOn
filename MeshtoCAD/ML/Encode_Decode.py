import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MeshEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.value_proj = nn.Linear(1, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        latent = global_mean_pool(x, batch)
        return latent

class TokenDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)

    def forward(self, tokens, values, latent):
        print("tokens min:", tokens.min().item(), "max:", tokens.max().item(), "VOCAB_SIZE:", self.embedding.num_embeddings)
        embedded = self.embedding(tokens) 
        value_emb = self.value_proj(values.unsqueeze(-1)) 
        mask = ((tokens == 305) | (tokens == 306)).unsqueeze(-1)
        embedded = embedded + value_emb * mask.float()
        h0 = self.latent_to_hidden(latent)
        h0 = h0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        output, _ = self.rnn(embedded.contiguous(), h0)
        logits = self.fc(output)
        return logits

class MeshToCADModel(nn.Module):
    def __init__(self, mesh_in_channels, mesh_hidden, mesh_latent, vocab_size, dec_hidden, dec_layers=2):
        super().__init__()
        self.encoder = MeshEncoder(mesh_in_channels, mesh_hidden, mesh_latent)
        self.decoder = TokenDecoder(mesh_latent, vocab_size, dec_hidden, dec_layers)

    def forward(self, mesh_data, tokens, values):
        latent = self.encoder(mesh_data)
        logits = self.decoder(tokens, values, latent)
        return logits