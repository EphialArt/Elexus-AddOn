import torch
from Encode_Decode import MeshToCADModel, batch_vertex_features
from MLDataset import MeshCADTokenDataset  

CHECKPOINT_PATH = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 502
MAX_LENGTH = 2000
START_TOKEN = 0      
EOS_TOKEN = 501    
PAD_VALUE = -1e6     

model = MeshToCADModel(
    mesh_in_channels=3,
    mesh_hidden=128,
    mesh_latent=256,
    vocab_size=VOCAB_SIZE,
    dec_hidden=256,
    dec_layers=2
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = MeshCADTokenDataset("MeshtoCAD/TokenizedJSON/test")
mesh_data = dataset[0] 
mesh_data = mesh_data.to(DEVICE)

with torch.no_grad():
    if not hasattr(mesh_data, 'batch') or mesh_data.batch is None:
        mesh_data.batch = torch.zeros(mesh_data.x.size(0), dtype=torch.long).to(mesh_data.x.device)

    vertex_feats, batch = model.encoder(mesh_data, return_all_vertices=True)
    vertex_feats_batched, vertex_mask = batch_vertex_features(vertex_feats, batch)

    vertex_feats_batched = vertex_feats_batched  # [1, S, latent_dim]
    vertex_mask = vertex_mask  # [1, S]

tokens = [START_TOKEN]
floats = [PAD_VALUE]  
ints = [-1]
uuids = [-1]

for step in range(MAX_LENGTH):
    token_tensor = torch.tensor(tokens, device=DEVICE).unsqueeze(0)  # [1, T]
    float_tensor = torch.tensor(floats, device=DEVICE).unsqueeze(0)  # [1, T]
    int_tensor = torch.tensor(ints, device=DEVICE)     # [1, T]
    uuid_tensor = torch.tensor(uuids, device=DEVICE)

    with torch.no_grad():
        logits, float_preds, int_preds, uuid_preds = model.decoder(
            token_tensor, float_tensor, int_tensor, uuid_tensor,
            vertex_feats_batched, vertex_mask
        )
   
    top_k = 20
    temperature = 1.0
    next_token_logits = logits[0, -1]
    probs = torch.softmax(next_token_logits / temperature, dim=-1)
    topk_logits, topk_indices = torch.topk(next_token_logits, top_k)
    topk_probs = torch.softmax(topk_logits / temperature, dim=-1)

    next_token = topk_indices[torch.multinomial(topk_probs, 1)].item()

    if next_token == 310:
        next_float = float_preds[0, -1].item()
        next_int = -1 
        next_uuid = -1
    elif next_token == 311:
        next_float = PAD_VALUE
        next_int = torch.argmax(int_preds[0, -1]).item()
        next_uuid = -1
    elif next_token == 312:
        next_uuid = torch.argmax(uuid_preds[0, -1]).item()
        next_float = PAD_VALUE
        next_int = -1
    else:
        next_float = PAD_VALUE
        next_int = -1
        next_uuid = -1

    tokens.append(next_token)
    floats.append(next_float)
    ints.append(next_int)
    uuids.append(next_uuid)

    if next_token == EOS_TOKEN:
        break

print("Generated tokens:", tokens)
print("Generated float values:", floats)
print("Generated int values:", ints)
print("Generated uuid values:", uuids)
