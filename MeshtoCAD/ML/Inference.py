import torch
from Encode_Decode import MeshToCADModel
from MLDataLoader import load_mesh_to_pyg
import json
VOCAB_SIZE = 501
MODEL_PATH = "MeshtoCAD/ML/best_model_epoch8.pt"  
EOS_TOKEN = 501

model = MeshToCADModel(
    mesh_in_channels=3,
    mesh_hidden=128,
    mesh_latent=256,
    vocab_size=VOCAB_SIZE,
    dec_hidden=256,
    dec_layers=2
)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def generate_tokens(model, mesh_data, max_len=4234, start_token=0, device='cuda'):
    model = model.to(device)
    model.eval()
    mesh_data = mesh_data.to(device)
    
    latent = model.encoder(mesh_data)
    print("Latent mean:", latent.mean().item())
    
    tokens = [start_token]
    values = [0.0]
    
    for step in range(max_len):
        input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        values_seq = torch.tensor(values, dtype=torch.float, device=device).unsqueeze(0)
        
        logits = model.decoder(input_seq, values_seq, latent)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        if next_token == EOS_TOKEN:
            break

        tokens.append(next_token)
        values.append(0.0 if next_token in (305, 306) else 0.0) 
        
        print(f"Step {step}: token={next_token}, top-5 probs={probs[0].topk(5).values.tolist()}")
    return tokens


mesh_path = "MeshtoCAD/TokenizedJSON/test/20232_e5b060d9_0002.obj"
mesh_data = load_mesh_to_pyg(mesh_path)
predicted_tokens = generate_tokens(model, mesh_data)

save_json(predicted_tokens, "MeshtoCAD/ML/output/predicted_tokens.json")
print("Predicted CAD token sequence:", predicted_tokens)