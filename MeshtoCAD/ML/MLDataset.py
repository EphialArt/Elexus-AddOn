from torch_geometric.data import Dataset
from MLDataLoader import load_mesh_to_pyg
import os
import torch
import json

class MeshCADTokenDataset(Dataset):
    def __init__(self, data_dir, transform=None, pre_transform=None):
        self.data_dir = data_dir
        self.base_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(data_dir)
            if f.endswith('.obj') and os.path.exists(os.path.join(data_dir, os.path.splitext(f)[0] + "_tokens.json"))
        ]

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base = self.base_names[idx]
        mesh_path = os.path.join(self.data_dir, base + ".obj")
        token_path = os.path.join(self.data_dir, base + "_tokens.json")
        try:
          mesh_data = load_mesh_to_pyg(mesh_path)
          with open(token_path, "r") as f:
              raw = json.load(f)

          tokens = []
          values = []
          i = 0
          while i < len(raw):
              t = int(raw[i])
              tokens.append(t)
              if t in (305, 306) and i + 1 < len(raw):
                  v = float(raw[i + 1])
                  values.append(v)
                  i += 2  
              else:
                  values.append(0.0)  
                  i += 1

          mesh_data.tokens = torch.tensor(tokens, dtype=torch.long)
          mesh_data.value_tensor = torch.tensor(values, dtype=torch.float)
          mesh_data.mesh_file = base + ".obj"
          return mesh_data
        except Exception as e:
          print(f"Skipping {base}: {e}")
          return None