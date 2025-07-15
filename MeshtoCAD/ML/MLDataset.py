from torch_geometric.data import Dataset
from MLDataLoader import load_mesh_to_pyg
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size, max_tokens=None, shuffle=True):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle

        # Sort indices by length
        self.sorted_indices = np.argsort(self.lengths)
        
        # Create buckets of batch indices
        self.batches = []
        if max_tokens is None:
            # Fixed batch size
            for i in range(0, len(self.sorted_indices), batch_size):
                self.batches.append(self.sorted_indices[i:i+batch_size])
        else:
            # Dynamic batching by max tokens per batch
            i = 0
            while i < len(self.sorted_indices):
                batch = []
                token_count = 0
                while i < len(self.sorted_indices) and len(batch) < batch_size:
                    idx = self.sorted_indices[i]
                    length = self.lengths[idx]
                    if token_count + length <= max_tokens or len(batch) == 0:
                        batch.append(idx)
                        token_count += length
                        i += 1
                    else:
                        break
                self.batches.append(np.array(batch))
        
        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield list(batch)

    def __len__(self):
        return len(self.batches)

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

            # Load tokens
            with open(token_path, "r") as f:
                raw = json.load(f)

            tokens, floats, ints, uuids = [], [], [], []
            i = 0
            while i < len(raw):
                t = int(raw[i])
                tokens.append(t)

                if t == 310 and i + 1 < len(raw):
                    floats.append(float(raw[i + 1]))
                    ints.append(0)
                    uuids.append(0)
                    i += 2
                elif t == 311 and i + 1 < len(raw):
                    ints.append(int(raw[i + 1]))
                    floats.append(0.0)
                    uuids.append(0)
                    i += 2
                elif t == 312 and i + 1 < len(raw):
                    uuids.append(int(raw[i+1]))
                    floats.append(0.0)
                    ints.append(0)
                    i+=2
                else:
                    floats.append(0.0)
                    ints.append(0)
                    uuids.append(0)
                    i += 1

            tokens = torch.tensor(tokens, dtype=torch.long)
            floats = torch.tensor(floats, dtype=torch.float)
            ints = torch.tensor(ints, dtype=torch.long)
            uuids = torch.tensor(uuids, dtype=torch.long)

            # Attach tensors to mesh_data for model input
            mesh_data.tokens = tokens
            mesh_data.float_tensor = floats
            mesh_data.int_tensor = ints
            mesh_data.uuid_tensor = uuids
            mesh_data.mesh_file = base + ".obj"

            return mesh_data

        except Exception as e:
            print(f"Skipping {base} due to error: {e}")
            return None
