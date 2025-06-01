from torch_geometric.data import Dataset
from ML.MLDataLoader import load_mesh_to_pyg
import os

class Preprocessing(Dataset):
    def __init__(self, mesh_dir, cad_dir, transform=None, pre_transform=None):
        super().__init__(mesh_dir, transform, pre_transform)
        self.cad_dir = cad_dir
        all_mesh_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(('.obj', '.stl'))])
        self.mesh_files = []
        for mesh_file in all_mesh_files:
            cad_filename = os.path.splitext(mesh_file)[0] + ".json"
            cad_path = os.path.join(cad_dir, cad_filename)
            if os.path.isfile(cad_path):
                self.mesh_files.append(mesh_file)

    def len(self):
        return len(self.mesh_files)

    def get(self, idx):
        mesh_path = os.path.join(self.root, self.mesh_files[idx])
        cad_filename = os.path.splitext(self.mesh_files[idx])[0] + ".json"
        cad_path = os.path.join(self.cad_dir, cad_filename)
        with open(cad_path, "r") as f:
            cad_steps = f.read()
        mesh_data = load_mesh_to_pyg(mesh_path)
        mesh_data.cad_steps = cad_steps
        mesh_data.mesh_file = self.mesh_files[idx]  # <-- Add this line
        return mesh_data
    
    