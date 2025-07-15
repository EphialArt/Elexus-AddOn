import torch
from torch_geometric.data import Data
import trimesh
import os
import numpy as np


def load_mesh_to_pyg(file_path):
    try:
        if not file_path.lower().endswith(('.obj', '.stl')):
            print(f"Error: Unsupported file format: {file_path}. Supported formats are: OBJ and STL.")
            return None

        mesh = trimesh.load(file_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump().values()))

        pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        x = pos  
        faces = mesh.faces
        edge_set = set()
        for face in faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                edge_set.add((u, v))
                edge_set.add((v, u))

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, pos=pos)
        return data

    except Exception as e:
        print(f"Error loading mesh from {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    mesh_path = "MeshtoCAD/TokenizedJSON/test/20232_e5b060d9_0002.obj" 
    mesh_data = load_mesh_to_pyg(mesh_path)
    if mesh_data is not None:
        print("Mesh data loaded successfully.")
        print(f"Vertices shape: {mesh_data.x.shape}")
        print(f"Edge index shape: {mesh_data.edge_index.shape}")
    else:
        print("Failed to load mesh data.")