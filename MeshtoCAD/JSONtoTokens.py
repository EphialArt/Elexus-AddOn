from torch_geometric.loader import DataLoader
from Preprocessing import Preprocessing
from Tokenize import tokenize_cad_steps
import json
import os
import shutil
from tqdm import tqdm

dataset = Preprocessing(mesh_dir="C:\\Users\\iceri\\Downloads\\r1.0.1\\r1.0.1\\reconstruction", cad_dir="C:\\Users\\iceri\\Downloads\\r1.0.1\\r1.0.1\\reconstruction")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

for idx, batch in enumerate(tqdm(loader, desc="Tokenizing batches")):
    for i in range(batch.num_graphs):
        mesh_file = batch.mesh_file[i]
        base_name = os.path.splitext(os.path.basename(mesh_file))[0]
        output_path = f"MeshtoCAD\\TokenizedJSON\\{base_name}_tokens.json"

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        cad_steps = batch.cad_steps[i]
        token_seq = tokenize_cad_steps(cad_steps)
        save_json(token_seq, output_path)

        mesh_ext = os.path.splitext(mesh_file)[1].lower()
        if mesh_ext in [".stl", ".obj"]:
            src_mesh_path = os.path.join(dataset.root, mesh_file)
            dst_mesh_path = f"MeshtoCAD\\TokenizedJSON\\{base_name}{mesh_ext}"
            if not os.path.exists(dst_mesh_path):
                shutil.copy2(src_mesh_path, dst_mesh_path)
    