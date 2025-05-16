from MeshAnalysis import load_mesh, extract_mesh_features
from Mapping import map_mesh_features_to_fusion_360
import json
from JsonToPy import load_json, save_json

if __name__ == '__main__':
    # 1. Load the mesh
    mesh_file_path = "MeshtoCAD/cube.obj"
    mesh = load_mesh(mesh_file_path)
    print(mesh)
    if mesh is None:
        print("Failed to load mesh.")
        exit()

    # 2. Extract mesh features
    mesh_features = extract_mesh_features(mesh)
    print("Extracted Mesh Features:", mesh_features)

    # 3. Map to Fusion 360 operations
    fusion_360_data = map_mesh_features_to_fusion_360(mesh_features, mesh.vertices)
    print("Fusion 360 Data:", json.dumps(fusion_360_data, indent=2))

    # 4. Save the JSON (optional)
    output_json_path = "fusion_360_steps.json"
    save_json(fusion_360_data, output_json_path)
    print(f"Saved Fusion 360 steps to {output_json_path}")
