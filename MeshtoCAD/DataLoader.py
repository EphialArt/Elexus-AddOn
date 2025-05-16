import torch
from torch_geometric.data import Data
import trimesh
import os
import numpy as np

def create_dummy_cube_obj(file_path):
    """Creates a dummy cube OBJ file for testing."""
    with open(file_path, "w") as f:
        f.write("v -0.5 -0.5 -0.5\n")
        f.write("v 0.5 -0.5 -0.5\n")
        f.write("v 0.5 0.5 -0.5\n")
        f.write("v -0.5 0.5 -0.5\n")
        f.write("v -0.5 -0.5 0.5\n")
        f.write("v 0.5 -0.5 0.5\n")
        f.write("v 0.5 0.5 0.5\n")
        f.write("v -0.5 0.5 0.5\n")
        f.write("f 1 2 3 4\n")
        f.write("f 5 6 7 8\n")
        f.write("f 1 5 6 2\n")
        f.write("f 2 6 7 3\n")
        f.write("f 3 7 8 4\n")
        f.write("f 4 8 5 1\n")

def create_dummy_cube_stl(file_path):
    """Creates a dummy cube STL file for testing."""
    # Define the vertices of the cube
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ])

    # Define the faces (triangles) of the cube
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0]
    ])

    # Create the mesh object
    cube_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export the mesh as an STL file
    cube_mesh.export(file_path, file_type='stl')

def load_mesh_to_pyg(file_path):
    """
    Loads a 3D mesh from a file (OBJ or STL) and converts it into a PyG Data object.

    Args:
        file_path (str): The path to the 3D model file.

    Returns:
        torch_geometric.data.Data: A PyG Data object representing the mesh.  Returns None on Error.
    """
    try:
        # 1. Determine File Type and Load Mesh
        if file_path.lower().endswith(('.obj', '.stl')):
            mesh = trimesh.load(file_path)  # Load OBJ or STL with trimesh
        else:
            print(f"Error: Unsupported file format: {file_path}.  Supported formats are: OBJ and STL.")
            return None

        # 2. Extract Vertex Coordinates
        pos = torch.tensor(mesh.vertices, dtype=torch.float32)  # Vertex coordinates (position)
        x = pos  # For this example, we'll use vertex coordinates as node features as well

        # 3. Create Edge Index from Faces
        faces = mesh.faces
        edge_list = []
        for face in faces:
            for i in range(len(face)):
                a, b = face[i], face[(i+1) % len(face)]
                edge_list.extend([[a, b], [b, a]])
        edge_set = set()
        for face in faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                edge_set.add((u, v))
                edge_set.add((v, u))  # if undirected

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

        # 4. Create PyG Data Object
        data = Data(x=x, edge_index=edge_index, pos=pos)
        return data

    except Exception as e:
        print(f"Error loading mesh from {file_path}: {e}")
        return None  # Return None in case of any error during loading

if __name__ == '__main__':
    # Example Usage
    obj_path = "cube.obj"  # Replace with a path to a simple OBJ file.  Create a dummy cube.obj if needed.
    stl_path = "cube.stl"  # Replace with a path to a simple STL file

    # Create dummy cube files if they don't exist
    if not os.path.exists(obj_path):
        create_dummy_cube_obj(obj_path)
    if not os.path.exists(stl_path):
        create_dummy_cube_stl(stl_path)
    #test the function
    data_obj = load_mesh_to_pyg(obj_path)
    if data_obj is not None:
        print(f"Loaded OBJ: {data_obj}")

    data_stl = load_mesh_to_pyg(stl_path)
    if data_stl is not None:
        print(f"Loaded STL: {data_stl}")