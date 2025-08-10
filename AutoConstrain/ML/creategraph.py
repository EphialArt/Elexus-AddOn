import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy

# Edge role encoding (can be one-hot or integer)
role_map = {
    "none": 0,
    "start_point": 1,
    "end_point": 2,
    "center_point": 3,
    "CoincidentConstraint": 4,
    "CollinearConstraint": 5,
    "ConcentricConstraint": 6,
    "VerticalConstraint": 7,
    "HorizontalConstraint": 8,
    "EqualConstraint": 9,
    "HorizontalPointsConstraint": 10,
    "VerticalPointsConstraint": 11,
    "MidPointConstraint": 12,
    "ParallelConstraint": 13,
    "PerpendicularConstraint": 14,
    "TangentConstraint": 15,
    "SymmetryConstraint": 16,
    "RectangularPatternConstraint": 17,
    "CircularPatternConstraint": 18,
    "SmoothConstraint": 19,
}

import torch
import numpy as np
from torch_geometric.data import Data

def build_pyg_graph(sketch):
    node_ids = []
    node_features = []
    node_types = []  # 0=point, 1=curve

    # Add Points
    for pt in sketch.points.values():
        node_ids.append(pt.id)
        node_types.append(0)
        node_features.append([pt.x, pt.y, pt.z])

    # Add Curves
    curve_type_map = {"SketchLine": 1, "SketchCircle": 2, "SketchArc": 3}
    for curve in sketch.curves.values():
        node_ids.append(curve.id)
        node_types.append(1)
        radius = curve.radius if curve.radius is not None else 0.0
        node_features.append([radius, 0, 0])  # Keep feature length consistent

    # Map from node ID to node index
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_index = []
    edge_attr = []

    # Helper to compute normalized vector between two points
    def compute_dir_vector(p1, p2):
        vec = np.array(p2) - np.array(p1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return (vec / norm).tolist()
        else:
            return [0.0, 0.0, 0.0]

    # Connect curves to points with geometric features
    for curve in sketch.curves.values():
        c_idx = id_to_idx[curve.id]

        # Use curve center_point if exists, else start_point as origin for vector calculation
        if curve.center_point is not None:
            origin = [curve.center_point.x, curve.center_point.y, curve.center_point.z]
        elif curve.start_point is not None:
            origin = [curve.start_point.x, curve.start_point.y, curve.start_point.z]
        else:
            origin = [0.0, 0.0, 0.0]  # fallback

        for role in ["start_point", "end_point", "center_point"]:
            pt = getattr(curve, role)
            if pt is not None:
                pt_id = pt.id if hasattr(pt, "id") else pt
                p_idx = id_to_idx.get(pt_id)
                if p_idx is not None:
                    pt_coords = [pt.x, pt.y, pt.z]
                    dir_vec = compute_dir_vector(origin, pt_coords)

                    # edge_attr vector = [role_label, dir_x, dir_y, dir_z]
                    edge_attr_vec = [role_map[role]] + dir_vec

                    # Undirected edges
                    edge_index.append([c_idx, p_idx])
                    edge_attr.append(edge_attr_vec)
                    edge_index.append([p_idx, c_idx])
                    edge_attr.append(edge_attr_vec)

    # Add constraint edges with geometric features
    for constraint in sketch.constraints.values():
        entities = list(constraint.entities.values())
        c_type = constraint.type
        c_label = role_map.get(c_type, 9)  # 9=unknown

        if len(entities) == 2:
            idx1 = id_to_idx.get(entities[0].id)
            idx2 = id_to_idx.get(entities[1].id)
            if idx1 is not None and idx2 is not None:
                # Compute direction vector between entity centers (approximate)
                def get_coords(entity):
                    if hasattr(entity, "x") and hasattr(entity, "y") and hasattr(entity, "z"):
                        return [entity.x, entity.y, entity.z]
                    # For curves, approximate center point or fallback to zeros
                    if hasattr(entity, "center_point") and entity.center_point is not None:
                        return [entity.center_point.x, entity.center_point.y, entity.center_point.z]
                    return [0.0, 0.0, 0.0]

                coord1 = get_coords(entities[0])
                coord2 = get_coords(entities[1])
                dir_vec = compute_dir_vector(coord1, coord2)

                edge_attr_vec = [c_label] + dir_vec
                edge_index.append([idx1, idx2])
                edge_attr.append(edge_attr_vec)
                edge_index.append([idx2, idx1])
                edge_attr.append(edge_attr_vec)

        elif len(entities) == 1:
            idx = id_to_idx.get(entities[0].id)
            if idx is not None:
                # Unary constraint: direction vector = zeros
                edge_attr_vec = [c_label, 0.0, 0.0, 0.0]
                edge_index.append([idx, idx])
                edge_attr.append(edge_attr_vec)

    # Convert to torch tensors, edge_attr must be float now
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def mask_constraints(sketch, drop_rate=0.3, seed=None):
    if seed is not None:
        random.seed(seed)

    sketch_copy = copy.deepcopy(sketch)
    constraint_ids = list(sketch_copy.constraints.keys())
    num_to_drop = int(len(constraint_ids) * drop_rate)

    to_drop = set(random.sample(constraint_ids, num_to_drop))
    for cid in to_drop:
        del sketch_copy.constraints[cid]

    return sketch_copy, to_drop

def visualize_graph(data):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)

    edges = data.edge_index.t().tolist()
    for i, (src, tgt) in enumerate(edges):
        label = data.edge_attr[i][0].item()
        G.add_edge(src, tgt, label=label)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


if __name__ == "__main__":
    from parsing import parse_sketches_from_file
    filepath = "C:\\Users\\iceri\\Downloads\\r1.0.1\\r1.0.1\\reconstruction\\20203_7e31e92a_0000.json"
    
    sketches = parse_sketches_from_file(filepath)
    sketch = sketches[1]
    
    partial_sketch, dropped_constraints = mask_constraints(sketch, drop_rate=0.4, seed=42)

    data_full = build_pyg_graph(sketch)
    data_partial = build_pyg_graph(partial_sketch)

    visualize_graph(data_full)
    visualize_graph(data_partial)
