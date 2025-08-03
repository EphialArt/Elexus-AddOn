import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy

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
        # Features: radius + dummy zeros for consistent feature length
        node_features.append([radius, 0, 0])

    # Map from node ID to node index
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_index = []
    edge_attr = []

    # Edge role encoding (can be one-hot or integer)
    role_map = {
        "start_point": 0,
        "end_point": 1,
        "center_point": 2,
        "CoincidentConstraint": 3,
        "CollinearConstraint": 4,
        "ConcentricConstraint": 5,
        "VerticalConstraint": 6,
        "HorizontalConstraint": 7,
        "EqualConstraint": 8,
        "HorizontalPointsConstraint": 9,
        "VerticalPointsConstraint": 10,
        "MidPointConstraint": 11,
        "OffsetConstraint": 12,
        "ParallelConstraint": 13,
        "PerpendicularConstraint": 14,
        "PolygonConstraint": 15,
        "TangentConstraint": 16,
        "SymmetryConstraint": 17,
        "RectangularPatternConstraint": 18,
        "CircularPatternConstraint": 19,
        "SmoothConstraint": 20,

    }

    # Connect curves to points
    for curve in sketch.curves.values():
        c_idx = id_to_idx[curve.id]
        for role in ["start_point", "end_point", "center_point"]:
            pt = getattr(curve, role)
            if pt is not None:
                # If pt is an object, use pt.id; if it's an ID, use pt
                pt_id = pt.id if hasattr(pt, "id") else pt
                p_idx = id_to_idx.get(pt_id)
                if p_idx is not None:
                    # Undirected graph edges
                    edge_index.append([c_idx, p_idx])
                    edge_attr.append([role_map[role]])
                    edge_index.append([p_idx, c_idx])
                    edge_attr.append([role_map[role]])

    # Add constraint edges
    for constraint in sketch.constraints.values():
        entities = list(constraint.entities.values())
        c_type = constraint.type

        # Binary constraint (between two curves)
        if len(entities) == 2:
            idx1 = id_to_idx.get(entities[0].id)
            idx2 = id_to_idx.get(entities[1].id)
            if idx1 is not None and idx2 is not None:
                edge_index.append([idx1, idx2])
                edge_attr.append([role_map.get(c_type, 9)])  # 9 = unknown
                edge_index.append([idx2, idx1])
                edge_attr.append([role_map.get(c_type, 9)])

        # Unary constraint (on one curve) -> self-loop edge
        elif len(entities) == 1:
            idx = id_to_idx.get(entities[0].id)
            if idx is not None:
                edge_index.append([idx, idx])
                edge_attr.append([role_map.get(c_type, 9)])

    # Convert to torch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
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
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)

    edges = data.edge_index.t().tolist()
    for i, (src, tgt) in enumerate(edges):
        G.add_edge(src, tgt, label=data.edge_attr[i].item())

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == "__main__":
    from parsing import parse_sketches_from_file
    filepath = "C:\\Users\\iceri\\Downloads\\r1.0.1\\r1.0.1\\reconstruction\\20203_7e31e92a_0000.json"
    
    sketches = parse_sketches_from_file(filepath)
    sketch = sketches[0]
    
    partial_sketch, dropped_constraints = mask_constraints(sketch, drop_rate=0.4, seed=42)

    data_full = build_pyg_graph(sketch)
    data_partial = build_pyg_graph(partial_sketch)

    visualize_graph(data_full)
    visualize_graph(data_partial)
