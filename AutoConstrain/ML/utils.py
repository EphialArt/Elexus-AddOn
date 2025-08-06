import torch
from model import ConstraintPredictorGNN
from parsing import parse_sketches_from_file
from creategraph import build_pyg_graph, mask_constraints, visualize_graph, role_map
import copy
import torch
import torch.nn.functional as F
from sketch import Constraint

MODEL_PATH = "AutoConstrain/ML/checkpoints/best_model_epoch1.pt"
SKETCH_PATH = "AutoConstrain/Dataset/test/20232_e5b060d9_0002.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_constraints(partial_sketch, model, x_id, idx_to_role, device, conf_threshold=0.8):
    # Build PyG graph data for the partial sketch
    data_partial = build_pyg_graph(partial_sketch)
    data_partial.x_id = x_id
    data_partial = data_partial.to(device)

    model.eval()
    with torch.no_grad():
        edge_logits, candidates, _ = model(data_partial)
        probs = F.softmax(edge_logits, dim=1)
        confidences, pred_classes = torch.max(probs, dim=1)

    predicted_sketch = copy.deepcopy(partial_sketch)
    counter = 0

    for i in range(len(candidates)):
        conf = confidences[i].item()
        if conf < conf_threshold:
            continue

        class_id = pred_classes[i].item()
        c_type = idx_to_role.get(class_id, "unknown")
        if c_type in {"start_point", "end_point", "center_point"}:
            continue

        n1_idx, n2_idx = candidates[i].tolist()
        ent1 = x_id[n1_idx]
        ent2 = x_id[n2_idx]

        resolved_ent1 = partial_sketch.points.get(ent1) or partial_sketch.curves.get(ent1)
        resolved_ent2 = partial_sketch.points.get(ent2) or partial_sketch.curves.get(ent2)

        if resolved_ent1 is None or resolved_ent2 is None:
            continue  # Can't build a valid constraint

        entities = {"entity1": resolved_ent1}
        if ent1 != ent2:
            entities["entity2"] = resolved_ent2

        constraint_id = f"predicted_{counter}"
        counter += 1

        constraint = Constraint(
            id=constraint_id,
            constraint_type=c_type,
            entities=entities
        )
        predicted_sketch.constraints[constraint_id] = constraint

    return predicted_sketch

idx_to_role = {v: k for k, v in role_map.items()}

sketch = parse_sketches_from_file(SKETCH_PATH)[0]
partial_sketch, dropped_constraints = mask_constraints(sketch, drop_rate=0.3)

data_full = build_pyg_graph(sketch)
data_full = data_full.to(DEVICE)

in_channels = data_full.x.size(1)
num_edge_classes = len(role_map)
model = ConstraintPredictorGNN(in_channels, hidden_channels=64, num_edge_classes=num_edge_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

predicted_sketch = predict_constraints(
    partial_sketch,
    model,
    x_id=list(partial_sketch.points.keys()) + list(partial_sketch.curves.keys()),
    idx_to_role=idx_to_role,
    device=DEVICE,
    conf_threshold=0.8
)

data_predicted = build_pyg_graph(predicted_sketch)

visualize_graph(data_predicted) 
visualize_graph(data_full) 
