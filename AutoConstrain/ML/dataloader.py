import glob
from parsing import parse_sketches_from_file
from creategraph import mask_constraints, build_pyg_graph, visualize_graph

def is_sketch_valid(sketch):
    for constraint in sketch.constraints.values():
        for entity in constraint.entities.values():
            if isinstance(entity, str):
                return False
    return True

def load_dataset(files, drop_rate=0.3, seed=None):
    full_dataset = []
    partial_dataset = []
    for f in files:
        try:
            sketches = parse_sketches_from_file(f)
        except Exception as e:
            print(f"Error parsing sketches from file {f}: {e}")
            continue
        for sketch in sketches:
            if not is_sketch_valid(sketch):
                # print(f"Skipping sketch {sketch.id} due to unresolved references.")
                continue
            partial_sketch, dropped_constraints = mask_constraints(sketch, drop_rate=drop_rate, seed=seed)
            # print(partial_sketch.constraints.items())
            # print(sketch.constraints.items())
            data_partial = build_pyg_graph(partial_sketch)
            data_full = build_pyg_graph(sketch)
            # Assign node ids (points + curves)
            node_ids = list(partial_sketch.points.keys()) + list(partial_sketch.curves.keys())
            for nid in node_ids:
                if nid not in partial_sketch.points and nid not in partial_sketch.curves:
                    print("Missing entity for id:", nid)
            with open("AutoConstrain/node_ids.txt", "w") as f:
                f.write("\n".join(node_ids))
                f.write("\n" + partial_sketch.id)
            data_partial.x_id = node_ids
            data_partial.sketch = partial_sketch
            data_partial.dropped_constraints = dropped_constraints

            data_full.x_id = node_ids
            data_full.sketch = sketch

            full_dataset.append(data_full)
            partial_dataset.append(data_partial)
    return full_dataset, partial_dataset

if __name__ == "__main__":
    full_dataset, partial_dataset = load_dataset("AutoConstrain/Dataset/train", drop_rate=0.9)
    print(full_dataset[0].sketch.name if full_dataset else "No valid sketches found.")
    print(partial_dataset[0])
    print(len(partial_dataset), "sketches")
    visualize_graph(partial_dataset[0])
    visualize_graph(full_dataset[0]) 