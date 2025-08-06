from html import entities
import json
from sketch import Sketch, Point, Curve, Constraint

def parse_sketches_from_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    sketches = []
    for entity_id, entity_info in data.get("entities", {}).items():
        if entity_info.get("type") != "Sketch":
            continue

        # Create Sketch object
        sketch = Sketch(
            id=entity_id,
            name=entity_info.get("name", "Unnamed Sketch"),
            sketch_type=entity_info.get("type")
        )

        # Parse Points
        for point_id, point_info in entity_info.get("points", {}).items():
            point = Point(
                id=point_id,
                x=point_info.get("x", 0.0),
                y=point_info.get("y", 0.0),
                z=point_info.get("z", 0.0)
            )
            sketch.points[point_id] = point

        # Parse Curves (initially with point IDs as references)
        for curve_id, curve_info in entity_info.get("curves", {}).items():
            curve = Curve(
                id=curve_id,
                curve_type=curve_info.get("type"),
                start_point=curve_info.get("start_point"),
                end_point=curve_info.get("end_point"),
                center_point=curve_info.get("center_point"),
                radius=curve_info.get("radius"),
                **{k: v for k, v in curve_info.items() if k not in ["type", "start_point", "end_point", "center_point", "radius"]}
            )
            sketch.curves[curve_id] = curve

        # Resolve point references in curves
        for curve in sketch.curves.values():
            if isinstance(curve.start_point, str):
                curve.start_point = sketch.points.get(curve.start_point)
            if isinstance(curve.end_point, str):
                curve.end_point = sketch.points.get(curve.end_point)
            if isinstance(curve.center_point, str):
                curve.center_point = sketch.points.get(curve.center_point)

        # Parse Constraints (initially with entity IDs)
        raw_constraints = {}
        for constraint_id, constraint_info in entity_info.get("constraints", {}).items():
            if constraint_info.get("type") in ["OffsetConstraint", "PolygonConstraint"]:
                continue
            entities = {k: v for k, v in constraint_info.items() if k != "type"}
            if any(v is None for v in entities.values()):
                continue
            constraint = Constraint(
                id=constraint_id,
                constraint_type=constraint_info.get("type"),
                entities=entities
            )
            raw_constraints[constraint_id] = constraint

        # Resolve entity references in constraints
        resolved_constraints = {}
        for cid, constraint in raw_constraints.items():
            skip = False
            for key, ref_id in list(constraint.entities.items()):
                if isinstance(ref_id, list):
                    resolved_list = []
                    for rid in ref_id:
                        resolved = sketch.points.get(rid) or sketch.curves.get(rid)
                        if resolved is None:
                            skip = True
                            break
                        resolved_list.append(resolved)
                    constraint.entities[key] = resolved_list
                else:
                    resolved = sketch.points.get(ref_id) or sketch.curves.get(ref_id)
                    if resolved is None:
                        skip = True
                        break
                    constraint.entities[key] = resolved
            if not skip:
                resolved_constraints[cid] = constraint
            else:
                # print(f"[!] Skipped constraint {cid} in {filepath} due to unresolved reference(s)")
                continue

        sketch.constraints = resolved_constraints

        sketches.append(sketch)

    return sketches

if __name__ == "__main__":
    filepath = "C:\\Users\\iceri\\Downloads\\r1.0.1\\r1.0.1\\reconstruction\\20203_7e31e92a_0000.json"
    sketches = parse_sketches_from_file(filepath)

    if sketches and sketches[0].points:
        print("Points in first sketch:", [p.id for p in sketches[0].points.values()])
    if sketches and sketches[0].curves:
        print("Curves in first sketch:", [c.id for c in sketches[0].curves.values()])
    if sketches and sketches[0].constraints:
        print("Constraints in first sketch:", [c.id for c in sketches[0].constraints.values()])
