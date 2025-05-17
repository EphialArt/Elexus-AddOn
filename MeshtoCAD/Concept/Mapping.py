import uuid

def create_sketch_from_planar_face(face_vertices, mesh_vertices, sketch_id_prefix="sketch"):
    if len(face_vertices) < 2:
        return None, None, None

    points = [mesh_vertices[v].tolist() for v in face_vertices]
    curve_uuids = [str(uuid.uuid4()) for _ in range(len(face_vertices))]
    curves = {}
    profile_curves = []

    for i in range(len(face_vertices)):
        start_pt = points[i]
        end_pt = points[(i + 1) % len(face_vertices)]
        curve_id = curve_uuids[i]
        curves[curve_id] = {
            "type": "Line3D",
            "start_point": {
                "type": "Point3D",
                "x": start_pt[0],
                "y": start_pt[1],
                "z": start_pt[2]
            },
            "end_point": {
                "type": "Point3D",
                "x": end_pt[0],
                "y": end_pt[1],
                "z": end_pt[2]
            }
        }
        profile_curves.append({
            "type": "Line3D",
            "start_point": curves[curve_id]["start_point"],
            "end_point": curves[curve_id]["end_point"],
            "curve": curve_id
        })

    sketch_id = f"{sketch_id_prefix}_{uuid.uuid4()}"
    profile_id = f"profile_{uuid.uuid4()}"

    sketch = {
        "name": sketch_id,
        "type": "Sketch",
        "plane": "XY",
        "points": points,
        "curves": curves,
        "profiles": {
            profile_id: {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": profile_curves
                    }
                ],
                "properties": {}
            }
        },
        "constraints": {},
        "dimensions": {}
    }
    return sketch, sketch_id, profile_id

def create_extrude_from_sketch(sketch_id, profile_id, distance, extrude_id_prefix="extrude", operation_type="NewBody", taper_angle=0):
    extrude_id = f"{extrude_id_prefix}_{uuid.uuid4()}"
    return {
        "name": extrude_id,
        "type": "ExtrudeFeature",
        "profiles": [
            {
                "profile": profile_id,
                "sketch": sketch_id
            }
        ],
        "extent_one": {"distance": {"value": distance}},
        "taperAngle": {"value": taper_angle},
        "operationType": operation_type,
    }

def map_mesh_features_to_fusion_360(mesh_features, mesh_vertices):
    metadata = {
        "parent_project": "Testing",
        "component_name": "Untitled",
        "component_index": 0
    }
    entities = {}
    timeline = []
    sketch_count = 0

    if not mesh_features:
        return {"timeline": [], "entities": {}}

    for face_index, face_vertices in enumerate(mesh_features.get("planar_faces", [])):
        sketch, sketch_id, profile_id = create_sketch_from_planar_face(face_vertices, mesh_vertices, f"sketch_{sketch_count}")
        if sketch:
            entities[sketch_id] = sketch
            timeline.append({"entity": sketch_id, "index": len(timeline)})
            # Extrude using the profile and sketch reference
            extrude = create_extrude_from_sketch(sketch_id, profile_id, distance=10, operation_type="NewBody", taper_angle=0)
            extrude_id = extrude["name"]
            entities[extrude_id] = extrude
            timeline.append({"entity": extrude_id, "index": len(timeline)})
            sketch_count += 1

    return {"metadata": metadata, "timeline": timeline, "entities": entities}