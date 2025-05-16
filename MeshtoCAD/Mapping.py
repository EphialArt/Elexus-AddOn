import uuid 

def create_sketch_from_planar_face(face_vertices, mesh_vertices, sketch_id_prefix="sketch"):
    """
    Creates a Sketch entity from a planar face.

    Args:
        face_vertices (list): A list of vertex *indices* defining the planar face.
        mesh_vertices (np.ndarray): The array of mesh vertex coordinates.
        sketch_id_prefix (str): Prefix for the generated sketch ID.

    Returns:
        dict: A dictionary representing the Sketch entity in the Fusion 360 JSON format,
              or None if the face has fewer than 2 vertices.
    """
    if len(face_vertices) < 2:
        return None  # Can't create a sketch from a single point

    points = [mesh_vertices[v].tolist() for v in face_vertices] # Get the actual coordinates
    lines = []
    for i in range(len(face_vertices)):
        start_index = face_vertices[i]
        end_index = face_vertices[(i + 1) % len(face_vertices)]
        lines.append({
            "start": points[i],
            "end": points[(i + 1) % len(face_vertices)],
            "type": "SketchLine"  # Add the type of curve
        })
    # Basic sketch.  You'd likely want to add constraints.
    return {
        "id": f"{sketch_id_prefix}_{uuid.uuid4()}",  # Generate a unique ID
        "type": "Sketch",
        "plane": "XY",  # You'll need to determine the correct plane
        "points": points,
        "curves": {f"line_{i}": line for i, line in enumerate(lines)},  # Store lines in a dictionary
        "constraints": {},
        "dimensions": {}
    }

def create_extrude_from_sketch(sketch_id, distance, extrude_id_prefix="extrude", operation_type="NewBody", taper_angle=0):
    """
    Creates an ExtrudeFeature entity from a Sketch.

    Args:
        sketch_id (str): The ID of the Sketch entity to extrude.
        distance (float): The extrusion distance.
        extrude_id_prefix (str): Prefix for the generated extrude ID.
        operation_type (str): The type of extrusion operation ("NewBody", "Cut", "Join").
        taper_angle (float): The taper angle of the extrusion.

    Returns:
        dict: A dictionary representing the ExtrudeFeature entity in the Fusion 360 JSON format.
    """
    return {
        "name": f"{extrude_id_prefix}_{uuid.uuid4()}",
        "type": "ExtrudeFeature",
        "profiles": [sketch_id],
        "extent_one": {"distance": {"value": distance}},
        "taperAngle": {"value": taper_angle},
        "operationType": operation_type,
        # Add other relevant parameters as needed
    }

def map_mesh_features_to_fusion_360(mesh_features, mesh_vertices):
    """
    Maps extracted mesh features to a sequence of Fusion 360 operations in JSON format.

    Args:
        mesh_features (dict): A dictionary of extracted mesh features
                              (e.g., from extract_mesh_features).

    Returns:
        dict: A dictionary containing 'timeline' and 'entities' in the Fusion 360
              JSON format.  Returns an empty dict on error.
    """
    entities = {}
    timeline = []
    sketch_count = 0
    extrude_count = 0

    if not mesh_features:
        return {"timeline": [], "entities": {}}

    # 1. Process Planar Faces into Sketches and Extrusions
    for face_index, face_vertices in enumerate(mesh_features.get("planar_faces", [])):
        sketch = create_sketch_from_planar_face(face_vertices, mesh_vertices, f"sketch_{sketch_count}")
        if sketch:
            sketch_id = sketch["id"]
            entities[sketch_id] = sketch
            timeline.append({"entity": sketch_id, "index": len(timeline)})  # Add to timeline
            sketch_count+=1
            # For simplicity, extrude all planar faces by a fixed distance (you'll need better logic)
            extrude = create_extrude_from_sketch(sketch_id, distance=10, operation_type="NewBody", taper_angle=0)  # Example distance
            extrude_id = extrude["id"]
            entities[extrude_id] = extrude
            timeline.append({"entity": extrude_id, "index": len(timeline)})  # Add to timeline
            extrude_count+=1

    # 2. Process Cylindrical Faces (Placeholder - Requires Cylinder Fitting)
    #  -  When you have a proper detect_cylindrical_faces, map them to RevolveFeatures

    # 3.  Add logic for other features (fillets, holes, etc.)

    return {"timeline": timeline, "entities": entities}
