import json

BASE_TOKENS = {
    "SKETCH": 0, "EXTRUDE": 1, "FILLET": 2, "CHAMFER": 3, "REVOLVE": 4, "SWEEP": 5, "LOFT": 6,
    "BOOLEAN": 7, "MOVE": 8, "COPY": 9, "MIRROR": 10, "PATTERN": 11, "FEATURE_SCRIPT": 12,
}
PARAM_TOKENS = {
    "ID": 20, "PLANE": 21, "POINTS": 22, "CURVES": 23, "LINES": 24, "ARCS": 25, "CIRCLES": 26,
    "SPLINES": 27, "START": 28, "END": 29, "CENTER": 30, "RADIUS": 31, "DIAMETER": 32, "DEGREE": 33,
    "CONTROL_POINTS": 34, "KNOTS": 35, "WEIGHTS": 36, "CONSTRAINTS": 37, "TYPE": 38, "ENTITIES": 39,
    "DIMENSIONS": 40, "ENTITY": 41, "VALUE": 42, "EXTENT_ONE": 43, "DISTANCE": 44, "TAPER_ANGLE": 45,
    "OPERATION_TYPE": 46, "FILLET_EDGES": 47, "CHAMFER_EDGES": 48, "DISTANCE_ONE": 49, "DISTANCE_TWO": 50,
    "ANGLE": 51, "AXIS": 52, "PROFILE": 53, "PATH": 54, "GUIDE_RAILS": 55, "BODIES": 56, "TOOL_BODY": 57,
    "TARGET_BODY": 58, "TRANSFORM": 59, "VECTOR": 60, "AMOUNT": 61, "COUNT": 62, "PATTERN_TYPE": 63,
    "MIRROR_PLANE": 64, "SOURCE_FEATURES": 65, "SCRIPT_CODE": 66, "SCRIPT_NAME": 67, "INPUTS": 68, "OUTPUTS": 69,
}
VALUE_TOKENS = {
    "PLANE_XY": 100, "PLANE_YZ": 101, "PLANE_ZX": 102,
    "CONSTRAINT_HORIZONTAL": 103, "CONSTRAINT_VERTICAL": 104, "CONSTRAINT_COINCIDENT": 105,
    "CONSTRAINT_TANGENT": 106, "CONSTRAINT_PARALLEL": 107, "CONSTRAINT_PERPENDICULAR": 108,
    "CONSTRAINT_EQUAL": 109, "CONSTRAINT_FIX": 110,
    "OPERATION_NEW_BODY": 111, "OPERATION_CUT": 112, "OPERATION_JOIN": 113, "OPERATION_INTERSECT": 114,
    "<FLOAT>": 200, "<INTEGER>": 201, "<UUID>": 202, "<BOOLEAN>": 203,
}
STRUCTURE_TOKENS = {"{": 300, "}": 301, "[": 302, "]": 303, ":": 304, ",": 305}

def tokenize_value(val):
    if isinstance(val, float):
        return [VALUE_TOKENS["<FLOAT>"], val]
    elif isinstance(val, int):
        return [VALUE_TOKENS["<INTEGER>"], val]
    elif isinstance(val, str):
        # UUID detection 
        if len(val) == 36 and val.count('-') == 4:
            return [VALUE_TOKENS["<UUID>"], val]
        # Check if it's a known value token
        return [VALUE_TOKENS.get(val.upper(), val)]
    elif isinstance(val, bool):
        return [VALUE_TOKENS["<BOOLEAN>"], int(val)]
    elif isinstance(val, list):
        tokens = [STRUCTURE_TOKENS["["]]
        for i, item in enumerate(val):
            tokens += tokenize_value(item)
            if i < len(val) - 1:
                tokens.append(STRUCTURE_TOKENS[","])
        tokens.append(STRUCTURE_TOKENS["]"])
        return tokens
    elif isinstance(val, dict):
        tokens = [STRUCTURE_TOKENS["{"]]
        for i, (k, v) in enumerate(val.items()):
            tokens.append(PARAM_TOKENS.get(k.upper(), k))
            tokens.append(STRUCTURE_TOKENS[":"])
            tokens += tokenize_value(v)
            if i < len(val) - 1:
                tokens.append(STRUCTURE_TOKENS[","])
        tokens.append(STRUCTURE_TOKENS["}"])
        return tokens
    return [val]

def tokenize_cad_steps(cad_json):
    if isinstance(cad_json, str):
        cad = json.loads(cad_json)
    else:
        cad = cad_json

    if not isinstance(cad, dict):
        print("Warning: CAD steps is not a dict. Skipping this sample.")
        return []

    tokens = []
    timeline = cad.get("timeline", [])
    entities = cad.get("entities", {})

    for step in timeline:
        entity_id = step["entity"]
        entity = entities[entity_id]
        entity_type = entity.get("type", "").upper()
        tokens.append(BASE_TOKENS.get(entity_type, entity_type))
        tokens += tokenize_value(entity)
    return tokens

if __name__ == "__main__":
    # Example usage
    cad_json = {
        "timeline": [
            {"entity": "sketch_1", "index": 0},
            {"entity": "extrude_1", "index": 1}
        ],
        "entities": {
            "sketch_1": {
                "type": "Sketch",
                "curves": [],
                "constraints": {},
                "dimensions": {}
            },
            "extrude_1": {
                "type": "ExtrudeFeature",
                "profile": "<UUID>",
                "extent_one": {"distance": {"value": 10}},
                "taperAngle": {"value": 0},
                "operationType": "NewBody"
            }
        }
    }
    tokenized_steps = tokenize_cad_steps(cad_json)
    print(tokenized_steps)