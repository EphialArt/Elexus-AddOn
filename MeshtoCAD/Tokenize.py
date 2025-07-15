import json
import re

BASE_TOKENS = {
    "SKETCH": 0, "EXTRUDEFEATURE": 1, "FILLET": 2, "CHAMFER": 3, "REVOLVE": 4, "SWEEP": 5, "LOFT": 6,
    "BOOLEAN": 7, "MOVE": 8, "COPY": 9, "MIRROR": 10, "PATTERN": 11, "FEATURE_SCRIPT": 12, "PROFILE": 13,
}
PARAM_TOKENS = {
    "NAME": 20, "PLANE": 21, "POINTS": 22, "CURVES": 23, "REFERENCE_VECTOR": 24, "LINE_ONE": 25, "LINE_TWO": 26,
    "MAX_POINT": 27, "START_POINT": 28, "END_POINT": 29, "CENTER_OF_MASS": 30, "RADIUS": 31, "DIAMETER": 32, "DEGREE": 33,
    "CONTROL_POINTS": 34, "IS_DRIVING": 35, "ORIENTATION": 36, "CONSTRAINTS": 37, "TYPE": 38, "ENTITIES": 39,
    "DIMENSIONS": 40, "ENTITY": 41, "VALUE": 42, "EXTENT_ONE": 43, "DISTANCE": 44, "TAPER_ANGLE": 45,
    "OPERATION_TYPE": 46, "FILLET_EDGES": 47, "CHAMFER_EDGES": 48, "DISTANCE_ONE": 49, "DISTANCE_TWO": 50,
    "ANGLE": 51, "AXIS": 52, "PROFILES": 53, "PROPERTIES": 54, "FACES": 55, "BODIES": 56, "TOOL_BODY": 57,
    "TARGET_BODY": 58, "TRANSFORM": 59, "VECTOR": 60, "AMOUNT": 61, "PARAMETER": 62, "PATTERN_TYPE": 63,
    "MIRROR_PLANE": 64, "SOURCE_FEATURES": 65, "ROLE": 66, "VISIBLE": 67, "FIXED": 68, "FULLY_CONSTRAINED": 69, 
    "REFERENCE_PLANE": 70, "X": 71, "Y": 72, "Z": 73, "CONSTRUCTION_GEOM": 74, "START_ANGLE": 75, "END_ANGLE": 76, 
    "START_EXTENT": 77, "CENTER_POINT": 78, "LENGTH": 79, "CURVE_ONE": 80, "CURVE_TWO": 81,
    "ENTITY_ONE": 82, "ENTITY_TWO": 83, "CURVE": 84, "PERIMETER": 85, "AREA": 86, "VOLUME": 87, "DENSITY": 88,
    "EXTRUDE_FACES": 89, "EXTRUDE_SIDE_FACES": 90, "EXTRUDE_BODIES": 91, "EXTRUDE_END_FACES": 92, "EXTRUDE_START_FACES": 93,
    "SURFACE_TYPE": 94, "POINT_ON_FACE": 95, "INDEX": 96, "EXTENT_TYPE": 97, "OPERATION": 98, "CONNECTIVE_TRANSFORM": 99,
    "NORMAL": 100, "U_DIRECTION": 101, "V_DIRECTION": 102, "PROFILE": 103, "SKETCH": 104, "LOOPS": 105,
    "IS_OUTER": 106, "PROFILE_CURVES": 107, "REFERENCE": 108, "TEXT_POSITION": 109, "ORIGIN": 110,
    "X_AXIS": 111, "Y_AXIS": 112, "Z_AXIS": 113, "CORRECTIVE_TRANSFORM": 114, "CENTROID": 115, "POINT": 116, "LINE": 117,
}

VALUE_TOKENS = {
    "PLANE_XY": 200, "PLANE_YZ": 201, "PLANE_ZX": 202,
    "CONSTRAINT_HORIZONTAL": 203, "CONSTRAINT_VERTICAL": 204, "CONSTRAINT_COINCIDENT": 205,
    "CONSTRAINT_TANGENT": 206, "CONSTRAINT_PARALLEL": 207, "CONSTRAINT_PERPENDICULAR": 208,
    "CONSTRAINT_EQUAL": 209, "CONSTRAINT_FIX": 210,
    "OPERATION_NEW_BODY": 211, "OPERATION_CUT": 212, "OPERATION_JOIN": 213, "OPERATION_INTERSECT": 214, "VECTOR3D": 215,
    "SKETCHCIRCLE": 216, "POINT3D": 217, "CENTROID": 218, "EXTRUDEFEATURE": 219, "NEWBODYFEATUREOPERATION": 220,
    "LOOPS": 221, "ORIGIN": 222, "SKETCH": 223, "SKETCHDIAMETERDIMENSION": 224, "BODY": 225, "BODIES": 226,
    "COINCIDENTCONSTRAINT": 227, "TEXT_POSITION": 228, "CONSTRUCTIONPLANE": 229, "PERPENDICULARCONSTRAINT": 230,
    "CIRCLE3D": 231, "PROFILEPLANEDEFINITION": 232, "PROFILES": 233, "PLANE": 234, "CYLINDERSURFACETYPE": 235,
    "PLANESURFACETYPE": 236, "PROFILEPLANESTARTDEFINITION": 237, "ONESIDEFEATUREEXTENTTYPE": 238,
    "ALONGDISTANCE": 239, "TAPERANGLE": 240, "DISTANCEEXTENTDEFINITION": 241, "REFERENCE": 242,
    "TRUE": 243, "FALSE": 244, "XZ": 245, "D1": 246, "D2": 247, "D3": 248, "BODY1": 249, "LINE3D": 250,
    "ARC3D": 251, "BREPFACE": 252, "MODELPARAMETER": 253, "SKETCHLINE": 254, "SKETCHARC": 255, "HORIZONTALCONSTRAINT": 256,
    "VERTICALCONSTRAINT": 257, "HORIZONTALDIMENSIONORIENTATION": 258, "VERTICALDIMENSIONORIENTATION": 259, "SKETCHLINEARDIMENSION": 260,
    "JOINFEATUREOPERATION": 261, "CUTFEATUREOPERATION": 262, "YX": 263, "LINE": 264, "POINT": 265, 
    "LINEAR DIMENSION-2": 266, "DIAMETER DIMENSION-2": 267, "LINEAR DIMENSION-3": 268, "DIAMETER DIMENSION-3": 269,
    "LINEAR DIMENSION-2": 266, "DIAMETER DIMENSION-2": 267, "LINEAR DIMENSION-3": 268, "DIAMETER DIMENSION-3": 269,
    "SKETCHARC": 270, "SKETCHCIRCLE": 271, "SKETCHCONICCURVE": 272, "SKETCHELLIPSE": 273,
    "SKETCHELLIPTICALARC": 274, "SKETCHFITTEDSPLINE": 275, "SKETCHFIXEDSPLINE": 276, "SKETCHLINE": 277,
    "CIRCULARPATTERNCONSTRAINT": 278, "COINCIDENTCONSTRAINT": 279, "COLLINEARCONSTRAINT": 280, "CONCENTRICCONSTRAINT": 281,
    "EQUALCONSTRAINT": 282, "HORIZONTALCONSTRAINT": 283, "HORIZONTALPOINTSCONSTRAINT": 284, "MIDPOINTCONSTRAINT": 285,
    "OFFSETCONSTRAINT": 286, "PARALLELCONSTRAINT": 287, "PERPENDICULARCONSTRAINT": 288, "POLYGONCONSTRAINT": 289,
    "RECTANGULARPATTERNCONSTRAINT": 290, "SMOOTHCONSTRAINT": 291, "SYMMETRYCONSTRAINT": 292, "TANGENTCONSTRAINT": 293,
    "VERTICALCONSTRAINT": 294, "VERTICALPOINTSCONSTRAINT": 295, "SKETCHANGULARDIMENSION": 296, "SKETCHCONCENTRICCIRCLEDIMENSION": 297,
    "SKETCHDIAMETERDIMENSION": 298, "SKETCHELLIPSEMAJORRADIUSDIMENSION": 299, "SKETCHELLIPSEMINORRADIUSDIMENSION": 300, "SKETCHLINEARDIMENSION": 301,
    "SKETCHOFFSETCURVESDIMENSION": 302, "SKETCHOFFSETDIMENSION": 303, "SKETCHRADIALDIMENSION": 304, "ANGULAR DIMENSION-2": 305,
    "ALIGNEDDIMENSIONORIENTATION": 306, "XY": 307, "TRUE" : 308, "FALSE": 309,

    "<FLOAT>": 310, "<INTEGER>": 311, "<UUID>": 312, "<BOOLEAN>": 313, "<NAME>": 314,
}

STRUCTURE_TOKENS = {"{": 400, "}": 401, "[": 402, "]": 403, ":": 404, ",": 405}

all_tokens = set(BASE_TOKENS.values()) | set(PARAM_TOKENS.values()) | set(VALUE_TOKENS.values()) | set(STRUCTURE_TOKENS.values())
VOCAB_SIZE = len(all_tokens)

UUID_PATTERN = re.compile(r"^[a-f0-9\-]{36}$", re.IGNORECASE)
UUIDS = {}

def tokenize_value(val, uuid_map=None, uuid_counter=None, key=None,):
    if isinstance(val, float):
        return [VALUE_TOKENS["<FLOAT>"], val]
    elif isinstance(val, bool):
        if int(val) == 0:
            return [VALUE_TOKENS["FALSE"]]
        elif int(val) == 1:
            return [VALUE_TOKENS["TRUE"]]
    elif isinstance(val, int):
        return [VALUE_TOKENS["<INTEGER>"], val]
    elif isinstance(val, str):
        if val.upper() in VALUE_TOKENS:
            return [VALUE_TOKENS[val.upper()]]
        elif val.upper() in PARAM_TOKENS:
            return [PARAM_TOKENS[val.upper()]]
        elif UUID_PATTERN.match(val):
            if val not in uuid_map:
                placeholder = f"<UUID_{uuid_counter[0]}>"
                uuid_map[val] = placeholder
                VALUE_TOKENS[placeholder] = uuid_counter[0]
                uuid_counter[0] += 1
            return [VALUE_TOKENS["<UUID>"], VALUE_TOKENS[uuid_map[val]]]
        else:
            return [VALUE_TOKENS["<NAME>"]]
    elif isinstance(val, list):
        tokens = [STRUCTURE_TOKENS["["]]
        for item in val:
            tokens.extend(tokenize_value(item, uuid_map=uuid_map, uuid_counter=uuid_counter))
            tokens.append(STRUCTURE_TOKENS[","])
        if len(tokens) > 1:
            tokens.pop() 
        tokens.append(STRUCTURE_TOKENS["]"])
        return tokens
    elif isinstance(val, dict):
        tokens = [STRUCTURE_TOKENS["{"]]
        for k, v in val.items():
            tokens.extend(tokenize_value(k, uuid_map=uuid_map, uuid_counter=uuid_counter))
            tokens.append(STRUCTURE_TOKENS[":"])
            tokens.extend(tokenize_value(v, uuid_map=uuid_map, uuid_counter=uuid_counter))
            tokens.append(STRUCTURE_TOKENS[","])
        if len(tokens) > 1:
            tokens.pop()  
        tokens.append(STRUCTURE_TOKENS["}"])
        return tokens
    else:
        return []

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

    uuid_map = {}
    uuid_counter = [0]

    for step in timeline:
        entity_id = step["entity"]
        entity = entities[entity_id]
        entity_type = entity.get("type", "").upper()
        tokens.append(BASE_TOKENS.get(entity_type, entity_type))
        tokens += tokenize_value(entity, uuid_map=uuid_map, uuid_counter=uuid_counter)

    tokens.append(501)
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
                "name": "Sketch 1",
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
    print(VOCAB_SIZE)