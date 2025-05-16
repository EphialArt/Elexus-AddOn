import json
import random

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

data = load_json("C:/Users/iceri/Downloads/r1.0.1/r1.0.1/reconstruction/20203_7e31e92a_0000.json")
entity_types = []
entities = data.get("entities", {})
timeline = data.get("timeline", {})
for entity in timeline:
    entity_id = entity.get("entity")
    entity_type = entities.get(entity_id, {}).get("type")
    entity_ = {"entity": entity_id, "type": entity_type}
    entity_types.append(entity_)

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def inject_errors(data, error_types, max_errors=1, max_retries=10):
    
    if not entities:
        print("No entities found in the data.")
        return data

    error_count = random.randint(1, max_errors)
    for _ in range(error_count):
        retries = 0
        while retries < max_retries:
            random_entity_id = random.choice(list(entities.keys()))
            random_entity = entities[random_entity_id]

            error_type = random.choice(error_types)

            run = apply_error(random_entity, error_type, random_entity_id)
            if run == "Success":
                print(_)
                break
            retries += 1
            print(f"Failure {retries}")
        if retries == max_retries:
            print(f"Failed to inject error after {max_retries} retries for entity {random_entity_id}.")

    return data

def apply_error(entity, error_type, entity_id=None):
    for x in entity_types:
        if x["entity"] == entity_id:
            entity_type = x["type"]
            break
    if error_type == "over_constrained_sketch":
        if entity.get("type") == "Sketch":
            curves = entity.get("curves", {})
            if not curves:
                print(f"No curves found for entity {entity_id}.")
                return None
            
            for _ in range(10):  
                random_curve_id = random.choice(list(curves.keys()))
                curve = curves[random_curve_id]
                if curve.get("type") == "SketchLine":
                    error = {'line': random_curve_id, 'type': 'HorizontalConstraint'}
                    print(f"Injecting 'overconstrained sketch' error into entity {entity_id} with curve {random_curve_id}")
                    entity.setdefault("constraints", {}).update({f"error_{random.randint(1000, 9999)}": error})
                    return "Success"
                    
            print(f"Failed to find a valid curve for entity {entity_id}.")
            return None
    
    elif error_type == "under_constrained_sketch":
        if entity.get("type") == "Sketch":
            constraints = entity.get("constraints", {})
            if not constraints:
                print(f"No constraints found for entity {entity_id}.")
                return None
            
            for _ in range(10): 
                random_constraint_id = random.choice(list(constraints.keys()))
                constraints.pop(random_constraint_id)
                print(f"Removed constrain with id {random_constraint_id} from entity {entity_id}")
                return "Success"
            
    elif error_type == "zero_thickness_walls":
        if entity.get("type") == "ExtrudeFeature":
            extent = entity.get("extent_one", {})
            if extent:
                distance = extent.get("distance", {})
                if distance:
                    distance["value"] = 0.00001
                    print(f"Injected 'zero thickness walls' into entity {entity_id} by setting distance to 0")
                    return "Success"	
        
    elif error_type == "duplicate_geometry":
        if entity_type == "ExtrudeFeature":
            last_index = max(item.get("index", 0) for item in timeline)
            
            new_timeline_entry = {
                "index": last_index + 1,
                "entity": entity_id
            }
            timeline.append(new_timeline_entry)
            
            print(f"Duplicated entity {entity_id} with new timeline index {last_index + 1}")
            return "Success"
        else:
            return None
    
    return None
            
error_types = ["zero_thickness_walls"]
data_with_errors = inject_errors(data, error_types)
save_json(data_with_errors, "C:/Users/iceri/OneDrive/Documents/GitHub/Elexus-AddOn/DatasetCreation/ReconstructionDataset/Results/modified_design.json")