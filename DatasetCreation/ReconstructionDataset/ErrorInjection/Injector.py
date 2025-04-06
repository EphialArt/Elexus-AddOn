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
                    print(f"Injecting error into entity {entity_id} with curve {random_curve_id}")
                    entity.setdefault("constraints", {}).update({f"error_{random.randint(1000, 9999)}": error})
                    return "Success"
            
            print(f"Failed to find a valid curve for entity {entity_id}.")
            return None
    return None
            

    
tyrone = { 'name': 'tyrone'}
error_types = ["over_constrained_sketch"]
data_with_errors = inject_errors(data, error_types)
save_json(data_with_errors, "C:/Users/iceri/OneDrive/Documents/GitHub/Elexus-AddOn/DatasetCreation/ReconstructionDataset/Results/modified_design.json")