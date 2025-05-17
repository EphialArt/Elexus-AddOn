import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_fusion_360_data(file_path):
    """
    Loads Fusion 360 design data from a JSON file and converts it into a structured Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary representing the design data, with keys 'timeline' and 'entities'.
              Returns None on error.
    """
    try:
        data = load_json(file_path)
        design_data = {
            "timeline": data.get("timeline", []),
            "entities": data.get("entities", {})
        }
        return design_data
    except Exception as e:
        print(f"Error loading or processing JSON: {e}")
        return None

def process_entity(entity_id, entity_data):
    """
    Processes a single entity from the 'entities' dictionary.  This function
    can be expanded to handle different entity types.

    Args:
        entity_id (str): The ID of the entity.
        entity_data (dict): The data associated with the entity.

    Returns:
        dict: A simplified representation of the entity.
    """
    entity_type = entity_data.get("type")
    processed_entity = {"id": entity_id, "type": entity_type}

    if entity_type == "Sketch":
        processed_entity["curves"] = entity_data.get("curves", {})
        processed_entity["constraints"] = entity_data.get("constraints", {})
        processed_entity["dimensions"] = entity_data.get("dimensions", {})

    elif entity_type == "ExtrudeFeature":
        processed_entity["profile"] = entity_data.get("profile")
        processed_entity["distance"] = entity_data.get("extent_one", {}).get("distance", {}).get("value")
        processed_entity["taper"] = entity_data.get("taperAngle", {}).get("value")
        processed_entity["operation_type"] = entity_data.get("operationType")

    return processed_entity

def process_design_data(design_data):
    """
    Processes the loaded design data to create a more usable structure.

    Args:
        design_data (dict): The dictionary returned by load_fusion_360_data.

    Returns:
        dict: A processed dictionary with a more convenient structure.
    """
    processed_data = {
        "timeline": [],
        "entities": {}
    }

    for timeline_entry in design_data["timeline"]:
        entity_id = timeline_entry.get("entity")
        if entity_id in design_data["entities"]:
            entity_data = design_data["entities"][entity_id]
            processed_entity = process_entity(entity_id, entity_data)  # Use the helper function
            processed_data["entities"][entity_id] = processed_entity
            processed_data["timeline"].append({"entity_id": entity_id, "index": timeline_entry.get("index")}) #keep index
        else:
            print(f"Warning: Entity ID '{entity_id}' not found in entities dictionary.")

    return processed_data

if __name__ == '__main__':
    # Example Usage
    json_file_path = "C:/Users/iceri/Downloads/r1.0.1/r1.0.1/reconstruction/20203_7e31e92a_0000.json"  # Replace with your file path
    loaded_data = load_fusion_360_data(json_file_path)
    if loaded_data:
        processed_data = process_design_data(loaded_data)
        print(json.dumps(processed_data, indent=2)) #pretty print
