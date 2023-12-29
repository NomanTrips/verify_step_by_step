import json

def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    """
    Convert a .jsonl file to a .json file.

    :param jsonl_file_path: Path to the .jsonl file
    :param json_file_path: Path where the .json file will be saved
    """
    try:
        # Open the .jsonl file and read line by line
        with open(jsonl_file_path, 'r') as file:
            jsonl_data = [json.loads(line) for line in file if line.strip()]

        # Write the data to a .json file
        with open(json_file_path, 'w') as json_file:
            json.dump(jsonl_data, json_file, indent=4)

        print("Conversion successful.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
jsonl_file_path = '../data/verify_pretrain.jsonl'  # Replace with your .jsonl file path
json_file_path = '../data/verify_pretrain.json'  # Replace with your desired output path

convert_jsonl_to_json(jsonl_file_path, json_file_path)
