import json

def replace_question_id_with_integer(json_file_path, output_file_path):
    """
    Replace the 'question_id' in each JSON object with an integer starting at 1.

    :param json_file_path: Path to the JSON file
    :param output_file_path: Path where the modified JSON file will be saved
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Replace 'question_id' with an integer
        for i, entry in enumerate(data, start=1):
            if 'meta' in entry and 'question_id' in entry['meta']:
                entry['meta']['question_id'] = i

        # Write the modified data to a new file
        with open(output_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        return "Successfully replaced 'question_id' with integers."
    except Exception as e:
        return f"An error occurred: {e}"

# Specify the path to your JSON file and the output file
json_file_path = '../data/verify_pretrain.json'  # Replace with your file path
output_file_path = '../data/verify_pretrain_renum.json'  # Replace with your desired output file path

# Run the function
result = replace_question_id_with_integer(json_file_path, output_file_path)
print(result)
