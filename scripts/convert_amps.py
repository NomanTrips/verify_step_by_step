import os
import json
import uuid
import sys

def process_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            problem = data.get("problem", "")
            hints = data.get("hints", [])
            text = f"Problem: {problem}\n"
            text += f"Solution:\n"
            for i, hint in enumerate(hints):
                if hint:  # Only add non-empty hints
                    text += f"Step [{i + 1}]: {hint}\n"

            # Create the new JSON structure
            new_data = {
                "text": text,
                "meta": {
                    "set_name": "AMPS",
                    "score": None,
                    "question_id": str(uuid.uuid4())
                }
            }
            return new_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None

def process_json_file_v2(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            problem = data.get("problem", "")
            hints = data.get("hints", [])
            text = f"Problem: {problem}\n"
            text += f"Solution:\n"

            # Adjust hint numbering if the first hint is empty
            hint_number = 1
            for hint in hints[:-1]:  # Exclude last hint (considered as Answer)
                if hint:  # Only add non-empty hints
                    text += f"Hint {hint_number}: {hint}\n"
                    hint_number += 1
            
            # Add the last hint as the Answer with 'Solution' prefix
            answer = hints[-1] if hints else ""
            text += f"Answer: {answer}\n"

            # Create the new JSON structure
            new_data = {
                "text": text,
                "meta": {
                    "set_name": "AMPS",
                    "score": None,
                    "question_id": str(uuid.uuid4())
                }
            }
            return new_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None

def process_folder(root_folder, output_folder):
    file_count = 0
    output_file = None
    output_file_size = 0

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                new_data = process_json_file_v2(file_path)

                if new_data:
                    if output_file is None or output_file_size >= 157286400:  # 150 MB in bytes
                        if output_file is not None:
                            output_file.close()
                        output_file_name = f"output_{file_count}.jsonl"
                        output_file_path = os.path.join(output_folder, output_file_name)
                        output_file = open(output_file_path, 'w')
                        file_count += 1
                        output_file_size = 0

                    json_line = json.dumps(new_data)
                    output_file.write(json_line + "\n")
                    output_file_size += len(json_line) + 1  # +1 for the newline character

    if output_file is not None:
        output_file.close()

# Example usage:
process_folder("/home/brian/Desktop/amps/khan/", "/home/brian/Desktop/verify_step_by_step")
