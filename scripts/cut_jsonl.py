import os

def cut_jsonl_file(input_file_path, output_file_path, target_size_mb):
    target_size = target_size_mb * 1024 * 1024  # Convert MB to bytes
    current_size = 0

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                output_file.write(line)
                current_size += len(line.encode('utf-8'))
                
                # Stop if the file size is approximately the target size
                if current_size >= target_size:
                    break

# Replace 'your_input_file.jsonl' with the path to your .jsonl file
# Replace 'output_file.jsonl' with the path where you want to save the new file
cut_jsonl_file('./proofpile_dev.jsonl', 'proofpile.jsonl', 200)
