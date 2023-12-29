import json
import random

def combine_and_shuffle_jsonl(file1_path, file2_path, output_file_path):
    all_data = []

    # Read the first file and append each line to the list
    with open(file1_path, 'r', encoding='utf-8') as file1:
        for line in file1:
            all_data.append(json.loads(line.strip()))

    # Read the second file and append each line to the list
    with open(file2_path, 'r', encoding='utf-8') as file2:
        for line in file2:
            all_data.append(json.loads(line.strip()))

    # Shuffle the combined data
    random.shuffle(all_data)

    # Write the shuffled data to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in all_data:
            output_file.write(json.dumps(entry) + '\n')

# Replace these with your actual file paths
file1_path = './output_0.jsonl'
file2_path = './proofpile.jsonl'
output_file_path = 'verify_pretrain.jsonl'

combine_and_shuffle_jsonl(file1_path, file2_path, output_file_path)
