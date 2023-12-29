import json

# Load your JSON file
file_path = './MATH_dataset.json'  # Change this to the path of your JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to add IDs to each example in a given dataset part
def add_ids(dataset_part):
    for i, example in enumerate(dataset_part):
        # Assuming you want to start ids from 1 and increment
        example['id'] = str(i + 1)

# Check and add ids for 'train' and 'test' parts if they exist
if 'train' in data:
    add_ids(data['train'])

if 'test' in data:
    add_ids(data['test'])

# Save the modified data back into a new JSON file
output_file_path = './MATH_dataset_ids.json'  # Change this to your desired output file name
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)  # 'indent=4' for pretty printing, remove or adjust as needed
