import json
import random

def split_dataset(json_file_path, output_file_path, train_ratio=0.95):
    """
    Split a JSON file into train and test sets and save as a new JSON file.

    :param json_file_path: Path to the original JSON file
    :param output_file_path: Path where the split JSON file will be saved
    :param train_ratio: Ratio of the train set size to the total dataset (default is 0.95)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Shuffle the data
        random.shuffle(data)

        # Calculate the number of examples in the train set
        num_train_examples = int(len(data) * train_ratio)

        # Split the data into train and test sets
        train_data = data[:num_train_examples]
        test_data = data[num_train_examples:]

        # Create a dictionary for train and test sets
        split_data = {
            'train': train_data,
            'test': test_data
        }

        # Write the split data to a new file
        with open(output_file_path, 'w') as file:
            json.dump(split_data, file, indent=4)

        return "Successfully split the dataset into train and test sets."
    except Exception as e:
        return f"An error occurred: {e}"

# Replace these paths with your actual file paths
json_file_path = '../data/verify_pretrain.json'
output_file_path = '../data/verify_pretrain_split.json'

# Run the function to split the dataset
result = split_dataset(json_file_path, output_file_path)
print(result)
