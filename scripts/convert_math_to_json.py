import json
import os

def load_and_process_data(directory):
    data_structure = {"train": [], "test": []}
    for split in ["train", "test"]:
        split_path = os.path.join(directory, split)
        for subject in os.listdir(split_path):
            subject_path = os.path.join(split_path, subject)
            for file in os.listdir(subject_path):
                file_path = os.path.join(subject_path, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    content.update({"subject": subject})  # Add subject info
                    data_structure[split].append(content)
    return data_structure

def main():
    # Change this to the location of your MATH dataset
    directory = '/home/brian/Desktop/MATH/'
    data = load_and_process_data(directory)
    
    # Output to a JSON file for use with Hugging Face's datasets
    with open('./data/MATH_dataset.json', 'w') as json_file:
        json.dump(data, json_file)

if __name__ == "__main__":
    main()
