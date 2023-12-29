import json

# Function to update the problem values in the dataset
def update_problems(data):
    for item in data["train"]:
        item["problem"] = "Problem: " + item["problem"] + " Solution: "
    return data

def main():
    # Reading the data from the input JSON file
    with open('../data/MATH_dataset_ids.json', 'r') as file:
        data = json.load(file)

    # Updating the "problem" values in the dataset
    updated_data = update_problems(data)

    # Writing the modified data to a new JSON file
    with open('../data/MATH_dataset.json', 'w') as file:
        json.dump(updated_data, file, indent=4)

if __name__ == "__main__":
    main()
