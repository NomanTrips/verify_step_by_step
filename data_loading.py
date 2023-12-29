import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llama_2_path = "/home/brian/Desktop/llama_2_7b_hf/"

class MathProblemDataset(Dataset):
    def __init__(self, jsonl_file):
        self.flattened_data = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                item = json.loads(line)
                problem_text = "Problem: " + item['question']['problem']
                for step in item['label']['steps']:
                    if step['completions'] is not None:
                        step_text = "\nSolution:\n " 
                        for completion in step['completions']:
                            step_text += completion['text'] + " "
                            combined_text = problem_text + step_text.strip()
                            self.flattened_data.append((combined_text, completion['rating']))

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        concatenated_text, rating = self.flattened_data[idx]
        return concatenated_text, rating

def map_label(label):
    #print(label)
    return label + 1 if label is not None else 2  # Map -1, 0, 1 to 0, 1, 2 and None to 2

def collate_fn(batch):
    texts, ratings = zip(*batch)
    processed_texts = ["[CLS] " + text for text in texts]
    mapped_ratings = torch.tensor([map_label(rating) for rating in ratings], dtype=torch.long)
    # Tokenization and padding can be done here if required
    return processed_texts, mapped_ratings

