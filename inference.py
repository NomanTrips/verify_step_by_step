import torch
from prm_model import VerifyPRM, Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the state dictionary from the file
state_dict = torch.load('./ckpt/phase2_1e-6_6000.pth')

desired_keys = ["classifier.weight", "classifier.bias"]

selected_layers = {k: v for k, v in state_dict.items() if k in desired_keys}

print(f"selected_layers.len: ", len(selected_layers))
 
prm_config = Config()
prm_config.load_peft_model = False
prm_model = VerifyPRM(prm_config)
# Now, load the classifier layer into the model
prm_model.load_state_dict(selected_layers, strict=False)
for param in prm_model.parameters():
    param.requires_grad = False

prompt = "Problem: \text{If you have } 8 \text{ apples in one container and } 15 \text{ apples in another container, how many apples do you have in total?} Solution: \text{Total apples} = 8 + 15 = 6"

encoding = prm_model.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
logits = prm_model.forward(input_ids, attention_mask)
predicted_classes = torch.argmax(logits, dim=1) # Convert logits to predicted class labels (0, 1, 2) 
class_mapping = {0: -1, 1: 0, 2: 1} # Define mapping from predicted class to actual label (-1, 0, 1)
rewards = [class_mapping[cls.item()] for cls in predicted_classes] # Map each predicted class to the actual label

print("Rewards:", rewards)