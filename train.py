import torch
from torch.cuda.amp import autocast
from prm_model import VerifyPRM, Config
from data_loading import *
from torch.optim.lr_scheduler import OneCycleLR
import bitsandbytes as bnb
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser(description='Training settings')
args = parser.parse_args()
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')

import wandb
wandb.init(project='prm_800k', name='1e-6')

def calculate_metrics(preds, labels):
    """
    Calculate accuracy, precision, recall, and F1 score.
    Args:
        preds: Predictions from the model.
        labels: Actual labels.
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Convert predictions to binary using argmax and convert to numpy
    predicted_classes = preds.argmax(dim=1).cpu().numpy()
    true_classes = labels.cpu().numpy()

    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model(model, train_dataloader, num_epochs=1):
    optimizer = bnb.optim.AdamW32bit(model.parameters(), lr=1e-6, betas=(0.9, 0.995), is_paged=True)   
    warmup_ratio = 0.3
    #if model.config.load_peft_model == True:
    #    model.llm.model.model.gradient_checkpointing_enable() # peft requires deeper call
    #else:
    #    model.llm.model.gradient_checkpointing_enable()

    total_steps = len(train_dataloader) * num_epochs
    scheduler = OneCycleLR(optimizer, max_lr=1e-5, total_steps=total_steps, # 2e-3
                       pct_start=warmup_ratio, anneal_strategy='cos')
    criterion = torch.nn.CrossEntropyLoss()
    step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        running_train_loss = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            
            texts, ratings = batch
            labels = ratings.to(device)
            encoding = model.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            max_length = 1024  # so no cuda oom, adjust if gpu rich
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

            with autocast(dtype=torch.bfloat16): # bf16 more stable?
                outputs = model.forward(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # 0.3
            optimizer.step()
            scheduler.step()            
    
            train_loss += loss.item()
            running_train_loss += loss.item()

            # Periodic metrics/save
            if step >1 and step % 50 == 0:
                avg_loss = running_train_loss / 50
                metrics = calculate_metrics(outputs, labels)
                acc = metrics['accuracy']
                prec = metrics['precision']
                f1 = metrics['f1']
                
                print(f"Step {step}/{total_steps} - Avg loss: {avg_loss} - acc: {acc} - precision: {prec} - f1: {f1}")
                running_train_loss = 0.0
                wandb.log({"step": step, "epoch": epoch, "avg_loss": avg_loss, "acc": acc, "precision": prec, "f1": f1})
            if step > 1 and step % 2000 == 0:
                model.llm.save_pretrained(f"./adapters/phase2_1e-6_{step}")
                layers_to_save = ['classifier']
                selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
                torch.save(selected_state_dict, f'./ckpt/phase2_1e-6_{step}.pth')

            step += 1

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Step {step}/{total_steps} - Avg Train Loss: {avg_train_loss}")
        model.llm.save_pretrained(f"./adapters/phase2_1e-6_{step}")
        layers_to_save = ['classifier']
        selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
        torch.save(selected_state_dict, f'./ckpt/phase2_1e-6_{step}.pth')

config = Config()

json_file = '/home/brian/Desktop/prm800k/prm800k/data/phase2_train.jsonl'
train_dataset = MathProblemDataset(json_file)
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
config.load_peft_model = True
model = VerifyPRM(config)
model.load_state_dict(torch.load('./ckpt/prm_800k_phase1_6187_first.pth'), strict=False)
for param in model.classifier.parameters():
    param.requires_grad = True

wandb.watch(model)
train_model(model, train_dataloader, num_epochs=1)
	