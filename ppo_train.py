import torch
from transformers import GPT2Tokenizer
from prm_model import VerifyPRM, Config
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import load_dataset
from tqdm import tqdm
from peft import LoraConfig
import wandb
wandb.init(project='ppo_verify', name='gpt2_test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ppo_config = PPOConfig(
            model_name="gpt2",
            learning_rate=1.41e-5,
            log_with="wandb",
            mini_batch_size=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
        )

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

device_map = {"": 0}

# load a model and ref model to train with ppo
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)

model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# build dataset
def build_dataset(query_dataset):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name or local path of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    ds = load_dataset("json", data_files=query_dataset, field="train", split="train")
    ds = ds.filter(lambda x: x["level"] == "Level 1", batched=False)
    max_input_length = 960 # 1024 - 64 =  960. gpt2 context length minus max tokens to generate
    def tokenize(sample):
        if 'problem' in sample:
            sample["input_ids"] = tokenizer.encode(sample["problem"])[: max_input_length]
            sample["query"] = tokenizer.decode(sample["input_ids"])
        else:
            print("Warning: 'problem' key not found in sample:", sample)
        return sample
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset('./data/MATH_dataset.json')

print(dataset)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# initialize trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset=dataset, data_collator=collator)

# load reward model
prm_config = Config()
prm_config.load_peft_model = False
prm_model = VerifyPRM(prm_config)
prm_model.load_state_dict(torch.load('./ckpt/phase2_1e-6_125461.pth'), strict=False)
for param in prm_model.parameters():
    param.requires_grad = False

from torch.nn.utils.rnn import pad_sequence

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 64,
    }
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # get reward scores for model under training (via prm model)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    print(texts)
    encoding = prm_model.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    logits = prm_model.forward(input_ids, attention_mask)
    predicted_classes = torch.argmax(logits, dim=1) # Convert logits to predicted class labels (0, 1, 2) 
    class_mapping = {0: -1, 1: 0, 2: 1} # Define mapping from predicted class to actual label (-1, 0, 1)
    rewards = [class_mapping[cls.item()] for cls in predicted_classes] # Map each predicted class to the actual label
    rewards_list = [torch.tensor([class_mapping[cls.item()]], dtype=torch.float32) for cls in predicted_classes]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
    ppo_trainer.log_stats(stats, batch, rewards_list)