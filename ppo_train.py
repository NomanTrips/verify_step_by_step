import torch
from transformers import GPT2Tokenizer
from prm_model import VerifyPRM, Config
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import load_dataset
from tqdm import tqdm
from peft import LoraConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ppo_config = PPOConfig(
            model_name="gpt2",
            learning_rate=1.41e-5,
            log_with=None,
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
    max_input_length = 904 # 1024 - 120 =  904. gpt2 context length minus max tokens to generate
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

dataset = build_dataset('./data/MATH_dataset_ids.json')

print(dataset)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# initialize trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset=dataset, data_collator=collator)

# load reward model
prm_config = Config()
prm_config.load_peft_model = True
prm_model = VerifyPRM(prm_config)
prm_model.load_state_dict(torch.load('./ckpt/phase2_1e-6_125461.pth'), strict=False)
for param in prm_model.parameters():
    param.requires_grad = False

from torch.nn.utils.rnn import pad_sequence

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    #print(f"query_tensors[0].shape: ", query_tensors[0].shape)
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 120,
    }
    #response_tensors = model.generate(query_tensors)
    #ref_response_tensors = model_ref.generate(query_tensors)
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    #batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # get reward scores for model under training (via prm model)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    encoding = prm_model.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    logits = prm_model.forward(input_ids, attention_mask)
    predicted_classes = torch.argmax(logits, dim=1) # Convert logits to predicted class labels (0, 1, 2) 
    class_mapping = {0: -1, 1: 0, 2: 1} # Define mapping from predicted class to actual label (-1, 0, 1)
    rewards = [class_mapping[cls.item()] for cls in predicted_classes] # Map each predicted class to the actual label
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # get reward scores for ref model
    #ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    #ref_encoding = prm_model.tokenizer(ref_texts, padding=True, truncation=True, return_tensors='pt')
    #ref_input_ids = ref_encoding['input_ids'].to(device)
    #ref_attention_mask = ref_encoding['attention_mask'].to(device)
    #ref_logits = prm_model.forward(ref_input_ids, ref_attention_mask)
    #ref_predicted_classes = torch.argmax(ref_logits, dim=1)
    #ref_rewards = [class_mapping[cls.item()] for cls in ref_predicted_classes]
    #batch["ref_rewards"] = ref_rewards

    #query_tensors_list = [query_tensors[i] for i in range(query_tensors.shape[0])] # ppo_trainer requires list of tensors               
    #response_tensors_list = [response_tensors[i] for i in range(response_tensors.shape[0])]

    stats = ppo_trainer.step(query_tensors, response_tensors, [rewards_tensor])
    #ppo_trainer.log_stats(stats, batch, rewards)