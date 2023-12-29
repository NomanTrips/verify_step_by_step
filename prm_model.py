import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaModel
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import bitsandbytes as bnb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="SEQUENCE_CLASSIFICATION",
    target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)

# Load the entire model on the GPU 0
device_map = {"": 0}

# pretrained llama 2 on math problems
llama_2_path = './ckpt/prm_phase2' #"./ckpt/prm_phase1" #"./ckpt/verify_prm_pretrain"
tokenizer_path = "/home/brian/Desktop/llama_2_7b_hf/" # llama 2 tokenizer

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
    
class VerifyPRM(nn.Module):
    def __init__(self, config):
        super(VerifyPRM, self).__init__()
        
        # llama 2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.new_tokens = ["[CLS]"] # Classification token
        self.num_added_tokens = self.tokenizer.add_tokens(self.new_tokens)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # Initialize llama 2
        self.llm = self.init_llm(config)
        self.config = config

        # Classification head
        self.classifier = nn.Linear(self.config.llama2_dim, 3).to(device)  # 3 classes
    
    def init_llm(self, config):
        model = LlamaModel.from_pretrained(
            llama_2_path,
            quantization_config=bnb_config,
            device_map=device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.resize_token_embeddings(len(self.tokenizer)) # resize embedding layer bc of added tokens for CLS and <pad> tokens
        if config.load_peft_model:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config, 'verify_prm')
            model.print_trainable_parameters()
        model.eval()
        return model

    def forward(self, input_ids, attention_mask):
        #print(f"input_ids: ", input_ids)
        outputs = self.llm.forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # The hidden states

        cls_representation = hidden_states[:, 0, :]
        logits = self.classifier(cls_representation)

        return logits

class Config:
    def __init__(self):
        self.text_embedding_dim = 4096  # llama 2 embedding dimension size
        self.llama2_dim = 4096  # llama 2 feature dimension size
        self.load_peft_model = False