import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
import subprocess

# login() 

device = 'cuda:0'
model_checkpoint = "mistralai/Mistral-7B-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, 
        device_map = 'auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        )

tokenizer.save_pretrained(f'mistral_dora_merged')
model_to_merge = PeftModel.from_pretrained(model, f"mistral_dora_final")
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(f'mistral_dora_merged')
subprocess.call(["lm_eval","--model_args",f"pretrained=mistral_dora_merged","--tasks","gsm8k","--device",device,"--batch_size","8"])
