import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
import subprocess

# login() 
parser = ArgumentParser()
parser.add_argument("--model", type=str, default='spectral')
args = parser.parse_args()

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

if args.model == 'original':
        model.save_pretrained('mistral_original')
        tokenizer.save_pretrained('mistral_original')
        subprocess.call(["lm_eval","--model_args","pretrained=mistral_original","--tasks","gsm8k","--device",device,"--batch_size","8"])
else:
        tokenizer.save_pretrained(f'mistral_{args.model}_merged')
        model_to_merge = PeftModel.from_pretrained(model, f"mistral_{args.model}_final")
        print(f'{args.model} in layer merge') 
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(f'mistral_{args.model}_merged')
        subprocess.call(["lm_eval","--model_args",f"pretrained=mistral_{args.model}_merged","--tasks","gsm8k","--device",device,"--batch_size","8"])
