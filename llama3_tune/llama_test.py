import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from argparse import ArgumentParser
import subprocess

device = "cuda:0"
parser = ArgumentParser()
parser.add_argument("--model", type=str, default='full')
parser.add_argument("--checkpoint", type=int, default=100)
args = parser.parse_args()
if args.model=='full':  
    print(f'TEST FULL TUNING checkpoint {args.checkpoint}')
    model = AutoModelForCausalLM.from_pretrained(
        f"llama3-full/checkpoint-{args.checkpoint}", 
        device_map = 'auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B" , trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"full-{args.checkpoint}/")
    model.save_pretrained(f"full-{args.checkpoint}")
    subprocess.call(["lm_eval","--model_args",f"pretrained=full-{args.checkpoint}","--tasks","gsm8k","--device",device,"--batch_size","8"])
elif args.model=='lora':  
    print(f'TEST LORA TUNING checkpoint {args.checkpoint}')
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map = 'auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B" , trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"lora-{args.checkpoint}/")
    model_to_merge = PeftModel.from_pretrained(model, f"llama3-lora/checkpoint-{args.checkpoint}")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(f"lora-{args.checkpoint}")
    subprocess.call(["lm_eval","--model_args",f"pretrained=lora-{args.checkpoint}","--tasks","gsm8k","--device",device,"--batch_size","8"])
elif args.model=='spectral':  
    print(f'TEST SPECTRAL TUNING checkpoint {args.checkpoint}')
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map = 'auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B" , trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"spectral-{args.checkpoint}/")
    model_to_merge = PeftModel.from_pretrained(model, f"llama3-spectral/checkpoint-{args.checkpoint}")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(f"spectral-{args.checkpoint}")
    subprocess.call(["lm_eval","--model_args",f"pretrained=spectral-{args.checkpoint}","--tasks","gsm8k","--device",device,"--batch_size","8"])
