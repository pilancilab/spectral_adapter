import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from argparse import ArgumentParser

"""login for model download permission"""
# login() 

parser = ArgumentParser()
parser.add_argument("--model", type=str, default='full')
args = parser.parse_args()

torch.manual_seed(0)
model_checkpoint = "meta-llama/Meta-Llama-3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if args.model != 'full':
    peft_config = LoraConfig(
            r=8,  
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj","down_proj","gate_proj"],
            spectral_top=True if args.model == 'spectral' else False
        )

model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, 
        device_map = 'auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        )


if args.model != 'full':
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
for n,p in model.named_parameters():
    print(n,p.shape)


dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=0)
dataset = dataset.select(range(0,10000))
data = dataset.to_pandas()
data["text"] = data[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data)

def tokenize(sample):
    model_inps =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return model_inps

tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

batch_size = 8
training_arguments = TrainingArguments(
    output_dir=f"llama3-{args.model}",
    save_strategy = "steps",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    save_steps=100,
    weight_decay=0.1,
    logging_steps=1,
    num_train_epochs=1,
    push_to_hub=False,
    seed = 0,
    lr_scheduler_type='constant'
)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()