import torch 
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as  np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, default='spectral')
parser.add_argument("--task", type=str, default='cola')
args = parser.parse_args()

if args.model=='lora':
    lr = {'mnli':1e-4,'sst2':6e-5,'mrpc':2e-4,'cola':1e-4,
          'qnli':1e-4,'qqp':1e-4,'rte':2e-4,'stsb':2e-4}
elif args.model=='spectral':
    lr = {'mnli':1e-4,'sst2':1e-4,'mrpc':7e-4,'cola':3e-4,
          'qnli':1e-4,'qqp':1e-4,'rte':3e-4,'stsb':5e-4}
epoch = {'mnli':1,'sst2':5,'mrpc':13,'cola':8,
         'qnli':1,'qqp':10,'rte':10,'stsb':30}

model_checkpoint = "microsoft/deberta-v3-base"
batch_size = 32
torch.manual_seed(0)
peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=24, 
        lora_alpha=32, 
        lora_dropout=0.01,
        target_modules = ["query_proj", "key_proj", "value_proj"],
        spectral_top=True if args.model=='spectral' else False
    )
task = args.task
print(f'Model: {args.model}    Task: {args.task}')
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
encoded_dataset = dataset.map(preprocess_function, batched=True)
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,device_map='cuda:0')
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "exact_match"

args = TrainingArguments(
            f"spectral_lora_tune_{args.model}_{args.task}",
            evaluation_strategy = "steps",
            eval_steps = 500,
            learning_rate=lr[task],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch[task],
            weight_decay=0.01,
            load_best_model_at_end=True,
            do_eval = True,
            logging_strategy="no",
            seed=0
        )
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions
    return metric.compute(predictions=predictions, references=labels)
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,  
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
print('task name: ', task, ' eval result: ', trainer.evaluate())




