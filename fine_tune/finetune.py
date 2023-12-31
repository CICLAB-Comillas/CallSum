import torch
import wandb
import numpy as np
import json

from huggingface_hub import login, logout
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from datetime import datetime

### ----------------------------------FUNCTIONS---------------------------------------------
def preprocess_function(examples):
    inputs = [doc for doc in examples["Transcripcion"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Resumen"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


### ----------------------------------MAIN---------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

with open("secrets.json", "r") as f:
    SECRETS = json.load(f)

MODEL_CHECKPOINT = "philschmid/bart-large-cnn-samsum"

# LOADING THE DATASET

train_datasets = load_dataset("CICLAB-Comillas/calls_10k_v1",sep=";")
eval_datasets = load_dataset("Jatme26/test-conv-dataset",sep=";")


# PREPROCESSING THE DATA

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

max_input_length = 1024
max_target_length = 256

tokenized_train_datasets = train_datasets.map(preprocess_function, batched=True)
tokenized_eval_datasets = eval_datasets.map(preprocess_function, batched=True)


# FINE-TUNING THE MODEL

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# Definition of Training Arguments
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 10000


wandb.login(key=SECRETS["wandb"])
wandb.init(
    # Project name
    project="bart",
    # Project id
    id=datetime.now().strftime('%d%m%y%H%M'),
    # Run name
    name=datetime.now().strftime('%d/%m/%y %H:%M'),
    # Metadata info
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "BART",
        "dataset": "calls10k_1",
        "notes": "First training try of the model",
    })

args = Seq2SeqTrainingArguments(
    evaluation_strategy = "steps",
    save_strategy="steps",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=30,
    max_steps=TRAIN_STEPS,
    logging_steps=5,
    eval_steps=250,
    save_steps=1000,
    output_dir="test-calls-summarization",
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train_datasets["train"],
    eval_dataset=tokenized_eval_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

trainer.evaluate()

wandb.alert(
    title = "Training finished",
    text = "Training was completed correctly",
    level = wandb.AlertLevel.INFO
)
wandb.finish()

login(token = SECRETS["huggingface"])
model.push_to_hub("CICLAB-Comillas/BARTola", use_auth_token=True)
logout()
