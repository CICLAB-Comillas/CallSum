from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import torch
from datasets import load_dataset
import pandas as pd
import json
from huggingface_hub import login, logout
import wandb


### ----------------------------------FUNCIONES---------------------------------------------
def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

### ----------------------------------MAIN---------------------------------------------

if __name__ == "__main__":

    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    df = pd.read_csv("dataset10.csv", sep = ';', encoding = 'utf-8')

    dataset_data = [
        {
            "instruction": f"Genera un resumen {row_dict['Extension']} de esta conversación", # TO DO: probar con variaciones del prompt
            "input": row_dict["Transcripcion"],
            "output": row_dict["Resumen"]
        }
        for row_dict in df.to_dict(orient='records')
    ]

    with open("alpaca-conv-summary-dataset.json", "w") as f:
        json.dump(dataset_data, f)

    # Alpaca LoRA
    BASE_MODEL = "decapoda-research/llama-7b-hf"

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True, # In a 8-bit tensor to save space and speed up the training process
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    data = load_dataset("json", data_files="alpaca-conv-summary-dataset.json")

    CUTOFF_LEN = 512

    train_val = data["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 100
    OUTPUT_DIR = "experiments"

    wandb.login(key='ceb9a7d70eb557ab0c262e219d2d1a19575dbf1f')
    wandb.init(
        # Nombre del proyecto
        project="alpaca", 
        # Nombre de la ejecución
        name=f"run-1",
        # Información de hiperparámetros y metadatos
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Alpaca",
            "dataset": "calls10k_1",
            "notes": "Primer intento de entrenamiento del modelo con 10 conversaciones",
        })
    
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training
    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=30,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1,
        save_steps=1,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb"
    )

    # For the batches
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    model = torch.compile(model)

    trainer.train()
    wandb.finish()
    wandb.alert(
        title = "Entrenamiento terminado",
        text = "El entrenamiento ha terminado correctamente",
        level = wandb.AlertLevel.INFO
    )
    model.save_pretrained(OUTPUT_DIR)

    login(token = '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')
    model.push_to_hub("CICLAB-Comillas/AlpaCalls", use_auth_token=True)
    logout()