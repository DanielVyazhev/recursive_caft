import logging
import ast
import pandas as pd
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling

from core.utils.prepare_dataset import prepare_dataset
from core.prompts.mmlu_branches_sft import branch_a_sys_prompt, branch_a_user_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "data/out/sft_data/branch_a_train.parquet"
OUTPUT_DIR = "data/out/models/branch_a"
EPOCHS = 3
LR = 1e-5
GLOBAL_BATCH_SIZE = 256
PER_DEVICE_BATCH_SIZE = 16

logging.info(f"Loading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

logging.info(f"Loading data: {DATA_FILE}")
df = pd.read_parquet(DATA_FILE)
logging.info(f"Loaded {len(df)} samples")

def get_sys_prompt(row):
    return branch_a_sys_prompt(row["subject"] if row["subject"] else None)

def get_user_prompt(row):
    options = ast.literal_eval(row["options"])
    return branch_a_user_prompt(row["question"], options)

def prepare_branch_a_dataset(tokenizer, df):
    """Prepare Branch A dataset with thinking + answer"""
    df_copy = df.copy()
    df_copy["sys_prompt"] = df_copy.apply(get_sys_prompt, axis=1)
    df_copy["user_prompt"] = df_copy.apply(get_user_prompt, axis=1)
    
    def process_row(row):
        messages = [
            {"role": "system", "content": row["sys_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
            {"role": "assistant", "content": f"{row['thinking']}\n\nAnswer: {row['answer_letter']}"}
        ]
        tokenized = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()
        }
    
    from datasets import Dataset
    dataset = Dataset.from_pandas(df_copy)
    processed = dataset.map(process_row, num_proc=4, remove_columns=dataset.column_names)
    return processed

train_ds = prepare_branch_a_dataset(tokenizer, df)
logging.info(f"Prepared {len(train_ds)} training samples")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

gradient_accumulation_steps = GLOBAL_BATCH_SIZE // PER_DEVICE_BATCH_SIZE
logging.info(f"Batch config: per_device={PER_DEVICE_BATCH_SIZE}, grad_accum={gradient_accumulation_steps}, global={GLOBAL_BATCH_SIZE}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=LR,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    lr_scheduler_type="linear",
    warmup_steps=50,
    seed=42,
    data_seed=42,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=data_collator,
    processing_class=tokenizer,
)

logging.info("Starting training...")
trainer.train()

logging.info(f"Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logging.info("Done!")
