#!/usr/bin/env python3

# ----------------------
# IMPORT LIBRARIES
# ----------------------

import sys
import os
from pathlib import Path
import boto3

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

# ----------------------
# CONFIGURATION
# ----------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "./qlora_model_outputs"
S3_BUCKET = sys.argv[1] if len(sys.argv) > 1 else "default-qlora-s3-bucket"

print(f"Using S3 bucket: {S3_BUCKET}")

# ----------------------
# LOAD DATASET
# ----------------------

dataset = load_dataset(
    "json",
    data_files={
        "train": "../data/llm/train/*.json",
        "validation": "../data/llm/validate/*.json"
    }
)

# ----------------------
# PROMPT FORMATTING
# ----------------------

def format_prompt(instruction, input_text, output_text):
    """
    Flatten instruction/input/output into a single string per example.
    TinyLlama benefits from instruction-response style.
    """
    if input_text:
        return f"{instruction}\n{input_text}\n\n{output_text}"
    else:
        return f"{instruction}\n\n{output_text}"

# ----------------------
# TOKENIZER
# ----------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# BATCH-SAFE TOKENIZATION FUNCTION
# ----------------------

def tokenize_function(batch):
    """
    Convert a batch of examples into token IDs for TinyLlama.
    Also sets labels=input_ids for causal LM.
    """
    prompts = [
        format_prompt(batch["instruction"][i],
                      batch["input"][i],
                      batch["output"][i])
        for i in range(len(batch["instruction"]))
    ]

    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=1024
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ----------------------
# MODEL LOADING WITH QLoRA + 4-BIT
# ----------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False

# ----------------------
# TRAINING ARGUMENTS
# ----------------------

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=5,
    report_to="none"
)

# ----------------------
# CALLBACK FOR LIVE S3 CHECKPOINT UPLOAD
# ----------------------

class S3CheckpointCallback(TrainerCallback):
    """
    Upload checkpoints to S3 immediately after saving.
    Compatible with transformers >=5.
    """
    def __init__(self, bucket_name, s3_prefix="qlora_checkpoints"):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3")

    def on_save(self, args, state, control, **kwargs):
        # In newer transformers, checkpoints are saved as output_dir/checkpoint-<global_step>
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if not os.path.exists(checkpoint_dir):
            # Sometimes the checkpoint may not exist if save_strategy="steps" hasn't triggered yet
            return

        print(f"Uploading checkpoint {checkpoint_dir} to S3...")

        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, checkpoint_dir)
                s3_path = f"{self.s3_prefix}/{os.path.basename(checkpoint_dir)}/{rel_path}"
                self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
                print(f"Uploaded {local_path} → s3://{self.bucket_name}/{s3_path}")


# ----------------------
# RESUME FROM LATEST LOCAL OR S3 CHECKPOINT
# ----------------------

def download_latest_s3_checkpoint(bucket, prefix="qlora_checkpoints", local_output="./qlora_model_outputs"):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        print("No S3 checkpoints found.")
        return None

    checkpoints = sorted(
        [obj["Key"] for obj in response["Contents"] if "checkpoint-" in obj["Key"]],
        key=lambda k: k.split("checkpoint-")[-1],
    )

    if not checkpoints:
        return None

    latest = checkpoints[-1].split("/")[1]  # e.g., checkpoint-500
    local_ckpt_dir = os.path.join(local_output, latest)
    os.makedirs(local_ckpt_dir, exist_ok=True)

    print(f"Downloading latest S3 checkpoint: {latest}")
    for obj in [o for o in response["Contents"] if latest in o["Key"]]:
        local_path = os.path.join(local_output, os.path.relpath(obj["Key"], prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, obj["Key"], local_path)
    return local_ckpt_dir

def get_latest_local_checkpoint(output_dir):
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=os.path.getmtime
    )
    return str(checkpoints[-1]) if checkpoints else None

resume_checkpoint = get_latest_local_checkpoint(OUTPUT_DIR) or download_latest_s3_checkpoint(S3_BUCKET)
if resume_checkpoint:
    print(f"Resuming from local checkpoint: {resume_checkpoint}")
else:
    print("No local checkpoint found. Starting fresh training.")

# ----------------------
# INITIALIZE TRAINER
# ----------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[S3CheckpointCallback(S3_BUCKET, s3_prefix="qlora_checkpoints")]
)

# ----------------------
# START TRAINING
# ----------------------

print("Starting QLoRA training...")

try:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
except torch.cuda.OutOfMemoryError:
    print("⚠️ GPU OOM detected. Reducing batch size and retrying...")
    training_args.per_device_train_batch_size = 2
    trainer.args = training_args
    trainer.train(resume_from_checkpoint=resume_checkpoint)

print("Training complete!")

# ----------------------
# SAVE FINAL MODEL AND TOKENIZER
# ----------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

peft_output_dir = os.path.join(OUTPUT_DIR, "qlora_adapter_only")
os.makedirs(peft_output_dir, exist_ok=True)

print(f"Saving LoRA adapter to {peft_output_dir} ...")
model.save_pretrained(peft_output_dir)  # Saves just the LoRA weights + config

# ----------------------
# UPLOAD FINAL MODEL TO S3
# ----------------------

def upload_final_model(local_dir, bucket_name, s3_prefix="qlora_model_outputs"):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = f"{s3_prefix}/{rel_path}"
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} → s3://{bucket_name}/{s3_path}")

upload_final_model(OUTPUT_DIR, S3_BUCKET)
print("Final model upload complete!")
