#!/usr/bin/env python3

# ----------------------
# IMPORT LIBRARIES
# ----------------------

import sys
import os
import json
from pathlib import Path
import boto3
import re

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

# -----------
# UTILITIES
# -----------
def validated_dir_name(text):
    # Replace all non-alphanumeric characters with underscores
    return re.sub(r'[^A-Za-z0-9]', '_', text)

def upload_final_results(local_dir, bucket_name, s3_prefix):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = f"{s3_prefix}/{rel_path}"
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} → s3://{bucket_name}/{s3_path}")

def download_latest_s3_checkpoint(bucket, checkpoint_uploads_dir, local_output_dir):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=checkpoint_uploads_dir)
    if "Contents" not in response:
        return None

    checkpoints = sorted(
        [obj["Key"] for obj in response["Contents"] if "checkpoint-" in obj["Key"]],
        key=lambda k: int(k.split("checkpoint-")[-1].split("/")[0])
    )
    if not checkpoints:
        return None

    latest = checkpoints[-1].split("/")[1]
    local_ckpt_dir = os.path.join(local_output_dir, latest)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    print(f"Downloading latest S3 checkpoint: {latest}")
    for obj in [o for o in response["Contents"] if latest in o["Key"]]:
        local_path = os.path.join(local_output_dir, os.path.relpath(obj["Key"], checkpoint_uploads_dir))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, obj["Key"], local_path)
    return local_ckpt_dir

def get_latest_local_checkpoint(local_output_dir):
    checkpoints = sorted(Path(local_output_dir).glob("checkpoint-*"), key=os.path.getmtime)
    return str(checkpoints[-1]) if checkpoints else None

class S3CheckpointCallback(TrainerCallback):
    def __init__(self, bucket_name, s3_prefix):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3")

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
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
# PRINT WORKING DIRECTORY
# ----------------------

# Full path to the current python file
full_python_path = os.path.abspath(__file__)

print(f"Executing Python Script [ {full_python_path} ] with Working Directory [ {os.getcwd()} ]")

# ----------------------
# LOAD CONFIGURATION
# ----------------------
if len(sys.argv) > 1:
    S3_BUCKET = sys.argv[1]
else:
    raise ValueError("Can't find a value for the S3 Bucket!")

CONFIG_PATH = "data/config/qlora_config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Can't find the qlora config at: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_NAME = config["model"]["name"]
BASE_LLM_MODEL = config["model"]["base_llm_model"]
OUTPUT_DIR = f"{validated_dir_name(MODEL_NAME)}_complete_llm"
ADAPTER_DIR = f"{validated_dir_name(MODEL_NAME)}_adapter_only"
CHECKPOINT_DIR = f"{validated_dir_name(MODEL_NAME)}_training_checkpoints"

print(f"✅ Starting Training LLM model: {MODEL_NAME}")

print(f"✅ Using base LLM model: {BASE_LLM_MODEL}")
print(f"✅ QLoRA LLM output directory: {OUTPUT_DIR}")
print(f"✅ QLoRA Adapter output directory: {ADAPTER_DIR}")
print(f"✅ Using S3 bucket: {S3_BUCKET}")

# ----------------------
# LOAD DATASET
# ----------------------

dataset = load_dataset(
    config["dataset"]["type"],
    data_files={
        "train": config["dataset"]["train_files"],
        "validation": config["dataset"]["validation_files"]
    }
)

# ----------------------
# PROMPT FORMATTING
# ----------------------

def format_prompt(instruction, input_text, output_text):
    """
    Flatten instruction/input/output into a single string per example.
    """
    if config["prompt_format"]["include_input"] and input_text:
        return f"{instruction}\n{input_text}\n\n{output_text}"
    else:
        return f"{instruction}\n\n{output_text}"

# ----------------------
# TOKENIZER
# ----------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# TOKENIZER FUNCTION
# ----------------------

def tokenize_function(batch):
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
        max_length=config["model"]["max_sequence_length"]
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

bnb_cfg = config["model"]["quantization"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=bnb_cfg["load_in_4bit"],
    bnb_4bit_compute_dtype=getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"]
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_LLM_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_cfg = config["model"]["lora"]
lora_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    target_modules=lora_cfg["target_modules"],
    lora_dropout=lora_cfg["lora_dropout"],
    bias=lora_cfg["bias"],
    task_type=lora_cfg["task_type"]
)

model = get_peft_model(base_model, lora_config)
model.config.use_cache = False

# ----------------------
# TRAINING ARGUMENTS
# ----------------------

training_args = TrainingArguments(
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    warmup_steps=config["training"]["warmup_steps"],
    max_steps=config["training"]["max_steps"],
    learning_rate=config["training"]["learning_rate"],
    fp16=config["training"]["fp16"] and torch.cuda.is_available(),
    logging_steps=config["training"]["logging_steps"],
    output_dir=OUTPUT_DIR,
    save_strategy=config["training"]["save_strategy"],
    save_steps=config["training"]["save_steps"],
    save_total_limit=config["training"]["save_total_limit"],
    report_to=config["training"]["report_to"]
)

# ----------------------
# CHECKPOINT RESUME LOGIC
# ----------------------
resume_checkpoint = None
if config["training"]["resume_from_uploads"]:
    resume_checkpoint = get_latest_local_checkpoint(OUTPUT_DIR) or \
        download_latest_s3_checkpoint(S3_BUCKET, CHECKPOINT_DIR, OUTPUT_DIR)

if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
else:
    print("No checkpoint found. Starting fresh training.")

# ----------------------
# TRAINER
# ----------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[S3CheckpointCallback(S3_BUCKET, CHECKPOINT_DIR)]
)

# ----------------------
# TRAINING
# ----------------------

print("🚀 Starting QLoRA training...")

try:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
except torch.cuda.OutOfMemoryError:
    print("⚠️ GPU OOM detected. Reducing batch size and retrying...")
    training_args.per_device_train_batch_size = 2
    trainer.args = training_args
    trainer.train(resume_from_checkpoint=resume_checkpoint)

print("✅ Training complete!")

# ----------------------
# SAVE MODEL + TOKENIZER
# ----------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

peft_output_dir = os.path.join(ADAPTER_DIR)
os.makedirs(peft_output_dir, exist_ok=True)
model.save_pretrained(peft_output_dir)

# ----------------------
# UPLOAD FINAL MODEL TO S3
# ----------------------

upload_final_results(OUTPUT_DIR, S3_BUCKET, OUTPUT_DIR)
upload_final_results(ADAPTER_DIR, S3_BUCKET, ADAPTER_DIR)

print(f"✅ Successfully uploaded Trained LLM model: {MODEL_NAME}")
