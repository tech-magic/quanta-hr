#!/usr/bin/env python3

# ----------------------
# IMPORT LIBRARIES
# ----------------------

import sys
import os
import json

import torch
from transformers import (
    Trainer,
    TrainingArguments
)

from modules.llm.qlora_utils import load_quantized_base_llm, load_lora_model_from_base_model
from modules.storage.blob_storage_helper import (
    validated_dir_name,
    LLMTrainingCheckpointCallback, 
    upload_final_results, 
    get_latest_local_checkpoint, 
    download_latest_s3_checkpoint
)
from modules.dataset.dataset_utils import build_tokenized_dataset

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

CONFIG_PATH = "config/qlora_config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Can't find the qlora config at: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_NAME = config["model"]["name"]
BASE_LLM_MODEL = config["model"]["base_llm_model"]

BASE_LLM_DIR = os.path.join(f"{validated_dir_name(BASE_LLM_MODEL)}_base_llm")
OUTPUT_DIR = os.path.join(f"{validated_dir_name(MODEL_NAME)}_complete_llm")
TRAINING_DIR = os.path.join(f"{validated_dir_name(MODEL_NAME)}_adapter_with_training")
ADAPTER_DIR = os.path.join(f"{validated_dir_name(MODEL_NAME)}_adapter_only")
CHECKPOINT_DIR = os.path.join(f"{validated_dir_name(MODEL_NAME)}_training_checkpoints")

print(f"✅ Started training for LLM model: {MODEL_NAME}")
print(f"✅ Using base LLM model: {BASE_LLM_MODEL}")

os.makedirs(BASE_LLM_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"✅ QLoRA {BASE_LLM_MODEL} (Base LLM) - Output directory: {BASE_LLM_DIR}")
print(f"✅ QLoRA {MODEL_NAME} Custom LLM (merged with BaseLLM) - Output directory: {OUTPUT_DIR}")
print(f"✅ QLoRA {MODEL_NAME} QLoRA Adapter with Training MetaData - Output directory: {TRAINING_DIR}")
print(f"✅ QLoRA {MODEL_NAME} QLoRA Adapter Only - Output directory: {ADAPTER_DIR}")
print(f"✅ QLoRA {MODEL_NAME} QLoRA local Training Checkpoints - Output directory: {CHECKPOINT_DIR}")
print(f"✅ Using S3 bucket: {S3_BUCKET}")

# ----------------------
# Customized LLM loading from BASE_LLM_MODEL for QLoRA (Quantization + LoRA)
# ----------------------
base_model, base_tokenizer = load_quantized_base_llm(BASE_LLM_MODEL, BASE_LLM_DIR, S3_BUCKET, config)
qlora_model = load_lora_model_from_base_model(base_model, config)

# ----------------------
# LOAD + TOKENIZE + MERGE TRAINING DATASETS
# ----------------------
tokenized_dataset = build_tokenized_dataset(config["datasets"], base_tokenizer, config["model"]["max_sequence_length"])

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
    resume_checkpoint = get_latest_local_checkpoint(CHECKPOINT_DIR) or \
        download_latest_s3_checkpoint(S3_BUCKET, CHECKPOINT_DIR, CHECKPOINT_DIR)

if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
else:
    print("No checkpoint found. Starting fresh training.")

# ----------------------
# TRAINER
# ----------------------
trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=base_tokenizer,
    callbacks=[LLMTrainingCheckpointCallback(S3_BUCKET, CHECKPOINT_DIR, CHECKPOINT_DIR)]
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

# Saving only the adapter
qlora_model.save_pretrained(ADAPTER_DIR)
base_tokenizer.save_pretrained(ADAPTER_DIR)

# Saving adapter with training metadata
trainer.save_model(TRAINING_DIR)
base_tokenizer.save_pretrained(TRAINING_DIR)

# Save complete customized QLoRA LLM
merged_model = qlora_model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
base_tokenizer.save_pretrained(OUTPUT_DIR)

# ----------------------
# UPLOAD FINAL RESULTS TO S3
# ----------------------
upload_final_results(ADAPTER_DIR, S3_BUCKET, ADAPTER_DIR)
upload_final_results(TRAINING_DIR, S3_BUCKET, TRAINING_DIR)
upload_final_results(OUTPUT_DIR, S3_BUCKET, OUTPUT_DIR)

print(f"✅ Successfully uploaded Trained LLM model: {MODEL_NAME}")
