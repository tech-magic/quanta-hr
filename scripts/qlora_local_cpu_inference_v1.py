#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# CONFIGURATION
# ----------------------

S3_BUCKET = sys.argv[1] if len(sys.argv) > 1 else "default-qlora-s3-bucket"

S3_PREFIX = "qlora_model_outputs"   # S3 folder with final model
LOCAL_MODEL_DIR = "data/qlora_model"

# ----------------------
# DEVICE SETUP
# ----------------------

# Force CPU to avoid MPS issues
device = torch.device("cpu")
print("Using CPU for inference.")

# ----------------------
# S3 DOWNLOAD (INCREMENTAL)
# ----------------------

def download_s3_dir_if_changed(bucket_name, s3_prefix, local_dir):
    """
    Download files from S3 only if missing locally or if remote is newer.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            s3_mod_time = obj["LastModified"]  # UTC
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Skip if local file exists and is up-to-date
            if os.path.exists(local_path):
                local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
                if local_mod_time >= s3_mod_time:
                    continue

            print(f"Downloading s3://{bucket_name}/{key} → {local_path}")
            s3.download_file(bucket_name, key, local_path)

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
download_s3_dir_if_changed(S3_BUCKET, S3_PREFIX, LOCAL_MODEL_DIR)

# ----------------------
# LOAD TOKENIZER AND MODEL (FULL MODEL)
# ----------------------

# Assuming the full model (LoRA merged) was saved with `trainer.save_model()`
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=torch.float32
).to(device)
model.eval()

# ----------------------
# PROMPT / INFERENCE FUNCTION
# ----------------------

def generate(prompt, max_new_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ----------------------
# EXAMPLE USAGE
# ----------------------

prompt_text = "Explain in simple terms how QLoRA fine-tuning works."
generated_text = generate(prompt_text)

print("\n=== Prompt ===")
print(prompt_text)
print("\n=== Model Output ===")
print(generated_text)
