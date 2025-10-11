#!/usr/bin/env python3
import os
import re
import json
import argparse
from datetime import datetime, timezone
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def validated_dir_name(text):
    # Replace all non-alphanumeric characters with underscores
    return re.sub(r'[^A-Za-z0-9]', '_', text)

# ---------------------------
# ARGUMENT PARSING
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA CPU Inference Runner")
    parser.add_argument("--config", help="Path to JSON config file (optional). If not provided, reads from stdin.")
    parser.add_argument("--s3-bucket", help="Override S3 bucket name (optional).")
    parser.add_argument("--prompt", help="Override prompt text (optional).")
    return parser.parse_args()

# ---------------------------
# LOAD CONFIG
# ---------------------------
CONFIG_PATH = "data/config/qlora_config.json"
def load_config(args):
    config_path = args.config or CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Can't find the qlora config at: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

# ---------------------------
# S3 DOWNLOAD HELPER
# ---------------------------
def download_s3_dir_if_changed(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            s3_mod_time = obj["LastModified"]
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path):
                local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
                if local_mod_time >= s3_mod_time:
                    continue
            print(f"⬇️ Downloading s3://{bucket_name}/{key} → {local_path}")
            s3.download_file(bucket_name, key, local_path)

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    args = parse_args()
    config = load_config(args)

    # ---------------------------
    # EXTRACT RELEVANT CONFIG
    # ---------------------------
    model_name = config["model"]["name"]

    s3_bucket = args.s3_bucket or os.getenv("QLORA_S3_BUCKET", "default-qlora-s3-bucket")

    remote_model_dir = f"{validated_dir_name(model_name)}_complete_llm"
    local_model_dir = os.path.join("model", remote_model_dir)

    remote_adapter_dir = f"{validated_dir_name(model_name)}_adapter_only"
    local_adapter_dir = os.path.join("model", remote_adapter_dir)

    remote_checkpoints_dir = f"{validated_dir_name(model_name)}_training_checkpoints"
    local_checkpoints_dir = os.path.join("model", remote_checkpoints_dir)

    prompt_text = args.prompt or "Explain in simple terms how QLoRA fine-tuning works."

    print(f"📥 Model name: {model_name}")
    print(f"🪣 S3 Bucket: {s3_bucket}")

    print(f"📂 Remote model dir: {remote_model_dir}")
    print(f"💾 Local model dir: {local_model_dir}")

    print(f"📂 Remote adapter dir: {remote_adapter_dir}")
    print(f"💾 Local adapter dir: {local_adapter_dir}")

    print(f"📂 Remote checkpoints dir: {remote_checkpoints_dir}")
    print(f"💾 Local checkpoints dir: {local_checkpoints_dir}")

    # ---------------------------
    # DOWNLOAD MODEL/ADAPTER/CHECKPOINTS IF NEEDED
    # ---------------------------
    os.makedirs(local_model_dir, exist_ok=True)
    if not os.listdir(local_model_dir):
        print("🚀 Downloading model from S3 ...")
        download_s3_dir_if_changed(s3_bucket, remote_model_dir, local_model_dir)
    else:
        print(f"✅ Model already exists locally in {local_model_dir}")

    os.makedirs(local_adapter_dir, exist_ok=True)
    if not os.listdir(local_adapter_dir):
        print("🚀 Downloading Model adapter from S3 ...")
        download_s3_dir_if_changed(s3_bucket, remote_adapter_dir, local_adapter_dir)
    else:
        print(f"✅ Model adapter already exists locally in {local_adapter_dir}")

    os.makedirs(local_checkpoints_dir, exist_ok=True)
    if not os.listdir(local_checkpoints_dir):
        print("🚀 Downloading Model checkpoints from S3 ...")
        download_s3_dir_if_changed(s3_bucket, remote_checkpoints_dir, local_checkpoints_dir)
    else:
        print(f"✅ Model checkpoints already exists locally in {local_checkpoints_dir}")

    # ---------------------------
    # LOAD MODEL & TOKENIZER
    # ---------------------------
    device = torch.device("cpu")
    print(f"🧠 Loading tokenizer from {local_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🧠 Loading model from {local_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # ---------------------------
    # GENERATE OUTPUT
    # ---------------------------
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n=== Prompt ===\n" + prompt_text)
    print("\n=== Model Output ===\n" + text)

if __name__ == "__main__":
    main()
