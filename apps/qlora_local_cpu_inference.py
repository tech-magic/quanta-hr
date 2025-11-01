#!/usr/bin/env python3
import os
import json
import argparse
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from modules.storage.blob_storage_helper import download_s3_dir_if_changed, validated_dir_name

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
CONFIG_PATH = "/workspace/config/qlora_config.json"
def load_config(args):
    config_path = args.config or CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Can't find the qlora config at: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

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
    local_model_dir = os.path.join("/workspace/model", remote_model_dir)

    remote_adapter_dir = f"{validated_dir_name(model_name)}_adapter_only"
    local_adapter_dir = os.path.join("/workspace/model", remote_adapter_dir)

    remote_checkpoints_dir = f"{validated_dir_name(model_name)}_training_checkpoints"
    local_checkpoints_dir = os.path.join("/workspace/model", remote_checkpoints_dir)

    prompt_text = args.prompt or "How many days of annual leave can I have per year?"

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
        torch_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
    ).to(device)
    model.eval()

    # ---------------------------
    # GENERATE OUTPUT
    # ---------------------------
    # Start the timer
    start_time = time.time()
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

    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Convert seconds to hh:mm:ss
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    print(f"Time taken to answer the LLM Query: {formatted_time}")



if __name__ == "__main__":
    main()
