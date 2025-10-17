#!/usr/bin/env python3
import os
import re
import json
import chainlit as cl
from datetime import datetime, timezone
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG_PATH = "data/config/qlora_config.json"

def validated_dir_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '_', text)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Can't find the qlora config at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

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

@cl.on_chat_start
async def start():
    await cl.Message(content="🚀 QLoRA Inference App ready! Type your prompt below.").send()

# Load model/tokenizer only once
@cl.cache
def load_model_and_tokenizer():
    config = load_config()
    model_name = config["model"]["name"]
    bucket_name = os.getenv("QLORA_S3_BUCKET", "default-qlora-s3-bucket")

    # Construct paths
    remote_model_dir = f"{validated_dir_name(model_name)}_complete_llm"
    local_model_dir = os.path.join("model", remote_model_dir)

    remote_adapter_dir = f"{validated_dir_name(model_name)}_adapter_only"
    local_adapter_dir = os.path.join("model", remote_adapter_dir)

    remote_checkpoints_dir = f"{validated_dir_name(model_name)}_training_checkpoints"
    local_checkpoints_dir = os.path.join("model", remote_checkpoints_dir)

    os.makedirs(local_model_dir, exist_ok=True)
    if not os.listdir(local_model_dir):
        print("🚀 Downloading model from S3 ...")
        download_s3_dir_if_changed(bucket_name, remote_model_dir, local_model_dir)

    os.makedirs(local_adapter_dir, exist_ok=True)
    if not os.listdir(local_adapter_dir):
        print("🚀 Downloading adapter from S3 ...")
        download_s3_dir_if_changed(bucket_name, remote_adapter_dir, local_adapter_dir)

    os.makedirs(local_checkpoints_dir, exist_ok=True)
    if not os.listdir(local_checkpoints_dir):
        print("🚀 Downloading checkpoints from S3 ...")
        download_s3_dir_if_changed(bucket_name, remote_checkpoints_dir, local_checkpoints_dir)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    return tokenizer, model, device

@cl.on_message
async def generate_response(message: cl.Message):
    tokenizer, model, device = load_model_and_tokenizer()
    prompt_text = message.content

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
    await cl.Message(content=text).send()
