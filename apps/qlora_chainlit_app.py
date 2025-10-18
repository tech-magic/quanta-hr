#!/usr/bin/env python3
import os
import json
import chainlit as cl
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from modules.storage.blob_storage_helper import validated_dir_name

CONFIG_PATH = "config/qlora_config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Can't find the qlora config at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

@cl.on_chat_start
async def start():
    await cl.Message(content="🚀 QLoRA Inference App ready! Type your prompt below.").send()

# Load model/tokenizer only once
@cl.cache
def load_model_and_tokenizer():
    config = load_config()
    model_name = config["model"]["name"]
    local_model_dir = os.path.join("model", f"{validated_dir_name(model_name)}_complete_llm")

    if not os.listdir(local_model_dir):
        pass

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
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
