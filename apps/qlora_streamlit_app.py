#!/usr/bin/env python3
import os
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

from modules.storage.blob_storage_helper import validated_dir_name

CONFIG_PATH = "/workspace/config/qlora_config.json"

# ---------------------------
# Load configuration
# ---------------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        st.error(f"❌ Can't find the qlora config at: {CONFIG_PATH}")
        st.stop()
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# ---------------------------
# Load model and tokenizer (cached)
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    config = load_config()
    model_name = config["model"]["name"]
    local_model_dir = os.path.join(
        "/workspace/model", f"{validated_dir_name(model_name)}_complete_llm"
    )

    # ---------------------------
    # LOAD MODEL & TOKENIZER
    # ---------------------------
    device = torch.device("cpu")
    st.write(f"🧠 Loading tokenizer from {local_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    st.write(f"🧠 Loading model from {local_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
    ).to(device)
    model.eval()

    return tokenizer, model, device

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="QLoRA Inference App", page_icon="🚀")
st.title("🚀 QLoRA Inference App")

# Load model only once
with st.spinner("🔥 Loading model and tokenizer..."):
    tokenizer, model, device = load_model_and_tokenizer()
st.success("✅ Model and tokenizer loaded!")

# Text input area
prompt_text = st.text_area("✍️ Enter your prompt:", height=150)

# Generate button
if st.button("Generate Response"):
    if not prompt_text.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("🤖 Generating response..."):
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            print(f"Received Query -> {inputs}")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            response_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated Response -> {response_text}")

        # Display response
        st.markdown("### 📝 Response")
        st.write(response_text)

# Optional footer
st.markdown("---")
st.caption("Powered by QLoRA + Streamlit")
