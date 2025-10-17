import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

from modules.storage.blob_storage_helper import upload_final_results

# ------------------------------------------------------------------------------
# LOAD BASE MODEL (WITH 4-BIT Quantization) + TOKENIZER USED BY THE BASE MODEL 
# ------------------------------------------------------------------------------

def load_bnb_config(config):
    bnb_cfg = config["model"]["quantization"]
    torch_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"]
    ), torch_dtype

def load_lora_config(config):
    lora_cfg = config["model"]["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )

def load_quantized_base_llm(
        base_llm_model,
        local_quantization_path,
        blob_storage_identifier,
        config
    ):

    q_config, torch_dtype = load_bnb_config(config)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_llm_model,
        quantization_config=q_config,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    base_tokenizer = AutoTokenizer.from_pretrained(base_llm_model)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # Saving Base Model + Tokenizer
    base_model.save_pretrained(local_quantization_path)
    base_tokenizer.save_pretrained(local_quantization_path)

    # Upload quantized Base LLM to S3 Bucket
    # TODO: check if there is already existing quantized Base Model in the S3 Bucket
    upload_final_results(local_quantization_path, blob_storage_identifier, local_quantization_path)

    return base_model, base_tokenizer

def load_lora_model_from_base_model(base_model, config):
    lora_model = get_peft_model(base_model, load_lora_config(config))
    lora_model.config.use_cache = False

    return lora_model