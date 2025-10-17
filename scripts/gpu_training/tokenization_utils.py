import copy

from typing import Dict
from transformers import AutoTokenizer

#####################################
# Tokenization Utility
#####################################

def build_tokenize_function(
    tokenizer,
    include_input: bool = True,
    text_column: str = "text",
    alpaca_mode: bool = False,
    max_length: int = 512,
    padding: str = "max_length"
):
    """
    Returns a tokenize_function appropriate for:
    - Causal LM / Seq2Seq / Encoder models
    - Alpaca format or plain text format
    """
    model_type = tokenizer.__class__.__name__.lower()

    def format_prompt(instruction: str, input_text: str = "", output_text: str = "") -> str:
        instruction = instruction or ""
        input_text = input_text or ""
        output_text = output_text or ""
        if include_input and input_text.strip():
            return f"{instruction.strip()}\n{input_text.strip()}\n\n{output_text.strip()}"
        else:
            return f"{instruction.strip()}\n\n{output_text.strip()}"

    def tokenizer_fn(batch: Dict):
        # Build raw prompts
        if alpaca_mode:
            prompts = [
                format_prompt(
                    batch["instruction"][i] if "instruction" in batch else "",
                    batch["input"][i] if "input" in batch else "",
                    batch["output"][i] if "output" in batch else ""
                )
                for i in range(len(batch["instruction"]))
            ]
        else:
            prompts = batch[text_column]

        # Dispatch by model type
        if any(m in model_type for m in ["llama", "gpt", "mistral", "falcon"]):
            tokens = tokenizer(prompts, truncation=True, padding=padding, max_length=max_length)
            tokens["labels"] = copy.deepcopy(tokens["input_ids"])
            return tokens

        elif any(m in model_type for m in ["t5", "bart", "mbart"]):
            if alpaca_mode:
                inputs = [
                    format_prompt(instr, inp, "")
                    for instr, inp in zip(batch["instruction"], batch["input"])
                ]
                targets = batch["output"]
            else:
                inputs = batch[text_column]
                targets = batch.get("target_text", batch[text_column])

            model_inputs = tokenizer(inputs, truncation=True, padding=padding, max_length=max_length)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, truncation=True, padding=padding, max_length=max_length)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        elif any(m in model_type for m in ["bert", "distilbert", "roberta"]):
            return tokenizer(prompts, truncation=True, padding=padding, max_length=max_length)

        else:
            tokens = tokenizer(prompts, truncation=True, padding=padding, max_length=max_length)
            tokens["labels"] = copy.deepcopy(tokens["input_ids"])
            return tokens

    return tokenizer_fn