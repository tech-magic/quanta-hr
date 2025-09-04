# pip install peft transformers bitsandbytes accelerate datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict

# --- Alpaca preprocessing ---
def preprocess_alpaca(example):
    prompt = example["instruction"]
    if example.get("input"):
        prompt += f"\nInput: {example['input']}"
    full_text = f"{prompt}\nOutput: {example['output']}"
    return {"text": full_text}

# --- Tokenization and sequence grouping ---
def tokenize_and_group(dataset, tokenizer, max_length=256):
    tokenized_dataset = dataset.map(preprocess_alpaca)

    # Tokenize without padding
    def tokenize_fn(batch):
        tokens = tokenizer(batch["text"], truncation=True, max_length=max_length)
        tokens["labels"] = tokens["input_ids"].copy()  # labels = input_ids
        return tokens

    tokenized_dataset = tokenized_dataset.map(tokenize_fn, batched=True, remove_columns=tokenized_dataset["train"].column_names)

    # Group multiple examples into sequences of max_length
    def group_texts(examples):
        concatenated_input_ids = sum(examples["input_ids"], [])
        concatenated_labels = sum(examples["labels"], [])
        result = {
            "input_ids": [concatenated_input_ids[i:i + max_length] for i in range(0, len(concatenated_input_ids), max_length)],
            "labels": [concatenated_labels[i:i + max_length] for i in range(0, len(concatenated_labels), max_length)],
            "attention_mask": [[1]*len(concatenated_input_ids[i:i + max_length]) for i in range(0, len(concatenated_input_ids), max_length)]
        }
        return result

    tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

# --- CPU-friendly training ---
def train_model_cpu(dataset: DatasetDict):
    model_name = "distilgpt2"  # small CPU-friendly model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # fix padding error

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)  # works on CPU too
    lora_config = LoraConfig(
        r=4,                  # smaller LoRA rank for CPU
        lora_alpha=16,
        target_modules=["c_attn"],  # distilgpt2 uses 'c_attn'
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Tokenize dataset
    tokenized_dataset = tokenize_and_group(dataset, tokenizer, max_length=256)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,    # keep small for CPU
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        output_dir="data/llm_adaptors/distilgpt2-hr-lora",
        save_total_limit=2,
        fp16=False,                       # CPU cannot use FP16
        bf16=False,
        report_to="none",
        group_by_length=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer
    )

    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train()
        eval_metrics = trainer.evaluate()
        print(f"Epoch {epoch+1} validation metrics: {eval_metrics}")

    # Save model + tokenizer
    model.save_pretrained("data/llm_adaptors/distilgpt2-hr-lora")
    tokenizer.save_pretrained("data/llm_adaptors/distilgpt2-hr-lora")
    print("CPU-friendly training completed and model saved!")


