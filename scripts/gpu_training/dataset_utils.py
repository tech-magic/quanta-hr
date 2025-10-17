from datasets import concatenate_datasets, DatasetDict

from gpu_training.alpaca_to_dataset import create_tokenized_alpaca_dataset
from gpu_training.docx_to_dataset import create_tokenized_dataset_from_documents

def build_tokenized_dataset(
    datasets_config,
    base_tokenizer,
    max_sequence_length: int = 512
):
    """
    Dynamically builds a merged tokenized DatasetDict from multiple dataset configs.
    
    Args:
        datasets_config (list): List of dicts with keys:
            - type: 'text', 'alpaca', etc.
            - input_files: list of file patterns
        model_name (str): Base LLM model for tokenization
        max_sequence_length (int): Max sequence length for tokenization
        
    Returns:
        DatasetDict: {"train": merged_train_dataset, "validation": merged_val_dataset}
    """

    print(f"Tokenizing datasets -> {datasets_config}")

    def load_dataset_by_type(dataset_type, input_files):
        if dataset_type == "docx":
            print(f"Input file patterns for DocX -> {input_files}")
            return create_tokenized_dataset_from_documents(
                base_tokenizer=base_tokenizer,
                input_file_patterns=input_files,
                max_sequence_length=max_sequence_length
            )
        elif dataset_type == "alpaca":
            print(f"Input file patterns for Alpaca -> {input_files}")
            return create_tokenized_alpaca_dataset(
                base_tokenizer=base_tokenizer,
                input_file_patterns=input_files,
                max_sequence_length=max_sequence_length
            )
        else:
            raise ValueError(f"❌ Unsupported dataset type: {dataset_type}")

    train_splits = []
    val_splits = []

    for dataset_cfg in datasets_config:
        ds_type = dataset_cfg["type"]
        input_files = dataset_cfg["input_files"]

        tokenized_ds = load_dataset_by_type(ds_type, input_files)

        if "train" in tokenized_ds:
            train_splits.append(tokenized_ds["train"])
        else:
            raise ValueError(f"Dataset {ds_type} missing 'train' split")

        if "test" in tokenized_ds:
            val_splits.append(tokenized_ds["test"])
        else:
            raise ValueError(f"Dataset {ds_type} missing validation or test split")

    merged_train_dataset = concatenate_datasets(train_splits)
    merged_val_dataset = concatenate_datasets(val_splits)

    return DatasetDict({
        "train": merged_train_dataset,
        "test": merged_val_dataset
    })
