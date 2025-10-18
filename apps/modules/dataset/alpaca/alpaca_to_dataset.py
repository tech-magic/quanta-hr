import glob

from typing import List
from datasets import load_dataset

from modules.llm.tokenization_utils import build_tokenize_function

#####################################
# Alpaca Dataset Builder
#####################################

def create_tokenized_alpaca_dataset(
    base_tokenizer,
    input_file_patterns: List[str],
    max_sequence_length: int = 512,
    include_prompt_inputs=True
):
    """
    Alpaca-style JSON dataset loader + tokenizer.
    """
    expanded_files = []
    for pattern in input_file_patterns:
        for file_path in glob.glob(pattern):
            expanded_files.append(file_path)

    print("Input files for the Alpaca Dataset:", expanded_files)

    dataset = load_dataset("json", data_files=expanded_files)
    dataset = dataset["train"].train_test_split(test_size=0.2)

    tokenize_function = build_tokenize_function(
        tokenizer=base_tokenizer,
        include_input=include_prompt_inputs,
        alpaca_mode=True,
        max_length=max_sequence_length
    )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_dataset

#####################################
# Main Example
#####################################

# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-2-7b-hf"

#     # Example: Tokenize Alpaca JSON files
#     alpaca_dataset = create_tokenized_alpaca_dataset(
#         base_llm_model=model_name,
#         input_file_patterns=["data/hr_policies/*.json"],
#         max_sequence_length=512
#     )
#     print("Alpaca tokenized dataset:", alpaca_dataset)