from typing import List

from modules.dataset.docx.semantic_chunking import convert_docs_to_chunks, create_dataset_from_chunks
from modules.llm.tokenization_utils import build_tokenize_function

#####################################
# DocX Dataset Builder
#####################################

def create_tokenized_dataset_from_documents(
    base_tokenizer,
    input_file_patterns: List[str],
    max_sequence_length: int = 512
):
    """
    Full pipeline:
    - Reads DOCX
    - Splits into chunks
    - Creates Dataset
    - Tokenizes for the target model
    """
    chunked_data = convert_docs_to_chunks(input_file_patterns, max_sequence_length)
    dataset = create_dataset_from_chunks(chunked_data)
    dataset = dataset.train_test_split(test_size=0.2)

    tokenize_function = build_tokenize_function(
        tokenizer=base_tokenizer,
        alpaca_mode=False,
        text_column="text",
        max_length=max_sequence_length
    )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

#####################################
# Main Example
#####################################

# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-2-7b-hf"

#     # Example: Tokenize DOCX files
#     docx_dataset = create_tokenized_dataset_from_documents(
#         base_llm_model=model_name,
#         input_file_patterns=["data/docs/*.docx"],
#         max_sequence_length=512
#     )
#     print("Docx tokenized dataset:", docx_dataset)