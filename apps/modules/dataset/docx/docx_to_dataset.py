import glob
import os

from modules.dataset.docx.docx_preprocess import preprocess
from modules.storage.blob_storage_helper import validated_dir_name

from modules.dataset.docx.docx_to_rag import build_rag
from modules.dataset.docx.docx_to_alpaca import distill

from modules.dataset.alpaca.alpaca_to_dataset import create_tokenized_alpaca_dataset

#####################################
# DocX Dataset Builder
#####################################


def create_tokenized_dataset_from_documents(
    dataset_config,
    base_dataset_output_dir,
    llm_tokenizer,
    llm_tokenizer_max_length: int = 1024
):

    input_file_patterns = dataset_config.get("input_files", [])
    max_sequence_length = dataset_config.get("max_sequence_length", 300)
    dataset_id = dataset_config.get("id", "default")
    section_detection_config = dataset_config.get("section_detection", {})

    base_dataset_rag_dir = os.path.join(base_dataset_output_dir, "rag")
    os.makedirs(base_dataset_rag_dir, exist_ok=True)

    curr_dataset_output_dir = os.path.join(base_dataset_output_dir, dataset_id)
    os.makedirs(curr_dataset_output_dir, exist_ok=True)

    curr_dataset_datafile_dir = os.path.join(curr_dataset_output_dir, "data_files")
    os.makedirs(curr_dataset_datafile_dir, exist_ok=True)

    q_lora_config = dataset_config.get("use_cases", {}).get("qlora", None)
    rag_config = dataset_config.get("use_cases", {}).get("rag", None)

    alpaca_generator_config = q_lora_config.get("alpaca_generator", {})
    alpaca_generation_provider = alpaca_generator_config.get("provider", "aws")
    alpaca_generation_profile = alpaca_generator_config.get("config", {}).get("profile", "default")
    alpaca_generation_llm = alpaca_generator_config.get("config", {}).get("llm_config", {}).get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")

    alpaca_files = []

    for pattern in input_file_patterns:
        for file_path in glob.glob(pattern):
            if file_path.endswith(".docx"):
                filename = os.path.basename(file_path)

                output_jsonl = os.path.join(curr_dataset_datafile_dir, f"{validated_dir_name(filename)}_preprocessed.jsonl")
                preprocess(file_path, section_detection_config, output_jsonl, max_sequence_length)
                                
                if rag_config is not None:
                    sentence_transformer_name = rag_config.get("sentence_transformer", "all-mpnet-base-v2")
                    build_rag(base_dataset_rag_dir, dataset_id, output_jsonl, sentence_transformer_name)

                output_summary_alpaca_file = os.path.join(curr_dataset_datafile_dir, f"{validated_dir_name(filename)}_summary_alpaca.json")
                distill(output_jsonl, output_summary_alpaca_file, alpaca_generation_provider, alpaca_generation_profile, alpaca_generation_llm, "summary")
                alpaca_files.append(output_summary_alpaca_file)

                output_qa_alpaca_file = os.path.join(curr_dataset_datafile_dir, f"{validated_dir_name(filename)}_qa_alpaca.json")
                distill(output_jsonl, output_qa_alpaca_file, alpaca_generation_provider, alpaca_generation_profile, alpaca_generation_llm, "qa")
                alpaca_files.append(output_qa_alpaca_file)

    return create_tokenized_alpaca_dataset(
                base_tokenizer=llm_tokenizer,
                include_prompt_inputs=True,
                input_file_patterns=alpaca_files,
                max_sequence_length=llm_tokenizer_max_length
            )
