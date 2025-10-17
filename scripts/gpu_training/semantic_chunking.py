import re
import tiktoken

from typing import List, Dict
from datasets import Dataset

from gpu_training.docx_processing import read_multiple_docx_files

#####################################
# Semantic Chunking
#####################################

def semantic_chunk_text(
    text: str,
    max_tokens: int = 512,
    tokenizer_name: str = "cl100k_base",
) -> List[str]:
    """
    Splits text into semantic chunks that respect a max token length.
    """
    enc = tiktoken.get_encoding(tokenizer_name)
    raw_chunks = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    chunks = []
    current_chunk = ""

    for segment in raw_chunks:
        if not segment.strip():
            continue

        if len(enc.encode(current_chunk + " " + segment)) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = segment
        else:
            current_chunk += " " + segment

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def convert_docs_to_chunks(file_patterns: List[str], max_tokens: int = 512, tokenizer_name: str = "cl100k_base") -> List[Dict]:
    """
    Converts multiple docx files into semantic chunks.
    """
    docs = read_multiple_docx_files(file_patterns)
    chunked_data = []

    for doc in docs:
        chunks = semantic_chunk_text(doc["text"], max_tokens, tokenizer_name)
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })
    return chunked_data

def create_dataset_from_chunks(chunked_data: List[Dict]) -> Dataset:
    """
    Converts chunked data into a HuggingFace Dataset.
    """
    return Dataset.from_list(chunked_data)