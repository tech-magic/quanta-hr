import os
import glob

from typing import List, Dict
from docx import Document

#####################################
# DOCX Processing
#####################################

def read_docx_file(file_path: str) -> str:
    """
    Reads a .docx file and returns the joined text content.
    """
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)

def read_multiple_docx_files(file_patterns: List[str]) -> List[Dict]:
    """
    Reads multiple docx files based on glob patterns.
    Returns a list of dict with filename and text.
    """
    all_texts = []
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            if file_path.endswith(".docx"):
                filename = os.path.basename(file_path)
                text = read_docx_file(file_path)
                all_texts.append({"filename": filename, "text": text})
    
    print(f"Data for DocX dataset -> {all_texts}")

    return all_texts