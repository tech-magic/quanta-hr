#!/usr/bin/env python3

# python apps/docx_preprocess.py data/ABC_Consulting_HR_Policies.docx
from docx import Document
from pathlib import Path
import re
import json

def extract_sections(docx_path, config):
    doc = Document(docx_path)

    patterns = [re.compile(p, re.I) for p in config.get("heading_patterns", [])]
    detect_all_caps = config.get("detect_all_caps", True)

    sections = []
    cur_heading = "Document"
    cur_text = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        # naive heading detection: all-caps or bold-like markers / patterns you can tune
        # configurable heading detection
        is_heading = any(current_pattern.match(txt) for current_pattern in patterns)
        if detect_all_caps and txt.isupper():
            is_heading = True

        if is_heading:
            if cur_text:
                sections.append({"heading": cur_heading, "text": "\n".join(cur_text)})
                cur_text = []
            cur_heading = txt
        else:
            cur_text.append(txt)
    if cur_text:
        sections.append({"heading": cur_heading, "text": "\n".join(cur_text)})
    return sections

def chunk_text(text, max_tokens=300):
    # approximate by words
    words = text.split()
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def preprocess(docx_path, config, out_jsonl="section_chunks.jsonl", max_tokens=300):
    sections = extract_sections(docx_path, config)
    out = []
    for i, s in enumerate(sections):
        chunks = chunk_text(s["text"], max_tokens=max_tokens)
        for j, c in enumerate(chunks):
            out.append({
                "doc_id": Path(docx_path).stem,
                "section_id": f"{i+1}.{j+1}",
                "heading": s["heading"],
                "text": c
            })
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(out)} chunks to {out_jsonl}")
    return out_jsonl
