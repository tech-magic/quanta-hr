from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import json
import os

def build_rag(base_rag_path, dataset_id, jsonl_file_path, sentence_transformer_name="all-mpnet-base-v2"):

    # Initialize embedding model
    embed = SentenceTransformer(sentence_transformer_name)        # or all-MiniLM-L6-v2 for speed

    # ✅ Use PersistentClient (replaces Client + Settings)
    client = PersistentClient(path=os.path.join(base_rag_path, "chroma_db"))

    # Create or get your collection
    col = client.get_or_create_collection(dataset_id)

    # Read your chunks
    ids, metadatas, docs = [], [], []
    with open(jsonl_file_path) as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                ids.append(f"{r['doc_id']}_{r['section_id']}")
                docs.append(r["text"])
                metadatas.append({
                    "doc_id": r["doc_id"],
                    "heading": r.get("heading", ""),
                    "section_id": r["section_id"]
                })
            except Exception as e:
                print(f"Skipping line {i}: {e}")

    # Encode and add embeddings
    embeddings = embed.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    col.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(f"✅ Indexed {col.count()} HR policy chunks.")

