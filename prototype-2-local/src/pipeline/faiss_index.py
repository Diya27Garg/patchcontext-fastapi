import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.json"
FAISS_INDEX_PATH = "data/processed/faiss_index.bin"
EMBEDDINGS_META_PATH = "data/processed/faiss_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality, fully local


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    print(f"Loading local embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    texts = [c["text"][:6000] for c in chunks]

    print(f"Embedding {len(texts)} chunks locally (no API calls)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for cosine similarity via inner product
    )
    embeddings = embeddings.astype("float32")

    print("\nBuilding FAISS index (IndexFlatIP for cosine similarity)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors, dim={dim}.\n")

    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save chunk_id order + embeddings (needed for MMR diversity calculations)
    meta = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "embeddings": embeddings.tolist(),
    }
    with open(EMBEDDINGS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    print(f"Saved metadata to {EMBEDDINGS_META_PATH}")
