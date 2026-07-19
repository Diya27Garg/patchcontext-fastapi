import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/tfidf_index.pkl"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    texts = [c["text"] for c in chunks]

    print("Building TF-IDF index (this runs locally, in seconds, no API calls)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams, better matches on phrases
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"Index built. Matrix shape: {matrix.shape}\n")

    os.makedirs("data/processed", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunk_ids": [c["chunk_id"] for c in chunks],
        }, f)

    print(f"Saved index to {INDEX_PATH}")