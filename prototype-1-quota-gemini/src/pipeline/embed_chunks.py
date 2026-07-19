import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")
client = genai.Client(api_key=GEMINI_API_KEY)

CHUNKS_PATH = "data/processed/chunks.json"
EMBEDDINGS_PATH = "data/processed/embeddings.json"

MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768

SLEEP_BETWEEN_CALLS = 3.0
SAVE_EVERY = 20
MAX_RETRIES = 3


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["chunk_id"]: item for item in data}
    return {}


def save_embeddings(embeddings_by_id):
    os.makedirs("data/processed", exist_ok=True)
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(embeddings_by_id.values()), f)


def embed_text(text: str):
    text = text[:6000]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=OUTPUT_DIM,
                ),
            )
            return result.embeddings[0].values
        except Exception as e:
            wait = 10 * attempt
            print(f"  Error on attempt {attempt}/{MAX_RETRIES}: {str(e)[:150]}")
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)
    return None


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    existing = load_existing_embeddings()
    print(f"Found {len(existing)} already-embedded chunks (resuming from checkpoint).\n")

    remaining = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"{len(remaining)} chunks left to embed. Single-threaded, {SLEEP_BETWEEN_CALLS}s pacing.\n")

    processed_since_save = 0
    for i, chunk in enumerate(remaining, 1):
        vector = embed_text(chunk["text"])
        if vector is not None:
            existing[chunk["chunk_id"]] = {"chunk_id": chunk["chunk_id"], "embedding": vector}
            processed_since_save += 1

        if i % 10 == 0 or i == len(remaining):
            print(f"Progress: {i}/{len(remaining)} embedded this run ({len(existing)}/{len(chunks)} total)")

        if processed_since_save >= SAVE_EVERY:
            save_embeddings(existing)
            processed_since_save = 0
            print(f"  Checkpoint saved ({len(existing)} total embeddings).")

        time.sleep(SLEEP_BETWEEN_CALLS)

    save_embeddings(existing)
    print(f"\nDone. {len(existing)}/{len(chunks)} chunks embedded.")