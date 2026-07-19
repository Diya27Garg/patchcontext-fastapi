import os
import json
import pickle
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

from faithfulness_guard import verify_answer, annotate_answer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_2")
client = genai.Client(api_key=GEMINI_API_KEY)

CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/tfidf_index.pkl"

GEN_MODEL_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]
TOP_K = 6


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(query: str, index, chunks_by_id, top_k=TOP_K):
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunk_ids = index["chunk_ids"]

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    top = [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    results = []
    for cid, score in top:
        chunk = chunks_by_id[cid]
        results.append({**chunk, "score": float(score)})
    return results


def build_prompt(query: str, retrieved_chunks):
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        source_label = f"[{i}] {c['source_type'].upper()} #{c.get('source_number')}: {c.get('title', '')}"
        context_blocks.append(f"{source_label}\nURL: {c['url']}\n{c['text'][:1500]}")

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are PatchContext, an assistant that explains design decisions in the FastAPI codebase using real GitHub discussion history.

Answer the user's question using ONLY the context below. For every claim, cite the source using its bracket number, e.g. [1], [2]. If the context doesn't fully answer the question, say so honestly rather than guessing.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (with inline [N] citations):"""
    return prompt


_working_model = None


def generate_answer(query: str, retrieved_chunks):
    global _working_model
    prompt = build_prompt(query, retrieved_chunks)

    models_to_try = [_working_model] if _working_model else GEN_MODEL_CANDIDATES

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if _working_model != model_name:
                print(f"  (using model: {model_name})")
                _working_model = model_name
            return response.text
        except Exception as e:
            last_error = e
            print(f"  Model '{model_name}' failed: {str(e)[:150]}")
            continue

    raise RuntimeError(f"All candidate models failed. Last error: {last_error}")


def ask(query: str):
    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    index = load_index()

    retrieved = retrieve(query, index, chunks_by_id)

    if not retrieved:
        return "No relevant information found in the indexed data.", [], []

    answer = generate_answer(query, retrieved)

    # --- hallucination guard: verify each cited claim against its source ---
    try:
        verdicts = verify_answer(answer, retrieved, client, _working_model)
        annotated = annotate_answer(verdicts)
    except Exception as e:
        print(f"  (guard skipped: {str(e)[:150]})")
        verdicts = []
        annotated = answer
    # -------------------------------------------------------------------

    return annotated, retrieved, verdicts


if __name__ == "__main__":
    print("PatchContext - ask a question about FastAPI's design history.")
    print("(Type 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        print("\nSearching and generating answer...\n")
        answer, sources, verdicts = ask(query)

        print("ANSWER:")
        print(answer)
        print("\nSOURCES:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['source_type']} #{s.get('source_number')}: {s.get('title', '')} -> {s['url']}")

        flagged = [v for v in verdicts if v and v["verdict"] not in (None, "SUPPORTED")]
        if flagged:
            print("\n⚠️  FLAGGED CLAIMS:")
            for v in flagged:
                print(f"  [{v['verdict']}] {v['sentence']}")
                if v.get("reason"):
                    print(f"      reason: {v['reason']}")

        print("\n" + "=" * 80 + "\n")
