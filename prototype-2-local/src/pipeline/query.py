import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import re
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

CHUNKS_PATH = "data/processed/chunks.json"
FAISS_INDEX_PATH = "data/processed/faiss_index.bin"
FAISS_META_PATH = "data/processed/faiss_meta.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

FETCH_K = 20
TOP_K = 6
MMR_LAMBDA = 0.7

_embed_model = None
_working_gen_model = None
_nli_pipeline = None
_nli_available = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_nli_pipeline():
    global _nli_pipeline, _nli_available
    if _nli_available is not None:
        return _nli_pipeline

    try:
        from transformers import pipeline
        print("Loading local NLI model (first run downloads it)...")
        _nli_pipeline = pipeline(
            "text-classification",
            model="typeform/distilbert-base-uncased-mnli",
        )
        _nli_available = True
        print("NLI model loaded successfully.\n")
    except Exception as e:
        print(f"NLI model failed to load ({str(e)[:150]}). Falling back to embedding-similarity guard.\n")
        _nli_pipeline = None
        _nli_available = False

    return _nli_pipeline


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def mmr_select(query_vec, candidate_vecs, candidate_indices, k, lambda_param):
    selected = []
    selected_vecs = []
    remaining = list(range(len(candidate_indices)))

    relevance = candidate_vecs @ query_vec

    while remaining and len(selected) < k:
        if not selected:
            best_local_idx = int(np.argmax(relevance[remaining]))
        else:
            selected_matrix = np.array(selected_vecs)
            diversity_penalty = candidate_vecs[remaining] @ selected_matrix.T
            max_diversity_penalty = diversity_penalty.max(axis=1)
            mmr_scores = (lambda_param * relevance[remaining]) - ((1 - lambda_param) * max_diversity_penalty)
            best_local_idx = int(np.argmax(mmr_scores))

        chosen = remaining.pop(best_local_idx)
        selected.append(chosen)
        selected_vecs.append(candidate_vecs[chosen])

    return [candidate_indices[i] for i in selected]


def retrieve(query: str, index, meta, chunks_by_id, top_k=TOP_K, fetch_k=FETCH_K):
    model = get_embed_model()
    query_vec = model.encode(query, normalize_embeddings=True).astype("float32")

    scores, faiss_positions = index.search(query_vec.reshape(1, -1), fetch_k)
    faiss_positions = faiss_positions[0]
    valid = [p for p in faiss_positions if p != -1]

    if not valid:
        return []

    candidate_vecs = np.array([meta["embeddings"][p] for p in valid], dtype="float32")
    candidate_chunk_ids = [meta["chunk_ids"][p] for p in valid]

    selected_chunk_ids = mmr_select(query_vec, candidate_vecs, candidate_chunk_ids, top_k, MMR_LAMBDA)

    results = []
    for cid in selected_chunk_ids:
        chunk = chunks_by_id[cid]
        results.append(chunk)
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


def generate_answer(query: str, retrieved_chunks):
    global _working_gen_model
    prompt = build_prompt(query, retrieved_chunks)

    models_to_try = [_working_gen_model] if _working_gen_model else GEN_MODEL_CANDIDATES

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if _working_gen_model != model_name:
                print(f"  (using generation model: {model_name})")
                _working_gen_model = model_name
            return response.text
        except Exception as e:
            last_error = e
            print(f"  Model '{model_name}' failed: {str(e)[:150]}")
            continue

    raise RuntimeError(f"All candidate models failed. Last error: {last_error}")


def split_into_claims(text: str):
    """
    Split answer text into clean claim candidates, avoiding markdown artifacts.
    Splits on newlines first (so bullet points don't merge together), then
    strips markdown bullet/bold markers, then splits on sentence boundaries,
    then further splits long compound sentences on internal clause
    boundaries so the NLI guard judges each clause independently instead of
    one giant multi-claim blob.
    """
    # Conjunctions/markers that commonly join two independent claims within
    # one sentence. Matched only when preceded by a comma or semicolon, to
    # avoid splitting on these words mid-clause (e.g. "systems like .NET").
    CLAUSE_SPLIT_PATTERN = re.compile(
        r'(?:,\s+|;\s+)(?=(?:while|however|although|though|whereas|but|'
        r'in contrast|conversely|meanwhile|additionally|furthermore|'
        r'on the other hand)\b)',
        flags=re.IGNORECASE,
    )
 
    claims = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\*\-]\s+', '', line)   # strip leading bullet markers
        line = re.sub(r'\*\*', '', line)          # strip bold markers
        if not line:
            continue
        for sentence in re.split(r'(?<=[.!?])\s+', line):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Further split long compound sentences on internal clause
            # boundaries (only if the sentence is long enough that it's
            # likely carrying more than one claim — short sentences are
            # left untouched to avoid over-splitting).
            if len(sentence) > 120:
                sub_clauses = CLAUSE_SPLIT_PATTERN.split(sentence)
                for clause in sub_clauses:
                    clause = clause.strip()
                    if clause:
                        claims.append(clause)
            else:
                claims.append(sentence)
    return claims


def check_claims_nli(answer: str, retrieved_chunks, nli):
    sentences = split_into_claims(answer)
    flagged = []

    for sentence in sentences:
        citation_matches = re.findall(r'\[(\d+)\]', sentence)
        if not citation_matches:
            continue

        clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
        if len(clean_sentence) < 15:
            continue

        for cite_num in citation_matches:
            idx = int(cite_num) - 1
            if idx < 0 or idx >= len(retrieved_chunks):
                continue

            source_text = retrieved_chunks[idx]["text"][:800]
            pair_input = f"{source_text} </s> {clean_sentence}"

            try:
                result = nli(pair_input, truncation=True)[0]
                label = result["label"].lower()
                score = result["score"]
            except Exception:
                continue

            if "contradiction" in label or ("neutral" in label and score > 0.7):
                flagged.append({
                    "sentence": clean_sentence,
                    "citation": f"[{cite_num}]",
                    "label": label,
                    "confidence": round(float(score), 3)
                })

    return flagged


def check_claims_fallback(answer: str, retrieved_chunks, embed_model):
    sentences = split_into_claims(answer)
    flagged = []

    for sentence in sentences:
        citation_matches = re.findall(r'\[(\d+)\]', sentence)
        if not citation_matches:
            continue

        clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
        if len(clean_sentence) < 15:
            continue

        for cite_num in citation_matches:
            idx = int(cite_num) - 1
            if idx < 0 or idx >= len(retrieved_chunks):
                continue

            source_text = retrieved_chunks[idx]["text"][:1500]
            vecs = embed_model.encode([clean_sentence, source_text], normalize_embeddings=True)
            sim = float(np.dot(vecs[0], vecs[1]))

            if sim < 0.25:
                flagged.append({
                    "sentence": clean_sentence,
                    "citation": f"[{cite_num}]",
                    "label": "low_similarity",
                    "confidence": round(sim, 3)
                })

    return flagged


def check_claims(answer: str, retrieved_chunks):
    nli = get_nli_pipeline()
    if nli is not None:
        return check_claims_nli(answer, retrieved_chunks, nli)
    else:
        return check_claims_fallback(answer, retrieved_chunks, get_embed_model())


def ask(query: str):
    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    index, meta = load_faiss()

    retrieved = retrieve(query, index, meta, chunks_by_id)
    if not retrieved:
        return "No relevant information found in the indexed data.", [], []

    answer = generate_answer(query, retrieved)
    flagged = check_claims(answer, retrieved)
    return answer, retrieved, flagged


if __name__ == "__main__":
    print("PatchContext2 - ask a question about FastAPI's design history.")
    print("(Type 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        print("\nSearching (MMR over FAISS) and generating answer...\n")
        answer, sources, flagged = ask(query)

        print("ANSWER:")
        print(answer)

        print("\nSOURCES:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['source_type']} #{s.get('source_number')}: {s.get('title', '')} -> {s['url']}")

        if flagged:
            print("\nHALLUCINATION GUARD - flagged claims (weakly supported by cited source):")
            for f in flagged:
                print(f"  {f['citation']} \"{f['sentence'][:100]}...\" (label={f['label']}, confidence={f['confidence']})")
        else:
            print("\nHALLUCINATION GUARD: all cited claims appear well-supported.")

        print("\n" + "=" * 80 + "\n")
