"""
LangChain orchestration layer for PatchContext.

This module does NOT reimplement retrieval, generation, or the hallucination
guard — it wraps the already-tested functions from query.py in LangChain's
Runnable (LCEL) composition pattern, so the pipeline is genuinely orchestrated
by LangChain rather than by plain sequential function calls.

Deliberately kept separate from agentic_ask.py and app.py so the existing,
verified pipeline is untouched. This is an additive orchestration layer.
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.runnables import RunnableLambda

from query import (  # noqa: E402
    load_chunks, load_faiss, retrieve, generate_answer, check_claims
)

# --- Load shared resources once ---
_chunks = load_chunks()
_chunks_by_id = {c["chunk_id"]: c for c in _chunks}
_index, _meta = load_faiss()


# --- Step 1: Retrieval (wraps query.py's real MMR-over-FAISS retrieval) ---
def _retrieve_step(inputs: dict) -> dict:
    query = inputs["query"]
    retrieved_chunks = retrieve(query, _index, _meta, _chunks_by_id)
    return {"query": query, "retrieved_chunks": retrieved_chunks}


# --- Step 2: Generation (wraps query.py's real Gemini generation w/ model fallback) ---
def _generate_step(inputs: dict) -> dict:
    query = inputs["query"]
    retrieved_chunks = inputs["retrieved_chunks"]

    if not retrieved_chunks:
        return {
            "query": query,
            "retrieved_chunks": [],
            "answer": "No relevant information found in the indexed data.",
        }

    answer = generate_answer(query, retrieved_chunks)
    return {"query": query, "retrieved_chunks": retrieved_chunks, "answer": answer}


# --- Step 3: Verification (wraps query.py's real NLI hallucination guard) ---
def _verify_step(inputs: dict) -> dict:
    query = inputs["query"]
    retrieved_chunks = inputs["retrieved_chunks"]
    answer = inputs["answer"]

    flagged = check_claims(answer, retrieved_chunks) if retrieved_chunks else []

    sources = [
        {
            "label": f"[{i + 1}]",
            "source_type": c.get("source_type"),
            "source_number": c.get("source_number"),
            "title": c.get("title", ""),
            "url": c.get("url", "#"),
        }
        for i, c in enumerate(retrieved_chunks)
    ]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "flagged_claims": flagged,
    }


# --- Compose the LangChain LCEL pipeline: retrieve -> generate -> verify ---
patchcontext_chain = (
    RunnableLambda(_retrieve_step)
    | RunnableLambda(_generate_step)
    | RunnableLambda(_verify_step)
)


def ask_via_langchain(query: str) -> dict:
    """Run a question through the LangChain-orchestrated pipeline."""
    return patchcontext_chain.invoke({"query": query})


if __name__ == "__main__":
    print("PatchContext - LangChain-orchestrated pipeline (LCEL Runnables)")
    print("(Type 'quit' to exit)\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        print("\nRunning LangChain-composed chain (retrieve | generate | verify)...\n")
        result = patchcontext_chain.invoke({"query": q})

        print("ANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  {s['label']} {s['source_type']} #{s.get('source_number')}: {s.get('title', '')} -> {s['url']}")

        if result["flagged_claims"]:
            print("\nHALLUCINATION GUARD - flagged claims:")
            for f in result["flagged_claims"]:
                print(f"  {f['citation']} \"{f['sentence'][:100]}...\" (label={f.get('label')})")
        else:
            print("\nHALLUCINATION GUARD: no claims flagged.")

        print("\n" + "=" * 80 + "\n")
