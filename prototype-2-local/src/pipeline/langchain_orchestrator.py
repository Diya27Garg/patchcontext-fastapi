"""
LangChain orchestration layer for PatchContext.

This is now the PRODUCTION pipeline used by app.py. It wraps the
already-tested functions from query.py in LangChain's Runnable (LCEL)
composition pattern: retrieve -> generate -> verify -> self_correct.

The self-correction/rollback step ports the agentic loop originally in
agentic_ask.py (targeted re-retrieval on flagged claims, regenerate, accept
the revision only if it strictly reduces flagged claims, otherwise roll
back).
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.runnables import RunnableLambda

import query as query_module  # noqa: E402
from query import (  # noqa: E402
    load_chunks, load_faiss, retrieve, generate_answer, check_claims, client,
    GEN_MODEL_CANDIDATES,
)

MAX_CORRECTIONS = 1

_chunks = load_chunks()
_chunks_by_id = {c["chunk_id"]: c for c in _chunks}
_index, _meta = load_faiss()


def _build_correction_prompt(original_query, original_answer, flagged, all_chunks_used, new_chunks):
    flagged_summary = "\n".join(
        f"- \"{f['sentence']}\" (citation {f['citation']}, guard label: {f.get('label', 'unclear')})"
        for f in flagged
    )
    combined_context = all_chunks_used + [c for c in new_chunks if c not in all_chunks_used]
    context_blocks = []
    for i, c in enumerate(combined_context, 1):
        label = f"[{i}] {c['source_type'].upper()} #{c.get('source_number')}: {c.get('title', '')}"
        context_blocks.append(f"{label}\nURL: {c['url']}\n{c['text'][:1500]}")
    context_text = "\n\n---\n\n".join(context_blocks)

    return f"""You are PatchContext, revising a previous answer after a fact-check flagged some claims as weakly supported.

ORIGINAL QUESTION: {original_query}

ORIGINAL ANSWER:
{original_answer}

A verification step flagged these specific claims as weakly supported by their cited source:
{flagged_summary}

You now have access to ADDITIONAL retrieved context (including new evidence found specifically to check the flagged claims). Using ALL the context below, produce a REVISED answer:
- Keep claims that remain well-supported.
- Fix, soften, or REMOVE claims that still are not clearly supported by the context, even after this additional retrieval.
- Continue citing every claim using [N] matching the context blocks below (renumbered from scratch).
- If a flagged claim still cannot be verified, simply omit it or state the uncertainty briefly - do NOT add any commentary about "the flagging process", "verification", or reference this revision process at all. Write ONLY the final answer itself, as if it were the first and only answer, with no meta-discussion.

CONTEXT:
{context_text}

REVISED ANSWER (with inline [N] citations):"""


def _retrieve_step(inputs: dict) -> dict:
    query = inputs["query"]
    retrieved_chunks = retrieve(query, _index, _meta, _chunks_by_id)
    return {"query": query, "retrieved_chunks": retrieved_chunks}


def _generate_step(inputs: dict) -> dict:
    query = inputs["query"]
    retrieved_chunks = inputs["retrieved_chunks"]
    if not retrieved_chunks:
        return {"query": query, "retrieved_chunks": [], "answer": "No relevant information found in the indexed data.", "empty": True}
    answer = generate_answer(query, retrieved_chunks)
    return {"query": query, "retrieved_chunks": retrieved_chunks, "answer": answer, "empty": False}


def _verify_step(inputs: dict) -> dict:
    if inputs.get("empty"):
        inputs["flagged"] = []
        return inputs
    inputs["flagged"] = check_claims(inputs["answer"], inputs["retrieved_chunks"])
    return inputs


def _self_correct_step(inputs: dict) -> dict:
    query = inputs["query"]
    trace = {"query": query, "iterations": []}

    if inputs.get("empty"):
        trace.update({
            "final_answer": inputs["answer"], "final_flagged": [], "sources": [],
            "flagged_claims": [], "corrected": False, "rolled_back": False,
            "total_correction_rounds": 0,
        })
        return trace

    all_chunks_used = list(inputs["retrieved_chunks"])
    current_answer = inputs["answer"]
    current_flagged = inputs["flagged"]

    trace["iterations"].append({
        "stage": "initial_generation", "answer": current_answer,
        "sources_used": len(all_chunks_used), "flagged_count": len(current_flagged),
        "flagged": current_flagged,
    })

    for correction_round in range(MAX_CORRECTIONS):
        if not current_flagged:
            break
        print(f"  [agent] {len(current_flagged)} claim(s) flagged -> triggering targeted re-retrieval (round {correction_round + 1})")

        flagged_text = " ".join(f["sentence"] for f in current_flagged)
        correction_query = f"{query} {flagged_text}"[:500]
        new_chunks = retrieve(correction_query, _index, _meta, _chunks_by_id, top_k=4)
        correction_prompt = _build_correction_prompt(query, current_answer, current_flagged, all_chunks_used, new_chunks)

        models_to_try = [query_module._working_gen_model] if query_module._working_gen_model else GEN_MODEL_CANDIDATES
        revised_answer = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(model=model_name, contents=correction_prompt)
                revised_answer = response.text
                query_module._working_gen_model = model_name
                break
            except Exception:
                continue

        if revised_answer is None:
            print("  [agent] correction generation failed, keeping previous answer")
            break

        combined_context_for_check = all_chunks_used + [c for c in new_chunks if c not in all_chunks_used]
        revised_flagged = check_claims(revised_answer, combined_context_for_check)

        trace["iterations"].append({
            "stage": f"correction_round_{correction_round + 1}",
            "reformulated_query": correction_query, "new_chunks_retrieved": len(new_chunks),
            "answer": revised_answer, "flagged_count": len(revised_flagged), "flagged": revised_flagged,
        })

        print(f"  [agent] after correction: {len(revised_flagged)} claim(s) flagged (was {len(current_flagged)})")

        if len(revised_flagged) < len(current_flagged):
            print("  [agent] correction IMPROVED the answer -> accepting revision")
            all_chunks_used += [c for c in new_chunks if c not in all_chunks_used]
            current_answer = revised_answer
            current_flagged = revised_flagged
            trace["iterations"][-1]["accepted"] = True
        else:
            print("  [agent] correction did NOT improve the answer -> rolling back to previous answer")
            trace["iterations"][-1]["accepted"] = False
            break

    trace["final_answer"] = current_answer
    trace["final_flagged"] = current_flagged
    trace["total_correction_rounds"] = len(trace["iterations"]) - 1
    trace["sources"] = [
        {"label": f"[{i + 1}]", "source_type": c.get("source_type"), "source_number": c.get("source_number"),
         "title": c.get("title", ""), "url": c.get("url", "#")}
        for i, c in enumerate(all_chunks_used)
    ]
    trace["flagged_claims"] = current_flagged
    accepted_any = any(it.get("accepted") is True for it in trace["iterations"])
    rejected_any = any(it.get("accepted") is False for it in trace["iterations"])
    trace["corrected"] = accepted_any
    trace["rolled_back"] = rejected_any and not accepted_any
    return trace


patchcontext_chain = (
    RunnableLambda(_retrieve_step)
    | RunnableLambda(_generate_step)
    | RunnableLambda(_verify_step)
    | RunnableLambda(_self_correct_step)
)


def ask_via_langchain(query: str) -> dict:
    """Run a question through the LangChain-orchestrated production pipeline."""
    return patchcontext_chain.invoke({"query": query})


if __name__ == "__main__":
    print("PatchContext - LangChain-orchestrated pipeline (LCEL Runnables, with self-correction)")
    print("(Type 'quit' to exit)\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        print("\nRunning LangChain-composed chain (retrieve | generate | verify | self_correct)...\n")
        trace = patchcontext_chain.invoke({"query": q})
        for step in trace["iterations"]:
            print(f"--- {step['stage']} ---")
            if "reformulated_query" in step:
                print(f"Reformulated query: {step['reformulated_query']}")
                print(f"New chunks retrieved: {step['new_chunks_retrieved']}")
            print(f"Flagged claims: {step['flagged_count']}\n")
        print("FINAL ANSWER:")
        print(trace["final_answer"])
        print(f"\nFinal flagged claims remaining: {len(trace['final_flagged'])}")
        print("\n" + "=" * 80 + "\n")
