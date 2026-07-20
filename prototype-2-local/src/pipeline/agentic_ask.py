print(">>> LOADED FIXED VERSION OF agentic_ask.py <<<")
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query import (  # noqa: E402
    load_chunks, load_faiss, retrieve, generate_answer, check_claims,
    get_embed_model
)


def build_correction_prompt(original_query, original_answer, flagged, all_chunks_used, new_chunks):
    """Ask the model to specifically revise the flagged claims using new evidence."""
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

    prompt = f"""You are PatchContext, revising a previous answer after a fact-check flagged some claims as weakly supported.

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
    return prompt


def ask_with_self_correction(query: str, max_corrections: int = 1):
    """
    Agentic loop: retrieve -> generate -> verify (NLI guard) -> if flagged,
    reformulate a targeted query, retrieve again, and regenerate a corrected answer.

    Returns a log of every stage, so the whole reasoning trace is inspectable.
    """
    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    index, meta = load_faiss()

    trace = {"query": query, "iterations": []}

    # --- Initial pass ---
    retrieved = retrieve(query, index, meta, chunks_by_id)
    if not retrieved:
        trace["final_answer"] = "No relevant information found in the indexed data."
        trace["final_flagged"] = []
        return trace

    answer = generate_answer(query, retrieved)
    flagged = check_claims(answer, retrieved)

    trace["iterations"].append({
        "stage": "initial_generation",
        "answer": answer,
        "sources_used": len(retrieved),
        "flagged_count": len(flagged),
        "flagged": flagged,
    })

    all_chunks_used = list(retrieved)
    current_answer = answer
    current_flagged = flagged

    # --- Self-correction loop ---
    for correction_round in range(max_corrections):
        if not current_flagged:
            break  # nothing to fix, agent is satisfied

        print(f"  [agent] {len(current_flagged)} claim(s) flagged -> triggering targeted re-retrieval (round {correction_round + 1})")

        # Reformulate query: combine original question with the flagged claim text
        # so retrieval specifically searches for evidence about the disputed claims.
        flagged_text = " ".join(f["sentence"] for f in current_flagged)
        correction_query = f"{query} {flagged_text}"[:500]

        new_chunks = retrieve(correction_query, index, meta, chunks_by_id, top_k=4)

        correction_prompt = build_correction_prompt(
            query, current_answer, current_flagged, all_chunks_used, new_chunks
        )

        # Reuse generate_answer's model-fallback logic by calling the underlying
        # client directly through the same prompt-building path used elsewhere.
        from query import client, _working_gen_model, GEN_MODEL_CANDIDATES
        import query as query_module

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
            "reformulated_query": correction_query,
            "new_chunks_retrieved": len(new_chunks),
            "answer": revised_answer,
            "flagged_count": len(revised_flagged),
            "flagged": revised_flagged,
        })

        print(f"  [agent] after correction: {len(revised_flagged)} claim(s) flagged "
              f"(was {len(current_flagged)})")

        # --- Agentic decision: only accept the revision if it actually helped ---
        if len(revised_flagged) < len(current_flagged):
            print(f"  [agent] correction IMPROVED the answer -> accepting revision")
            all_chunks_used += [c for c in new_chunks if c not in all_chunks_used]
            current_answer = revised_answer
            current_flagged = revised_flagged
            trace["iterations"][-1]["accepted"] = True
        else:
            print(f"  [agent] correction did NOT improve the answer -> rolling back to previous answer")
            trace["iterations"][-1]["accepted"] = False
            break  # keep current_answer/current_flagged as they were, stop looping

    trace["final_answer"] = current_answer
    trace["final_flagged"] = current_flagged
    trace["total_correction_rounds"] = len(trace["iterations"]) - 1

    # --- Fields expected by the Streamlit UI (app.py) ---
    trace["sources"] = [
        {
            "label": f"[{i+1}]",
            "source_type": c.get("source_type"),
            "source_number": c.get("source_number"),
            "title": c.get("title", ""),
            "url": c.get("url", "#"),
        }
        for i, c in enumerate(all_chunks_used)
    ]
    trace["flagged_claims"] = current_flagged

    accepted_any = any(it.get("accepted") is True for it in trace["iterations"])
    rejected_any = any(it.get("accepted") is False for it in trace["iterations"])
    trace["corrected"] = accepted_any
    trace["rolled_back"] = rejected_any and not accepted_any

    return trace


if __name__ == "__main__":
    print("PatchContext2 - Agentic self-correction demo")
    print("(Type 'quit' to exit)\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        print("\nRunning agentic pipeline (retrieve -> generate -> verify -> correct if needed)...\n")
        trace = ask_with_self_correction(q, max_corrections=1)

        for i, step in enumerate(trace["iterations"]):
            print(f"--- {step['stage']} ---")
            if "reformulated_query" in step:
                print(f"Reformulated query: {step['reformulated_query']}")
                print(f"New chunks retrieved: {step['new_chunks_retrieved']}")
            print(f"Flagged claims: {step['flagged_count']}")
            print()

        print("FINAL ANSWER:")
        print(trace["final_answer"])
        print(f"\nFinal flagged claims remaining: {len(trace['final_flagged'])}")
        if trace["final_flagged"]:
            for f in trace["final_flagged"]:
                print(f"  - \"{f['sentence'][:100]}...\" (label={f.get('label')})")
        print("\n" + "=" * 80 + "\n")
