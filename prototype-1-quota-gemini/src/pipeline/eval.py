"""
Eval harness for PatchContext.

Runs a fixed set of questions through the real pipeline (retrieval +
generation + faithfulness guard) and reports:
  - retrieval hit rate: did the expected source(s) actually get retrieved
  - citation coverage: did the generated answer cite a retrieved source at all
  - guard flag rate: how often the faithfulness guard flagged a claim
  - per-question detail, so you can eyeball failures, not just the average

Usage:
    python eval.py
    python eval.py --questions eval_questions.json
"""

import argparse
import json
import time

from query import ask


def source_label(chunk) -> str:
    return f"{chunk['source_type']}#{chunk.get('source_number')}"


def run_eval(questions_path: str):
    with open(questions_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        q = case["question"]
        expected = set(case.get("expected_sources", []))

        print(f"\n{'='*80}\nQ: {q}")
        start = time.time()

        try:
            answer, sources, verdicts = ask(q)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": q, "error": str(e),
                "retrieval_hit": False, "citation_used": False,
                "flagged_count": 0,
            })
            continue

        elapsed = time.time() - start
        retrieved_labels = {source_label(s) for s in sources}
        hit = bool(expected & retrieved_labels) if expected else None

        citation_used = any(f"[{i}]" in answer for i in range(1, 10))

        flagged = [v for v in verdicts if v and v["verdict"] not in (None, "SUPPORTED")]

        print(f"  Retrieved:  {sorted(retrieved_labels)}")
        print(f"  Expected:   {sorted(expected) if expected else '(not specified)'}")
        print(f"  Hit:        {hit}")
        print(f"  Cited:      {citation_used}")
        print(f"  Flagged:    {len(flagged)} claim(s)")
        print(f"  Time:       {elapsed:.1f}s")

        results.append({
            "question": q,
            "expected": sorted(expected),
            "retrieved": sorted(retrieved_labels),
            "retrieval_hit": hit,
            "citation_used": citation_used,
            "flagged_count": len(flagged),
            "time_sec": round(elapsed, 1),
        })

    n = len(results)
    errors = sum(1 for r in results if "error" in r)
    scored = [r for r in results if r.get("retrieval_hit") is not None]
    hits = sum(1 for r in scored if r["retrieval_hit"])
    cited = sum(1 for r in results if r.get("citation_used"))
    total_flagged = sum(r.get("flagged_count", 0) for r in results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"  Questions run:        {n}")
    print(f"  Errors:               {errors}")
    if scored:
        print(f"  Retrieval hit rate:   {hits}/{len(scored)} ({100*hits/len(scored):.0f}%)")
    print(f"  Citation coverage:    {cited}/{n} ({100*cited/n:.0f}%)")
    print(f"  Total flagged claims: {total_flagged}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nFull results written to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval_questions.json")
    args = parser.parse_args()
    run_eval(args.questions)