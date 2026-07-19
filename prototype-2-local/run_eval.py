import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import sys

sys.path.insert(0, "src/pipeline")
from query import ask  # noqa: E402

QUESTIONS_PATH = "data/eval_questions.json"
RESULTS_PATH = "data/eval_results.json"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    questions = load_questions()
    print(f"Running {len(questions)} evaluation questions...\n")

    results = []
    for q in questions:
        print(f"[{q['id']}/{len(questions)}] ({q['category']}) {q['question']}")
        try:
            answer, sources, flagged = ask(q["question"])
        except Exception as e:
            print(f"  ERROR: {e}")
            answer, sources, flagged = f"[ERROR: {e}]", [], []

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "sources": [
                {"label": f"[{i+1}]", "source_type": s["source_type"],
                 "source_number": s.get("source_number"), "url": s["url"]}
                for i, s in enumerate(sources)
            ],
            "flagged_claims": flagged,

            "manual_correctness": None,
            "manual_citation_ok": None,

            "total_citable_claims": None,
            "guard_true_positives": None,
            "guard_false_positives": None,
            "missed_hallucinations": None,

            "notes": ""
        })
        print(f"  -> {len(sources)} sources, {len(flagged)} flagged claims\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(results)} results to {RESULTS_PATH}")
    print("\nNext: open eval_results.json and fill in the manual_* and guard_* fields for each question.")
    print("For the confusion matrix fields specifically:")
    print("  1. Count total_citable_claims = every sentence in the answer with a [N] citation")
    print("  2. For each FLAGGED claim, decide: genuinely bad (true_positive) or actually fine (false_positive)")
    print("  3. For UNFLAGGED claims, check if any are actually wrong/unsupported -> missed_hallucinations")
