import json

RESULTS_PATH = "data/eval_results.json"


def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    results = load_results()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_claims = 0
    unannotated = []

    correctness_counts = {"yes": 0, "partial": 0, "no": 0}
    citation_ok_count = 0
    citation_total = 0

    for r in results:
        if r.get("total_citable_claims") is None:
            unannotated.append(r["id"])
            continue

        tp = r.get("guard_true_positives") or 0
        fp = r.get("guard_false_positives") or 0
        fn = r.get("missed_hallucinations") or 0
        total_claims_this_q = r.get("total_citable_claims") or 0

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_claims += total_claims_this_q

        # manual_correctness can be stored as a boolean (True/False) or as the
        # string "partial" for partially-correct answers. Handle both.
        correctness = r.get("manual_correctness")
        if correctness is True:
            correctness_counts["yes"] += 1
        elif correctness is False:
            correctness_counts["no"] += 1
        elif correctness == "partial":
            correctness_counts["partial"] += 1
        elif correctness in correctness_counts:
            # fallback for legacy string-encoded values ("yes"/"partial"/"no")
            correctness_counts[correctness] += 1

        # manual_citation_ok can be stored as a boolean or as "yes"/"no" strings.
        citation_ok = r.get("manual_citation_ok")
        if citation_ok is not None:
            citation_total += 1
            if citation_ok is True or citation_ok == "yes":
                citation_ok_count += 1

    if unannotated:
        print(f"WARNING: {len(unannotated)} questions not yet annotated (ids: {unannotated})")
        print("Fill in eval_results.json before running this for final numbers.\n")

    total_tn = total_claims - total_tp - total_fp - total_fn

    print("=" * 60)
    print("HALLUCINATION GUARD - CONFUSION MATRIX")
    print("=" * 60)
    print(f"{'':20}{'Predicted: Flagged':>20}{'Predicted: OK':>20}")
    print(f"{'Actual: Bad claim':20}{total_tp:>20}{total_fn:>20}   <- recall row")
    print(f"{'Actual: Good claim':20}{total_fp:>20}{total_tn:>20}")
    print()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Total citable claims reviewed: {total_claims}")
    print(f"  True Positives  (correctly flagged bad claims):  {total_tp}")
    print(f"  False Positives (wrongly flagged good claims):   {total_fp}")
    print(f"  False Negatives (missed bad claims):              {total_fn}")
    print(f"  True Negatives  (correctly left good claims alone): {total_tn}")
    print()
    print(f"Precision: {precision:.2%}  (of flagged claims, % that were genuinely bad)")
    print(f"Recall:    {recall:.2%}  (of genuinely bad claims, % that were caught)")
    print(f"F1 score:  {f1:.2%}")
    print()

    print("=" * 60)
    print("OVERALL ANSWER QUALITY")
    print("=" * 60)
    print(f"Fully correct:  {correctness_counts['yes']}")
    print(f"Partially correct: {correctness_counts['partial']}")
    print(f"Incorrect: {correctness_counts['no']}")
    if citation_total > 0:
        print(f"\nCitation accuracy: {citation_ok_count}/{citation_total} ({citation_ok_count/citation_total:.1%})")
