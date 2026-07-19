"""
Faithfulness guard for PatchContext — entailment-based hallucination detection.

Why not TF-IDF similarity: cosine similarity on word overlap measures the
wrong thing. A faithful paraphrase can score low; an unfaithful sentence
that reuses source vocabulary can score high. What actually matters is
entailment -- does the cited source text support this specific claim.

This reuses your existing Gemini call (same model that already works for
generation) to do a single structured verification pass per answer: one
extra API call, not one per sentence, so it doesn't reopen the quota
problem you hit earlier.

Written for the `google.genai` Client style you're using in query.py
(client.models.generate_content(model=..., contents=...)), not the older
GenerativeModel object API.

Integration in query.py, right after you get `answer` and `retrieved`:

    from faithfulness_guard import verify_answer, annotate_answer

    verdicts = verify_answer(answer, retrieved, client, _working_model)
    print(annotate_answer(verdicts))
"""

import json
import re

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _build_verification_prompt(sentence_source_pairs):
    """
    sentence_source_pairs: list of (sentence, source_text) tuples
    """
    items = []
    for i, (sentence, source_text) in enumerate(sentence_source_pairs):
        items.append(
            f'{i}. CLAIM: "{sentence}"\n   SOURCE: "{source_text}"'
        )
    joined = "\n\n".join(items)

    return f"""You are a strict fact-checker. For each CLAIM below, judge whether
the paired SOURCE text actually supports it. Do not use outside knowledge —
judge only whether the SOURCE justifies the CLAIM.

Respond with ONLY a JSON array, one object per numbered item, no other text:
[
  {{"index": 0, "verdict": "SUPPORTED", "reason": "brief reason"}},
  {{"index": 1, "verdict": "UNSUPPORTED", "reason": "brief reason"}}
]

Verdict must be exactly one of: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED.

{joined}
"""


def verify_answer(answer_text: str, sources: list, client, model_name: str,
                   source_text_key: str = "text"):
    """
    client: your existing genai.Client instance.
    model_name: the resolved working model name, i.e. query.py's
                `_working_model` (falls back to the first candidate if
                verification runs before generation has picked one).
    sources: your `retrieved` list -- indexable by citation number - 1,
             each item already has a "text" field.

    Returns a list of dicts: {sentence, source_idx, verdict, reason}
    Sentences without a citation are returned with verdict=None (nothing
    to check).
    """
    sentences = _split_sentences(answer_text)

    checkable = []      # (sentence_idx, sentence, source_text)
    results = [None] * len(sentences)

    for i, sentence in enumerate(sentences):
        cited = CITATION_PATTERN.findall(sentence)
        if not cited:
            results[i] = {"sentence": sentence, "source_idx": None,
                           "verdict": None, "reason": None}
            continue
        idx = int(cited[0]) - 1  # verify against first citation
        if idx < 0 or idx >= len(sources):
            results[i] = {"sentence": sentence, "source_idx": None,
                           "verdict": None, "reason": "citation index out of range"}
            continue
        source_text = sources[idx].get(source_text_key, "")
        checkable.append((i, sentence, source_text))

    if not checkable:
        return results

    prompt = _build_verification_prompt([(s, t) for _, s, t in checkable])

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    raw = response.text.strip()
    # Strip markdown fences if the model wraps the JSON
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        verdicts = json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: mark everything unverified rather than crashing the app
        verdicts = [{"index": n, "verdict": "UNVERIFIED",
                     "reason": "guard could not parse model response"}
                    for n in range(len(checkable))]

    verdict_by_index = {v["index"]: v for v in verdicts}

    for local_i, (sent_i, sentence, source_text) in enumerate(checkable):
        v = verdict_by_index.get(local_i, {})
        results[sent_i] = {
            "sentence": sentence,
            "source_idx": None,
            "verdict": v.get("verdict", "UNVERIFIED"),
            "reason": v.get("reason", ""),
        }

    return results


def annotate_answer(verdicts) -> str:
    # See bottom of file for exact query.py integration.
    """Build a display-ready version of the answer with inline flags."""
    out = []
    for v in verdicts:
        if v["verdict"] in ("UNSUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIED"):
            out.append(f"⚠️ [{v['verdict']}] {v['sentence']}")
        else:
            out.append(v["sentence"])
    return " ".join(out)
