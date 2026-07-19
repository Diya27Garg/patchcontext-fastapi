# PatchContext

**🔗 Live demo:** [PASTE YOUR STREAMLIT CLOUD URL HERE]

> **Note:** This repository contains two prototypes built during development.
> **`prototype-2-local/` is the final, current version** and the one that
> should be run and evaluated. `prototype-1-quota-gemini/` is preserved for
> transparency to document the engineering path, but is deprecated.

---

## Internship Context

This project was built as part of the **Celebal Excellence Internship
Program**, in the **Data Science** domain, during the internship period
**25 May 2026 – 25 July 2026**.

---

## 1. Overview

PatchContext is a Retrieval-Augmented Generation (RAG) system built over the
**FastAPI** GitHub repository's commit history, pull requests, and issue
threads. It answers design-rationale questions — such as *"why does FastAPI
use `Depends()` for dependency injection?"* — with answers grounded directly
in real developer discussions, and every claim in the answer is cited back to
its exact commit SHA, pull request, or issue.

Rather than treating documentation as the source of truth, PatchContext treats
the actual conversations between maintainers and contributors as the ground
truth — capturing the reasoning and trade-offs behind design decisions that
official docs typically don't explain.

---

## 2. Repository Structure

```
patchcontext-fastapi/
├── README.md                      <- this file
├── prototype-1-quota-gemini/      <- deprecated first prototype
│   └── README.md
└── prototype-2-local/             <- FINAL, current version
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── app.py                     <- Streamlit UI
    ├── compute_metrics.py
    ├── run_eval.py
    ├── check_adversarial.py
    ├── data/
    │   ├── eval_questions.json
    │   ├── eval_results.json
    │   └── processed/
    │       └── chunks.json
    └── src/
        └── pipeline/
            ├── query.py
            ├── agentic_ask.py
            ├── chunk_data.py
            └── faiss_index.py
```

---

## 3. Architecture (Prototype 2 — Final Version)

| Component | Technology | Role |
|---|---|---|
| Vector store | FAISS | Stores embeddings of chunked commit/PR/issue text for similarity search |
| Retrieval | MMR (Maximal Marginal Relevance) | Selects diverse, non-redundant chunks instead of near-duplicate results |
| Embeddings | Local `sentence-transformers` model | Runs on-device — removes dependency on an external embedding API |
| Generation | Gemini (`gemini-flash-lite-latest`) via `google-genai` | Produces the final cited answer from retrieved context |
| Hallucination guard | Local NLI (Natural Language Inference) model | Cross-checks every cited claim against its source text and flags unsupported ones |
| Orchestration | LangChain / LangGraph | Coordinates the retrieval → generation → verification flow |
| Agentic self-correction | `agentic_ask.py` | Re-verifies its own answer and can trigger a targeted correction pass |
| UI | Streamlit (`app.py`) | Interactive front-end — question input, cited sources, self-correction trace, and per-session question history |

### How the agentic self-correction loop works

1. Retrieve context for the question and generate an initial answer.
2. Run the NLI guard on the answer; if any claims are flagged as weakly
   supported, reformulate a targeted query built from the flagged claim text
   and retrieve additional evidence specifically for those claims.
3. Regenerate a revised answer using the combined original + new context.
4. Re-run the guard on the revised answer and **compare the number of flagged
   claims before and after**.
5. **Accept the revision only if it reduces the number of flagged claims.**
   If the revision does not improve on the original (equal or more flags), the
   correction is rejected and the system rolls back to the original answer.

This means the system doesn't just "try again and hope" — it makes a measured
accept/reject decision based on an explicit, checkable signal.

---

## 4. Design Decisions & the Two-Prototype Story

### Why two prototypes exist

The first prototype (`prototype-1-quota-gemini/`) used the Gemini API for
both embeddings and generation. It worked correctly, but ran into Gemini API
rate limits, which made it impractical to reliably index and evaluate the
full FastAPI commit/PR/issue history at scale.

**The pivot:** Prototype 2 moved the embedding step to a local
`sentence-transformers` model, removing the API-quota bottleneck for that
stage entirely, while keeping Gemini for the final answer generation step
(a lighter, less frequent call than embedding every chunk of a full commit
history). The hallucination guard was also implemented as a local NLI model
for the same reason — to keep verification fast and independent of API limits.

Prototype 1 is kept in this repository, unmodified, to document this decision
honestly rather than presenting only the finished result.

---

## 5. What Makes This Project Distinct (USP)

- **Answers are grounded in actual developer discussions**, not just
  documentation — capturing the reasoning behind a design decision, not just
  what the design is.
- **A working hallucination guard that was actually evaluated**, not just
  built. Every flagged claim across a 20-question benchmark was manually
  checked against its real GitHub source, producing an honest confusion
  matrix (see Section 6) rather than an unverified claim of "hallucination
  detection."
- **A genuinely agentic self-correction loop** with a measurable accept/reject
  decision (flagged-claim count before vs. after), not a single-pass
  generate-and-hope pipeline.
- **Transparent engineering history** — the quota-to-local pivot is preserved
  and documented rather than hidden behind only the final version.

---

## 6. Evaluation Methodology & Results

The hallucination guard was evaluated on a 20-question benchmark spanning
FastAPI design rationale, performance, dependency injection, authentication,
testing/tooling, and adversarial (nonsense) questions.

**Method:** for every claim the guard flagged, the cited GitHub source (issue
or PR) was read directly and the claim was manually judged as:
- a **true positive** — the guard correctly caught a genuinely unsupported claim, or
- a **false positive** — the guard incorrectly flagged a claim that was actually supported by the source.

Unflagged claims were also spot-checked for missed hallucinations.

### Confusion Matrix (104 total citable claims across 20 questions)

|  | Guard flagged | Guard passed |
|---|---|---|
| **Actually a hallucination** | TP = 8 | FN = 0 |
| **Actually supported** | FP = 17 | TN = 79 |

| Metric | Value |
|---|---|
| Precision | 32.00% |
| Recall | 100.00% |
| F1 score | 48.48% |

### Interpretation

The guard's main weakness is **over-flagging** (low precision), not
under-flagging (recall is high). Reading the false positives directly showed
a clear pattern: the NLI model frequently misfires on **negated phrasing**
(e.g., "FastAPI does *not* have an explicit feature for X") and on **correct
paraphrases with low lexical overlap** to the source text, treating both as
contradictions even when the underlying meaning is accurate.

The true positives, by contrast, were genuine catches — for example, a
fabricated code example that does not appear anywhere in its cited pull
request, and an invented causal explanation with no textual support in its
source. This shows the guard works as intended for outright fabrications, and
the clearest next improvement is a negation-aware or paraphrase-tolerant
re-check to reduce false alarms.

**Caveat on recall:** the 100% figure reflects that manual review did not find
additional hallucinations among the unflagged claims during a skim-level
check — it is not a claim that all 79 unflagged claims were exhaustively
re-verified word-for-word against their sources.

### Overall Answer Quality

Separately from the claim-level guard evaluation above, each of the 20
answers was also judged holistically — is the answer as a whole correct, and
are its citations accurate?

| | Count | Share |
|---|---|---|
| Fully correct | 15 | 75% |
| Partially correct | 5 | 25% |
| Incorrect | 0 | 0% |
| **Citation accuracy** | **15 / 20** | **75.0%** |

No answer was judged fully incorrect, and the majority (15/20) were both
correct and correctly cited. The remaining 5 "partially correct" answers are
consistent with the guard's known over-flagging behavior described above —
a partially correct judgment typically reflects one shaky or over-cautious
claim within an otherwise sound answer, not a wrong answer overall.

Run `python compute_metrics.py` inside `prototype-2-local/` to reproduce both
the confusion matrix and this answer-quality breakdown together.

---

## 7. Setup & Running (Prototype 2)

```bash
cd prototype-2-local
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real values:
```
GITHUB_TOKEN=your_github_personal_access_token
GEMINI_API_KEY_2=your_gemini_api_key
```

### Ask a single question

```bash
python src/pipeline/query.py "why does fastapi use depends"
```

### Run the full agentic self-correction pipeline (interactive)

```bash
python src/pipeline/agentic_ask.py
```

### Reproduce the evaluation results

```bash
python compute_metrics.py
```

### Run the Streamlit UI

A Streamlit interface is included for interactive use, run from inside
`prototype-2-local/`:

```bash
python -m streamlit run app.py
```

This opens the app in your browser (typically `http://localhost:8501`). The
UI lets you type a question, watch the retrieval/generation steps as they
happen, and view the final answer alongside its cited sources and the
self-correction trace described in Section 3. Each question asked in a
session is kept in a sidebar list for quick reference, alongside a few sample
questions to try and a short primer on FastAPI itself.

> **Live version:** the app is also deployed on
> [Streamlit Community Cloud](https://streamlit.io/cloud) — see the link at
> the top of this README. Running it locally (below) is still the recommended
> way to reproduce results, since the hosted version depends on the deployed
> environment's own resource limits.

![Streamlit UI screenshot](docs/streamlit_ui.png)

---

## 8. Known Limitations

- The hallucination guard currently has low precision (32%) and would
  over-flag valid claims in practical use; a negation-aware refinement is the
  clearest next step.
- A full RAGAs-suite evaluation (faithfulness, answer relevancy, context
  precision/recall) has not yet been run. The evaluation in Section 6 is a
  custom, guard-specific confusion matrix, not the full RAGAs benchmark
  originally scoped for this project.
- `prototype-1-quota-gemini/` is preserved for transparency only and is not
  maintained or intended to be run going forward.

## 9. What Would Come Next

- Add a negation-aware or paraphrase-tolerant re-check step to the guard to
  reduce false positives.
- Run the full RAGAs evaluation suite across a larger benchmark.
- Expand manual verification of unflagged claims beyond a quick skim, to
  tighten confidence in the recall figure.
