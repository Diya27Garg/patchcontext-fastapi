import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "pipeline"))

import time
import random
import streamlit as st
from agentic_ask import ask_with_self_correction

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="PatchContext — FastAPI", page_icon="🧩", layout="centered")

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>PatchContext — FastAPI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p style='text-align: center; color: #9a9a9a; font-size: 1.05rem; margin-top: 0.3rem;'>
    FastAPI's code tells you <i>what</i> it does. This tells you <i>why</i>.<br>
    Every answer here is pulled from real commit history, pull request discussions,
    and issue threads — not guessed. If the project's own developers argued about it,
    debated it, or explained it in a comment somewhere, that's what you'll see cited below.
    </p>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# Session state setup
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []          # list of dicts: question, trace, feedback
if "mode" not in st.session_state:
    st.session_state.mode = "ask"          # "ask" or "result"
if "active_idx" not in st.session_state:
    st.session_state.active_idx = None     # index into history currently displayed
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

SAMPLE_QUESTIONS = [
    "Why does FastAPI use Depends() instead of decorators for dependency injection?",
    "Why did FastAPI add support for background tasks?",
    "Why does FastAPI rely on Pydantic instead of writing its own validation?",
    "Why was async support added early instead of being bolted on later?",
    "Why does FastAPI generate OpenAPI docs automatically?",
]

LOADING_MESSAGES = [
    "Skimming through commit history…",
    "Digging through old PR discussions…",
    "Cross-checking issue threads…",
    "Making sure this claim actually holds up…",
]

FASTAPI_NOTE = (
    "FastAPI is a Python web framework built for speed of both the API and the "
    "developer writing it. It leans heavily on type hints — for validation, "
    "serialization, and auto-generated docs — which is a big part of why so many "
    "of its design decisions trace back to how Python's typing system works, "
    "not just arbitrary framework choices."
)

SOURCE_TYPE_ICONS = {
    "issue": "🐞",
    "pull_request": "🔀",
    "commit": "📌",
}
SOURCE_TYPE_LABELS = {
    "issue": "Issue",
    "pull_request": "PR",
    "commit": "Commit",
}


def format_source_line(s: dict) -> str:
    """Build a readable, descriptive line for a source, e.g.
    '🐞 Issue #504 — Dependency Injection - Singleton?'"""
    stype = s.get("source_type")
    icon = SOURCE_TYPE_ICONS.get(stype, "📄")
    type_label = SOURCE_TYPE_LABELS.get(stype, stype or "Source")
    number = s.get("source_number")
    title = (s.get("title") or "").strip()

    base = f"{type_label} #{number}" if number is not None else type_label

    if title:
        short_title = title if len(title) <= 80 else title[:77] + "…"
        return f"{icon} **{base}** — {short_title}"
    return f"{icon} **{base}**"


# ============================================================
# Helper: run a question through the backend with a bit of personality
# ============================================================
def run_question(question: str):
    placeholder = st.empty()
    shuffled = LOADING_MESSAGES[:]
    random.shuffle(shuffled)
    for msg in shuffled:
        placeholder.markdown(f"_{msg}_")
        time.sleep(0.7)

    try:
        with st.spinner("Putting together a grounded answer…"):
            trace = ask_with_self_correction(question)
        error = None
    except Exception as e:
        trace = None
        error = str(e)

    placeholder.empty()

    st.session_state.history.append({
        "question": question,
        "trace": trace,
        "error": error,
        "feedback": None,
    })
    st.session_state.active_idx = len(st.session_state.history) - 1
    st.session_state.mode = "result"


# ============================================================
# Sidebar: session history + sample questions + FastAPI note
# ============================================================
with st.sidebar:
    st.subheader("Your questions this session")
    if not st.session_state.history:
        st.caption("Nothing asked yet — your questions will show up here as you go.")
    else:
        for i, item in enumerate(st.session_state.history):
            label = item["question"] if len(item["question"]) <= 45 else item["question"][:42] + "…"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.active_idx = i
                st.session_state.mode = "result"
                st.rerun()

    st.divider()
    st.subheader("Try one of these")
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.session_state.mode = "ask"
            st.rerun()

    st.divider()
    with st.expander("A quick note on FastAPI"):
        st.write(FASTAPI_NOTE)

# ============================================================
# Main area: ask mode vs result mode
# ============================================================
if st.session_state.mode == "ask":
    question = st.text_input(
        "Your question",
        value=st.session_state.pending_question,
        placeholder="e.g. why does FastAPI use Depends() instead of decorators for DI?",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        st.session_state.pending_question = ""
        if not question.strip():
            st.warning("Type a question first.")
        else:
            run_question(question.strip())
            st.rerun()

else:
    idx = st.session_state.active_idx
    item = st.session_state.history[idx]
    trace = item["trace"]

    st.markdown(f"**Question:** {item['question']}")
    st.divider()

    if item["error"]:
        st.error(f"Something went wrong calling the backend: {item['error']}")
    elif trace:
        # ---------- Main answer ----------
        st.subheader("Answer")
        st.write(trace.get("final_answer", "(no answer returned)"))

        # ---------- Self-correction / rollback indicator ----------
        corrected = trace.get("corrected")
        rolled_back = trace.get("rolled_back")
        if corrected is not None or rolled_back is not None:
            st.divider()
            st.subheader("Self-correction trace")
            if rolled_back:
                st.info(
                    "🔄 The model revised its answer, checked the revision, and rolled "
                    "back to the original because it was measurably worse."
                )
            elif corrected:
                st.success("✅ The model revised and improved its answer after self-checking.")
            else:
                st.caption("No revision was needed — first-pass answer was accepted as-is.")

        # ---------- Sources (descriptive, clickable) ----------
        sources = trace.get("sources", [])
        if sources:
            st.divider()
            st.subheader(f"Sources ({len(sources)})")
            for s in sources:
                if isinstance(s, dict):
                    line = format_source_line(s)
                    url = s.get("url", "#")
                    label = s.get("label", "")
                    st.markdown(f"{label} {line}  \n[{url}]({url})", unsafe_allow_html=False)
                else:
                    st.markdown(f"- {s}")

        # ---------- Hallucination guard flags ----------
        flags = trace.get("flagged_claims") or trace.get("hallucination_flags")
        if flags:
            st.divider()
            st.subheader(f"⚠️ Flagged by hallucination guard ({len(flags)})")
            st.caption(
                "These specific claims were checked against their cited source and the "
                "guard could not confirm they're well-supported. This doesn't necessarily "
                "mean they're wrong — see the project README for the guard's measured "
                "precision on this dataset."
            )
            for f in flags:
                if isinstance(f, dict):
                    citation = f.get("citation", "")
                    sentence = f.get("sentence", "")
                    label = f.get("label", "")
                    confidence = f.get("confidence")
                    conf_str = f" · confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
                    st.warning(f"**{citation}** \"{sentence}\"  \n*guard label: {label}{conf_str}*")
                else:
                    st.warning(str(f))
        elif trace.get("final_flagged") == [] and sources:
            st.divider()
            st.caption("✅ No claims were flagged — all cited statements appear well-supported by their sources.")

    # ---------- Feedback + next question ----------
    st.divider()
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("👍", key=f"thumbs_{idx}"):
            st.session_state.history[idx]["feedback"] = "up"
            st.session_state.mode = "ask"
            st.session_state.pending_question = ""
            st.rerun()
    with col2:
        st.caption("Mark helpful and ask your next question")

    if item["feedback"] == "up":
        st.success("Thanks — noted.")
