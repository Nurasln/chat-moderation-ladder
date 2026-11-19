"""Real-time chat moderation demo (Streamlit).

Run:  streamlit run app.py

- Type a chat message -> it is labeled clean / offensive with a confidence score.
- Offensive messages are masked before they are shown.
- The side panel lets you switch models and logs the response time of every message.
- Default model = the quantized INT8 BERT (requirement R5).
"""
import time

import pandas as pd
import streamlit as st

from src.config import LABELS
from src.download import ensure_models
from src.normalize import normalize
from src.predictors import ALL_PREDICTORS, BertInt8Predictor, LexiconPredictor

LATENCY_BUDGET_MS = 50
THRESHOLD = 0.5

st.set_page_config(page_title="Chat Moderation", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads them)…")
def load_predictors():
    ensure_models()
    predictors = {cls.name: cls() for cls in ALL_PREDICTORS}
    # Warm-up: the first calls of a model are slow (lazy init, caches). Do them now,
    # so the latency shown for real messages is the steady-state latency (as in the benchmark).
    for p in predictors.values():
        for _ in range(5):
            p.predict_proba("warm up message for the model")
    return predictors


def mask(text, lexicon_matcher, offensive):
    """Mask offensive words: 'you idiot' -> 'you i****'. If the model flags the message but no
    known offensive word is found, hide the whole message."""
    if not offensive:
        return text
    out, hits = [], 0
    for token in text.split():
        norm = normalize(token)
        if norm and any(lexicon_matcher.match(w) for w in norm.split()):
            out.append(token[0] + "*" * (len(token) - 1))
            hits += 1
        else:
            out.append(token)
    return " ".join(out) if hits else "🚫 [message hidden by moderation]"


def classify(predictor, text):
    start = time.perf_counter()
    p_off = predictor.predict_proba(text)
    ms = (time.perf_counter() - start) * 1000
    label = int(p_off >= THRESHOLD)
    confidence = p_off if label == 1 else 1 - p_off
    return label, confidence, ms


predictors = load_predictors()
matcher = predictors[LexiconPredictor.name].matcher
if "log" not in st.session_state:
    st.session_state.log = []

# ---------------- Side panel ----------------
with st.sidebar:
    st.header("⚙️ Model")
    names = list(predictors)
    model_name = st.radio("Classify with", names, index=names.index(BertInt8Predictor.name))
    compare = st.toggle("Also show all models for each message", value=False)

    st.header("⏱️ Response times")
    log = pd.DataFrame(st.session_state.log)
    if len(log):
        st.metric("Average (this model)",
                  f"{log[log.model == model_name].ms.mean():.1f} ms" if (log.model == model_name).any() else "–",
                  help=f"Budget: {LATENCY_BUDGET_MS} ms per message on CPU")
        st.dataframe(log[["model", "label", "confidence", "ms"]].iloc[::-1].round(2),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No messages yet.")
    if st.button("Clear chat"):
        st.session_state.log = []
        st.rerun()

# ---------------- Chat ----------------
st.title("🛡️ Real-Time Chat Moderation")
st.caption(f"English messages · clean / offensive · CPU · latency budget {LATENCY_BUDGET_MS} ms")

for entry in st.session_state.log:
    with st.chat_message("user", avatar="🚫" if entry["label"] == "offensive" else "💬"):
        st.write(entry["shown"])
        st.caption(f"**{entry['label']}** · confidence {entry['confidence']:.0%} · "
                   f"{entry['ms']:.1f} ms · {entry['model']}")
        if entry.get("all_models"):
            st.dataframe(pd.DataFrame(entry["all_models"]), hide_index=True)

if text := st.chat_input("Type a message…"):
    label, confidence, ms = classify(predictors[model_name], text)
    entry = {
        "model": model_name, "label": LABELS[label], "confidence": confidence, "ms": ms,
        "shown": mask(text, matcher, label == 1),
    }
    if compare:
        entry["all_models"] = []
        for name, pred in predictors.items():
            l, c, t = classify(pred, text)
            entry["all_models"].append({"model": name, "label": LABELS[l], "confidence": f"{c:.0%}", "ms": round(t, 1)})
    st.session_state.log.append(entry)
    st.rerun()
