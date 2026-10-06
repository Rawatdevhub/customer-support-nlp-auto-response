from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.query_extraction import extract_core_query
from src.baseline_model import MODEL_PATH, predict, train

st.set_page_config(page_title="Customer Support Auto-Response", page_icon="💬", layout="wide")
st.title("💬 Customer Support Query Understanding")
st.caption("Stage 1: extract the core query. Stage 2: classify the intent and retrieve a suggested response.")

if not MODEL_PATH.exists():
    st.warning("The baseline model has not been trained yet. Run the training command from the README.")
    st.stop()

raw = st.text_area("Paste a full support ticket", height=220, placeholder="Hello support,\n\nI was charged twice for my order...\n\nRegards, Customer")
if st.button("Analyze ticket", type="primary") and raw.strip():
    core = extract_core_query(raw)
    result = predict(core)
    left, middle, right = st.columns(3)
    with left:
        st.subheader("1. Raw ticket")
        st.write(raw)
    with middle:
        st.subheader("2. Extracted query")
        st.write(core)
    with right:
        st.subheader("3. Intent and response")
        st.metric("Predicted intent", result["intent"])
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.write("**Suggested response**")
        st.info(result["response"])
        st.write("**Top alternatives**")
        for label, score in result["top_intents"]:
            st.write(f"{label}: {score:.1%}")
