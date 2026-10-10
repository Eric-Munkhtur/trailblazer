"""Trailblazer demo UI (Streamlit).

Sidebar: agent trace checklist filling in live.
Main: trial card -> lit-up graph -> verdict + feature contributions ->
grounded explanation. LLM narrator toggle (OFF -> identical verdict).
Demo trials pre-cached in artifacts/demo_cache/ for offline judging.
"""
import streamlit as st

st.set_page_config(page_title="Trailblazer", page_icon="🔬", layout="wide")
st.title("🔬 Trailblazer")
st.caption("Structural reasoning for clinical trial plausibility — the pre-flight check.")
st.info("Scaffold committed. The live pipeline lands tonight — see DEVLOG.md.", icon="🚧")
