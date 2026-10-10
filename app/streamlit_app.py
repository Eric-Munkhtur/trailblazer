"""Trailblazer — Agent 1: Ingestion & Agentic Fetching (UI).

Paste an NCT number -> the agent queries ClinicalTrials.gov API v2, extracts
intervention (drug), indication (disease), phase, and eligibility protocol,
then maps entities onto PrimeKG (LOOK UP / FIND / PIN trace).

Offline-safe: trials are cached in artifacts/demo_cache/.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from agent import AssemblyAgent
from ctgov import CACHE_DIR

st.set_page_config(page_title="Trailblazer", page_icon="🔬", layout="wide")

st.title("🔬 Trailblazer")
st.caption("Agentic Assembly — NCT number → entities extracted → mapped onto the "
           "PrimeKG knowledge graph. Protocol fields only; never results.")

# --- agent (cached across reruns) ---------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent() -> AssemblyAgent:
    return AssemblyAgent()


# --- demo trials quick-pick ---------------------------------------------
st.sidebar.header("Demo trials")
DEMO = {
    "NCT02251548": "Ibrutinib · chronic leukemia (clean map)",
    "NCT07091175": "Dupilumab · nephrotic syndrome (clean map)",
    "NCT04437511": "Donanemab · Alzheimer's (honest partial)",
}
picked = st.sidebar.radio("Quick pick", list(DEMO.keys()),
                          format_func=lambda k: f"{k} — {DEMO[k]}",
                          label_visibility="collapsed")
st.sidebar.caption("Cached locally — the demo runs offline.")

# --- input ---------------------------------------------------------------
col1, col2 = st.columns([3, 1])
nct = col1.text_input("NCT trial number", value=picked, label_visibility="collapsed")
run = col2.button("🤖 Run agent", use_container_width=True)

if run or nct:
    agent = get_agent()
    with st.spinner("Agent working…"):
        try:
            rep = agent.analyze(nct)
        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

    # ---- trace ----------------------------------------------------------
    st.subheader("Agent trace")
    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}
    for s in rep.steps:
        st.markdown(f"{icon[s.status]} **{s.n}. {s.name}** — {s.detail}")

    status_colors = {"MAPPED": "green", "PARTIAL": "orange", "UNASSESSABLE": "red"}
    st.markdown(f"### Status: :{status_colors.get(rep.status, 'gray')}[{rep.status}]")

    # ---- trial card -----------------------------------------------------
    if rep.trial:
        t = rep.trial
        st.subheader("Ingested trial metadata")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Phase", ", ".join(t["phases"]) or "—")
        c2.metric("Sponsor", t["sponsor"][:22] or "—")
        c3.metric("Status", t["status"].replace("_", " ").lower() or "—")
        c4.metric("Interventions", len(t["interventions"]))

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**🧪 Intervention (drug candidate)**")
            for i in t["interventions"]:
                st.markdown(f"- {i}")
            if rep.drug:
                if rep.drug.ok:
                    st.success(f"Pinned on graph: **{rep.drug.node_name}** "
                               f"(`{rep.drug.node_id}`, {rep.drug.method})")
                else:
                    st.error("Not found in PrimeKG — honest refusal "
                             "(below-threshold candidates logged)")
        with d2:
            st.markdown("**🏥 Indication (disease)**")
            for c in t["conditions"]:
                st.markdown(f"- {c}")
            if rep.disease:
                if rep.disease.ok:
                    st.success(f"Pinned on graph: **{rep.disease.node_name}** "
                               f"(`{rep.disease.node_id}`, {rep.disease.method})")
                else:
                    st.error("Not found in PrimeKG — honest refusal")

        with st.expander(f"📋 Eligibility protocol ({len(t['eligibility_criteria'])} chars)"):
            st.text(t["eligibility_criteria"] or "(none published)")

        with st.expander("🎯 Primary outcome(s) — shown, never used for mapping"):
            for o in t["primary_outcomes"]:
                st.markdown(f"- {o}")
    else:
        st.info("No cached trial and no network — try a demo trial from the sidebar.")

st.divider()
st.caption("Research prototype · decision support only · public data only "
           "(ClinicalTrials.gov API v2, PrimeKG) · not a medical device.")
