"""Verdia — Agent 1: Ingestion & Agentic Fetching (UI).

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

st.set_page_config(page_title="Verdia", page_icon="🔬", layout="wide")

st.title("🔬 Verdia")
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

def graph_section(rep):
    """Agent 2: full-graph star-field + live search animation between pins."""
    import random

    import pandas as pd
    import streamlit.components.v1 as components
    from graph_view import GraphView
    from viz import TYPES, build_html

    ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")

    @st.cache_resource(show_spinner="Loading the biology map (once, ~30s)…")
    def get_graph() -> GraphView:
        return GraphView()

    @st.cache_resource(show_spinner="Building the galaxy layout…")
    def get_galaxy():
        df = pd.read_csv(os.path.join(ART, "positions.csv.gz"))
        tidx = {t: i for i, t in enumerate(TYPES)}
        dust = {"xs": df["x"].tolist(), "ys": df["y"].tolist(), "zs": df["z"].tolist(),
                "ts": [tidx.get(t, len(TYPES) - 1) for t in df["type"]]}
        pos = dict(zip(df["name"], zip(df["x"], df["y"], df["z"])))
        return dust, pos

    if rep.drug and rep.drug.ok and rep.disease and rep.disease.ok:
        sg = get_graph().subgraph_for(rep.drug.node_name, rep.disease.node_name)
        if sg.found_routes:
            st.markdown(f"**Traced route(s)** through the knowledge graph "
                        f"({sg.found_routes} found):")
            for r in sg.routes[:5]:
                st.markdown("&nbsp;&nbsp;→ ".join(
                    f"`{x}`" if not x.startswith("[") else f"**{x}**"
                    for x in r["chain"]))
        else:
            st.markdown("**No connecting route exists in PrimeKG** — the drug and "
                        "disease sit in disconnected parts of the map. This trial is "
                        "mechanistically *uncharted*: either novel biology or a long "
                        "shot. The agent reports this honestly instead of inventing a "
                        "plausible-sounding chain.")
    elif (rep.drug and rep.drug.ok) or (rep.disease and rep.disease.ok):
        pin = rep.drug if (rep.drug and rep.drug.ok) else rep.disease
        sg = get_graph().subgraph_for(pin.node_name, pin.node_name)  # neighborhood only
    else:
        return

    if sg.nodes:
        dust, pos = get_galaxy()
        rng = random.Random(42)
        for n in sg.nodes:                     # seat action nodes at their galaxy spot
            x, y, z = pos.get(n["name"], (rng.randint(-800, 800), 0, rng.randint(-800, 800)))
            n["x"], n["y"], n["z"] = int(x), int(y), int(z)
        html = build_html(sg.nodes, sg.links, dust=dust, height=720)
        components.html(html, height=740, scrolling=False)




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

        # ---- Agent 2: 3D map view ---------------------------------------
        st.subheader("🧬 Agent 2 — the map (PrimeKG)")
        if rep.status in ("MAPPED", "PARTIAL"):
            try:
                graph_section(rep)
            except Exception as e:
                st.warning(f"Map view unavailable: {e}")
        else:
            st.info("Nothing pinned — nothing to draw (honest refusal).")
    else:
        st.info("No cached trial and no network — try a demo trial from the sidebar.")

st.divider()
st.caption("Research prototype · decision support only · public data only "
           "(ClinicalTrials.gov API v2, PrimeKG) · not a medical device.")
