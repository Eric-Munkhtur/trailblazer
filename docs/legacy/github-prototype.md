# Verdia 🔬

**Structural Reasoning AI for Clinical Trial Outcome Prediction — the pre-flight check for clinical trials.**

> Roughly 9 in 10 drugs that enter clinical trials never reach approval, and a failed pivotal
> trial burns a median **$19M+** (≈$41K per patient). Verdia is an early-warning system:
> given only a trial's ID — **before any public results exist** — it traces the drug's route
> through the real map of human biology and reports whether the trial is mechanistically
> plausible, with every hop of the evidence visible.

Built for the Regeneron challenge at HackRU Fall 2026. Regeneron's brief:

> *"Build a Structural Reasoning AI system capable of predicting the outcome (efficacy or
> safety) of a clinical trial prior to its public results... reason natively over structural
> representations... This provides the transparent, mechanistic rationale required by human
> experts and regulatory audiences."*

## What it does

You type an NCT number (a ClinicalTrials.gov trial ID). An agent then:

1. **Fetches** the trial's intervention (drug), condition (disease), phase, and primary endpoint — *protocol fields only; the code never reads the results section.*
2. **Resolves** the drug and disease onto [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/) — Harvard's multimodal knowledge graph of drugs, proteins, genes, pathways, and diseases (17K+ diseases, ~4M relationships).
3. **Traces** the structural routes between them: shortest paths through drug → target → pathway → disease, edge semantics, contradiction checks (contraindication edges → safety flags).
4. **Scores** plausibility with a transparent formula + trained structural embeddings — not a black box, weights shown on screen.
5. **Explains** in plain English, grounded *only* in the retrieved graph facts. The LLM narrates; it never decides. Disable the narrator and the verdict is identical.

If the entities can't be mapped, the system says **UNASSESSABLE** — an honest "I don't know" is a valid output.

## Architecture

```
┌────────────── OFFLINE (prep/, Google Colab) ──────────────┐
│ PrimeKG → filtered compact graph.pkl + name→node indexes  │
│ PyKEEN structural embeddings → embeddings + MRR/Hits@10   │
│ TOP/HINT benchmark → outcome model + ablation table       │
└────────────────────────────────────────────────────────────┘
        │ small committed artifacts
        ▼
┌────────────── ONLINE (app/, local demo) ──────────────────┐
│ Streamlit UI ← agent.py orchestrating:                    │
│   ctgov.py → resolver.py → graph_engine.py → scorer.py    │
│   → narrator.py (LLM, facts-only) → viz.py (lit-up path)  │
└────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The 5 demo trials are cached locally (`artifacts/demo_cache/`), so the full demo
runs with **zero internet** at the judging table.

## Evaluation (the claim)

On the public **TOP/HINT benchmark** (~1.9K Phase III trials with success/fail labels),
under a **time-based split** (train on older trials, test on newer — no future leakage):

| System | PR-AUC |
|---|---|
| Majority class | baseline |
| Zero-shot LLM (text-only) | see `results/` |
| Symbolic path features only | see `results/` |
| **Symbolic + structural embeddings (Verdia)** | see `results/` |

We also report results **with indication edges excluded**, so the graph can't leak
already-approved status into the prediction. Link-prediction metrics (MRR, Hits@10)
for the trained embeddings are in `results/embedding_metrics.json`.

## Repo layout

```
app/        # the live system (API client, resolver, graph engine, scorer, UI)
prep/       # Colab notebooks: graph build, embedding training, benchmark
artifacts/  # compact graph, indexes, embeddings, demo cache
results/    # metrics, ablation tables
```

## Scope & honest limitations

- v1 implements the **"geometric pathways of a biological knowledge graph"** half of the
  brief. 3D spatial docking ("spatial constraints of drug-target interaction") is future work.
- v1 predicts **efficacy**; safety appears as contraindication/off-pathway flags, not as a
  trained safety-outcome model.
- Decision **support**, never a decision maker. Not a medical device.

## License

[MIT](LICENSE) © 2026 Eric Munkhtur
