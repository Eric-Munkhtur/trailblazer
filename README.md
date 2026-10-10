# Verdia · Agentic Assembly

Give it an NCT number. An AI agent pulls the trial from ClinicalTrials.gov, extracts and normalizes the intervention(s) and disease(s), grounds them in public biomedical knowledge bases, and assembles a live, explorable 3D knowledge graph.

## Run

```bash
npm install
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY
npm run dev            # http://localhost:5173
```

`npm start` builds the frontend and serves everything from the API server on port 8787.

`LLM_PROVIDER=mock npm run dev` runs an offline demo with no LLM: extraction comes straight from the trial's structured fields, and the synthesis step is a placeholder.

## Agent pipeline

| Step | Tool | Output in the graph |
| --- | --- | --- |
| Retrieve | ClinicalTrials.gov API v2 | Trial node |
| Extract & normalize | LLM | Intervention + disease nodes (canonical names, modality, arm role, combination components) |
| Mechanisms | ChEMBL | Molecular targets and `TARGETS` edges with action type; LLM-proposed targets when ChEMBL has none (marked inferred) |
| Ontology | NLM MeSH | MeSH polyhierarchy rebuilt from tree numbers (`BROADER` edges up to category roots) |
| Disease genetics | Open Targets | Disease ↔ gene associations, therapeutic areas, and **bridges** (a drug target that is also genetically associated with the disease) |
| Synthesize | LLM | Mechanistic hypothesis, key paths through the graph, hypothesized links, gaps |

## The 3D space

In "Spatial" mode the layout itself is structured: interventions and their chemical ontology extend left, diseases and their disease ontology extend right, broader concepts sit further out and curve back in depth, molecular targets bridge the two underneath, and the trial sits on top. Turn "Spatial" off for a free force-directed layout.

Click a node to inspect it and see its connections, click a key path in the Synthesis tab to trace it, toggle node types in the legend, and use Export JSON to download the graph.
