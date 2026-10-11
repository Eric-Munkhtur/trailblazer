# Verdia · PrimeKG neural map

The whole of [PrimeKG](https://github.com/mims-harvard/PrimeKG) (129,375 biomedical entities, 4,050,249 relations) rendered as a 3D "brain". Give it an NCT number: an AI agent reads the trial from ClinicalTrials.gov, extracts its interventions and diseases, grounds them as neurons in PrimeKG, and fires signals from both sides through the graph until the wavefronts meet. The strongest pathways between drug and disease then lock in as looping spike trains.

## Run

```bash
npm install
npm run primekg        # one-time: downloads PrimeKG (~520 MB) from Harvard Dataverse and builds the graph (~6 s after download)
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY
npm run dev            # http://localhost:5173
```

`npm start` builds the frontend and serves everything from the API server on port 8787.

`LLM_PROVIDER=mock npm run dev` runs an offline demo without an LLM: extraction comes from the trial's structured fields, grounding takes the top lexical match, and the synthesis step is a placeholder.

You can also type a drug, disease, or gene name instead of an NCT number to find and inspect it in the graph.

## Agent pipeline

| Step | What happens | In the brain |
| --- | --- | --- |
| Retrieve | ClinicalTrials.gov API v2 | — |
| Extract & normalize | LLM: canonical drug names, combination components, modality, arm role, putative targets, diseases | — |
| Ground | Lexical search over PrimeKG names proposes candidates; the LLM picks the node that denotes the same concept. Drugs without a DrugBank node are seeded at their LLM-proposed gene targets | Candidate neurons flicker, chosen seeds light up and get labels |
| Propagate | Bidirectional search, two hops from each side, over relation-weighted edges with a hub penalty | Cyan (intervention) and pink (disease) wavefronts spread hop by hop |
| Meet & lock | Meet-in-the-middle path reconstruction, diversified across pathway kinds | Meeting neurons flash; up to 24 pathways loop as green spike trains |
| Interpret | LLM explains the mechanistic pathways and proposes hypotheses and gaps | Key pathways in the Connections panel |

Pathways are classified as known indication, off-label use, contraindication, shared target, protein-interaction bridge, pathway bridge, phenotype overlap, shared disease genetics, disease hierarchy, therapeutic analogy, functional bridge, or multi-hop.

## The 3D space

Gene/protein nodes form the core. Every other node type occupies its own cortical region on the surrounding shell (drugs, diseases, phenotypes, GO terms, pathways, anatomy, exposures). Within each region, nodes are placed by a diffusion embedding of the graph, so strongly connected entities sit near each other. The faint fibres are a stratified sample of about 90k real PrimeKG edges.

Click any neuron, or any chip in the panels, to inspect it: its description, DrugBank/MONDO features, and connections grouped by relation, with signals fired to its neighbours. Click a pathway card to isolate and trace it. Toggle node types in the legend.

## Data

PrimeKG: Chandak, Huang & Zitnik, *Building a knowledge graph to enable precision medicine*, Scientific Data 2023. Files from [Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM). The build writes the server graph (CSR adjacency, names, features) to `data/primekg/graph/` and the browser assets to `public/primekg/`; both are gitignored.
