# Verdia · PrimeKG neural map

The whole of [PrimeKG](https://github.com/mims-harvard/PrimeKG) (129,375 biomedical entities, 4,050,249 relations), laid out in 3D by its own connectivity. Give it an NCT number: an AI agent reads the trial from ClinicalTrials.gov, extracts its interventions and diseases, grounds them as nodes in PrimeKG, and fires signals from both sides through the graph until the wavefronts meet. The strongest pathways between drug and disease then lock in as looping spike trains.

## Run

```bash
npm install
npm run primekg        # one-time: downloads PrimeKG (~520 MB) from Harvard Dataverse and builds the graph and its layout (~1 min after download)
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY
npm run dev            # http://localhost:5173
```

`npm start` builds the frontend and serves everything from the API server on port 8790 (set `PORT` in `.env` to change it; the dev proxy follows it).

`LLM_PROVIDER=mock npm run dev` runs an offline demo without an LLM: extraction comes from the trial's structured fields, grounding takes the top lexical match, and the synthesis step is a placeholder.

Besides NCT numbers, the search box takes a drug, disease, or gene name (find and inspect it) or `A -> B` (connect two nodes, e.g. `metformin -> Alzheimer disease`).

## Agent pipeline

| Step | What happens | In the graph |
| --- | --- | --- |
| Retrieve | ClinicalTrials.gov API v2 | — |
| Extract & normalize | LLM: canonical drug names, combination components, modality, arm role, putative targets, diseases | — |
| Ground | Lexical search over PrimeKG names proposes candidates; the LLM picks the node that denotes the same concept. Drugs without a DrugBank node are seeded at their LLM-proposed gene targets | Candidate nodes flicker, chosen seeds light up and get labels |
| Propagate | Bidirectional search, two hops from each side, over relation-weighted edges with a hub penalty | Replayed in cost time: every node the search reaches lights up when its cumulative path cost says it should (0.9 s per unit), and the cheapest edges of each hop travel as cyan (intervention) and pink (disease) signals. The trace reports how long the real search took (tens of milliseconds) |
| Meet & lock | Meet-in-the-middle path reconstruction, diversified across pathway kinds | Meeting nodes flash when both wavefronts have reached them; up to 24 pathways loop as green spike trains |
| Interpret | LLM explains the mechanistic pathways and proposes hypotheses and gaps | Key pathways in the Connections panel |

Pathways are classified as known indication, off-label use, contraindication, shared target, protein-interaction bridge, pathway bridge, phenotype overlap, shared disease genetics, disease hierarchy, therapeutic analogy, functional bridge, or multi-hop.

## The 3D space

Node positions are computed by `npm run primekg` with ForceAtlas2 (Jacomy et al. 2014) in three dimensions, in LinLog mode, which pulls densely interconnected communities together and pushes weakly connected ones apart (Noack 2007). Repulsion between all 129k nodes uses a Barnes–Hut octree, runs in parallel worker threads (`scripts/forcelayout.mjs`), and starts from a spectral embedding; the build is seeded, so the same PrimeKG release always gives the same map. No region is assigned by hand: drugs, diseases, genes, and GO terms end up where their edges put them.

The layout uses about 1.2M edges. Tissue-expression and drug–drug edges (2.9M) connect nearly everything to everything and are left out, except that each anatomy term is tied to its three most tissue-specific genes and nodes with no other edges keep their three most specific neighbours.

Each point is sized by its real degree. The fibres are a stratified sample of 140k real edges; they fade with length and with local density so the texture of clusters shows instead of a saturated core, and grow stronger as you zoom in.

## Interacting

- **Hover** a node to see all its edges (or an even per-relation sample of 1,500 for big hubs) with its neighbours highlighted.
- **Click** a node, or any chip in the panels, to pin it: the inspector shows its description, DrugBank/MONDO features, and connections grouped by relation, and its best-connected neighbours are named in the graph.
- **Connect to…** in the inspector (or `A -> B` in the search box) runs the neural search between any two nodes, without an LLM.
- **Zoom** in and the most prominent nearby nodes (by degree within their type) are named.
- Click a pathway card to isolate and trace it; toggle node types in the legend; Esc cancels.

## Data

PrimeKG: Chandak, Huang & Zitnik, *Building a knowledge graph to enable precision medicine*, Scientific Data 2023. Files from [Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM). The build writes the server graph (CSR adjacency, names, features) to `data/primekg/graph/` and the browser assets (positions, types, degrees, names, edge sample) to `public/primekg/`; both are gitignored.
