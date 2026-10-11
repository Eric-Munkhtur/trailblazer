# Attribution and license scope

The root MIT license is the existing license for **original Verdia code** by Eric Munkhtur. It does not relicense third-party code, data, pretrained or trained models, figures, or other authors' material.

## HINT / TOP

Tianfan Fu, Kexin Huang, Cao Xiao, Lucas M. Glass, and Jimeng Sun. *HINT: Hierarchical interaction network for clinical-trial-outcome predictions*. Patterns (2022). https://doi.org/10.1016/j.patter.2022.100445

Source: https://github.com/futianfan/clinical-trial-outcome-prediction
Pinned source: `8dc0497f23fdb84e2905da7655924a91e6e79798`.

`model_service/vendor/HINT/` is upstream HINT code; it is not Verdia-authored MIT code. The original architecture figure embedded in `paper/verdia.tex` is also from that repository and remains attributed. Verdia's annotations and eligibility extensions are described separately.

The upstream Benchmark Usage Agreement applies to its dataset and code and limits use to noncommercial academic research or nonprofit educational use under its stated conditions. Read the full original agreement before use:
https://github.com/futianfan/clinical-trial-outcome-prediction/tree/8dc0497f23fdb84e2905da7655924a91e6e79798#benchmark-usage-agreement

This release does not newly distribute TOP trial rows, original model checkpoints, trained checkpoint files, or regenerated sentence caches. Fetch licensed inputs from the upstream provider and follow their terms. Model-serving assets stay in gitignored `data/predictions/`.

## PrimeKG

Payal Chandak, Kexin Huang, and Marinka Zitnik. *Building a knowledge graph to enable precision medicine*. Scientific Data (2023). https://doi.org/10.1038/s41597-023-01960-3

Source and dataset: https://github.com/mims-harvard/PrimeKG and https://doi.org/10.7910/DVN/IXA7BM

Source datasets carry their own terms. A graph build does not grant new rights to descriptive text. The public Azure deployment omits bundled descriptive feature text. The older `artifacts/` directory already existed in the GitHub repository and is retained as legacy project material; it is not an MIT claim over the originating graph data.

## BioBERT and runtime libraries

BioBERT: https://github.com/dmis-lab/biobert; inference snapshot `dmis-lab/biobert-base-cased-v1.1`, revision `924f12e0c3db7f156a765ad53fb6b11e7afedbc8`. No weights are included in this release. Follow the provider's license and attribution.

Three.js, Express, Vite, PyTorch, RDKit, Transformers, NumPy, scikit-learn and other dependencies retain their respective licenses. Registry and compound information comes from ClinicalTrials.gov and PubChem. The legacy bundled JavaScript libraries retain their embedded notices.

The original interface design direction credited Zajno's Smart Earth Protection concept: https://dribbble.com/shots/25860515-Educational-Website-on-Space-Pollution . The current application uses its own Three.js scene and implementation.
