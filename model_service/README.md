# Verdia live model service

This is inference from existing checkpoints. No additional model training or
benchmark rerun is performed here. The website's PrimeKG search remains an
independent context view; the HINT score does not consume those graph paths.

## Flow

1. Node retrieves the current ClinicalTrials.gov record for the entered NCT ID.
2. Only Phase II trials with resolvable drug, disease and eligibility inputs proceed.
3. Each non-placebo intervention must be recorded as a drug and resolve to one
   PubChem compound. Registry names, cleaned names and registry aliases are tried.
   PubChem supplies SMILES; RDKit reads the molecular graph. A missing ingredient
   blocks prediction rather than yielding a partial-input score.
4. Disease names use normalized matches to consistent name-to-ICD mappings
   in the TOP Phase II **training** inputs. Conservative possessive and two-part
   comma-order aliases expand the dictionary from 1,091 to 1,137 keys. No stage,
   subtype or severity qualifier is dropped. Ambiguous or absent names are rejected.
   These inherited dataset mappings are not a clinical coding service. Codes
   outside the checkpoint vocabulary are reported and omitted; an empty mapping
   is rejected instead of using HINT's arbitrary index-zero fallback.
5. The unchanged upstream eligibility line parser forms inclusion/exclusion
   groups. When it cannot split them, the entire text is used in both groups,
   matching training. BioBERT creates fresh fp32 sentence vectors. The final
   encoder layer is averaged over real tokens including CLS/SEP, with real-token
   segment IDs all one. Long lines are capped at 512 tokens and counted.
6. The sentence-attention checkpoint and matched HINT control each run once on
   the same inputs in eval/no-gradient mode. Sigmoid is applied once. Our other
   context-attention arm is shown in the benchmark table, not run live.
7. The response records scores, resolved inputs, eligibility hash, checkpoint
   hashes, encoder revision, timestamp, elapsed inference time and limitations.

The main score is a model output for TOP's trial-success label. It is not a
calibrated real-world probability or a separate efficacy/safety head. Accuracy,
ROC-AUC and AP in the UI come from the saved historical test metrics. Accuracy
uses threshold 0.5 and is calculated from the saved confusion matrix.

Live preprocessing (current registry text + PubChem + exact dictionary lookup)
has not been benchmarked end to end. Its scores can differ from the saved TOP
scores because inputs and floating-point execution differ. Known train/valid/test
membership is displayed; completed trials are demonstrations, not prospective
predictions. No labels, registry outcome measures, or completion status are
passed to the neural model. There is no claim of improvement over HINT.

## Assets and identity

`scripts/package-model.py` exports benchmark metadata, saved test scores without
labels, split membership, and a disease dictionary from the training CSV only.
It checks CSV hashes, duplicate IDs, split overlap, score ranges, and alignment
of the recorded test predictions as part of packaging. It does not rerun tests.

Assets live in gitignored `data/predictions/`. The production service only needs:

- `authors_phase_II.ckpt`: reviewed original full object, hash checked before
  loading; preserves the original GRAM code-index mapping and ordinary buffers.
- `attention_state.pt` and `control_state.pt`: strict state-dictionary restores,
  verified against hashes in the experiment metrics.
- `benchmark.json`, `disease-map.json`, `membership.json`: small exported records.

Never accept checkpoint uploads. Only the trusted original object is loaded with
`weights_only=False`; both trained state dictionaries use `weights_only=True`.
The 657 MB training sentence cache and the raw trial CSVs are not deployed.

BioBERT is pinned to `dmis-lab/biobert-base-cased-v1.1` revision
`924f12e0c3db7f156a765ad53fb6b11e7afedbc8`. Download its config, vocab and weights
with `huggingface_hub.snapshot_download` once, then run offline. The upstream
HINT code is vendored at commit `8dc0497f23fdb84e2905da7655924a91e6e79798`;
see `THIRD_PARTY.md` for attribution and noncommercial terms.

## Azure runtime

Python 3.12 / torch 2.4.1 CPU in `/home/azureuser/verdia-model-venv`.
`deploy/verdia-model.service` runs the worker at `127.0.0.1:8791`. Node on 8790
is the only caller; Caddy exposes the Node app over HTTPS. No model port is public.
Health is `/health`; internal `POST /predict` accepts only prepared input fields.
The public app offers `GET /api/prediction/:nct` for historical metadata and
`POST /api/prediction/:nct` for live inference (no arbitrary feature uploads).

One inference at a time, up to three distinct trials queued (duplicate NCT
requests share a job), two CPU threads, at most 200 nonempty eligibility lines,
600 atoms per molecule, twelve drugs and 60,000 eligibility characters. The
worker has a 2.5 GB memory limit. Node caches up to 32 completed live predictions
for 15 minutes and labels reused responses. It limits resolution and inference
timeouts. Unknown inputs and service failures return an explicit unavailable
state, never a made-up score.

Graph requests and prediction requests have separate concurrency limits so a
mapping animation does not consume the prediction worker's request allowance.

The original browser graph style is preserved. Rendering and controls are off
until a query is submitted and seeds are mapped. Clearing the input aborts the
client streams, hides the scene, clears results, and stops its animation loop.
Searches reveal mapped nodes and edges over a faint graph background. Both panels remain scrollable;
the Prediction tab includes input provenance, attention line weights (pooling
emphasis, not causal attribution), and a downloadable JSON record.

When live inputs cannot be resolved but a historical TOP test prediction exists,
the UI prominently labels and displays that saved result. It never calls it a
fresh prediction. Observational studies and unsupported phases show their study
type, a specific reason, and buttons for the two known supported Phase II demos.
