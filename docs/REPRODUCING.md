# Reproducing and serving Verdia

This guide separates the deployed demo, the recorded experiments, and a new reproduction. No training or automated test suite is run by installing the website. The public package has not been reproduced from scratch as a release.

## 1. Run the website

Follow the root README for Node 22, `npm ci`, `npm run primekg`, and `LLM_PROVIDER=mock npm run dev`. This gives graph exploration; prediction remains unavailable until the model assets and worker are provisioned. The mock provider still uses live ClinicalTrials.gov data and requires internet.

The graph builder downloads descriptive feature files for local use. Before a public deployment, omit those descriptions: the production `data/primekg/graph/features.json` is `{}`. Keep original local data separate. See the deployment guide.

## 2. Obtain the original inputs

Read [HINT's Benchmark Usage Agreement](https://github.com/futianfan/clinical-trial-outcome-prediction#benchmark-usage-agreement) first. Code is pinned to:

```
8dc0497f23fdb84e2905da7655924a91e6e79798
```

A sparse upstream checkout avoids unrelated repository directories, but selected data/model files can still be large:

```bash
git clone --filter=blob:none --no-checkout https://github.com/futianfan/clinical-trial-outcome-prediction.git upstream/hint
git -C upstream/hint sparse-checkout init --cone
git -C upstream/hint sparse-checkout set HINT data save_model icdcode results
git -C upstream/hint checkout 8dc0497f23fdb84e2905da7655924a91e6e79798
```

The matched-continuation runner expects the following in an `assets/` directory, separate from the source:

```
phase_II.ckpt                    # upstream save_model/phase_II.ckpt
phase_II_train.csv               # upstream data/
phase_II_valid.csv
phase_II_test.csv
icdcode2ancestor_dict.pkl        # original hierarchy, retain exact identity
sentence2embedding.pkl           # regenerated BioBERT cache, not upstream empty stub
```

The original hierarchy can also be exported from the reviewed authors' checkpoint's `disease_encoder.icdcode2ancestor`. Preserve the checkpoint's exact GRAM code-index mapping when restoring weights; rebuilding it from unordered sets is not equivalent. Loading the full original checkpoint or a pickle cache executes Python deserialization, so use only the reviewed upstream/local artifacts and check their recorded hashes.

Recorded MSI checkpoint SHA-256:

```
da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63
```

Recorded MSI sentence-cache SHA-256:

```
d4c8f2301ef0eb3213eac9de4c8a13c32a7e5d43efa8da292ab9d230d9e54a90
```

All split/hierarchy identities are in `research/results/control/config.json`. The cache contains 201,374 Phase II sentences. The Colab final runs used a different regenerated cache; consult `research/provenance/` and the paper rather than substituting it silently.

### Regenerating embeddings

`research/baseline/scripts/gen_embeddings.py` archives the full-cache generator; `subset_cache_phase2.py` extracts the Phase II subset without recomputing vectors. `gen_subset_embeddings.py` is the later direct-subset generator. These are Colab `/content` scripts, require their recorded dependencies and upstream checkout, and are not portable one-command installers. Read the scripts before invoking them and record path changes.

Semantics: BioBERT v1.1, fp32, evaluation mode, final-layer token mean including CLS/SEP, token type IDs 1 for real tokens, 512-token cap. Embedding values and pickle bytes need not be identical across regenerations. A newly generated cache must get a new hash and run record. The included numerical-equivalence evidence does not prove arbitrary future regenerations equivalent.

## 3. Repeat the matched continuation experiment

The original MSI environment is recorded in `research/results/control/environment.txt`: Python 3.11, torch 2.4.1 + CUDA 12.1, GTX 1050 Ti. A compatible setup is:

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r model_service/requirements.txt
```

Run from the repository root. Use unique output directories; existing outputs are intentionally rejected. These commands assume the exact recorded cache is available. If using regenerated inputs, substitute and record their actual hashes and identify the result as a new reproduction.

```bash
PYTHONHASHSEED=0 .venv/bin/python research/attention/train_pair_arm.py \
  --repo upstream/hint --assets assets --output runs/continuation_s0/original_s0 \
  --arm original --seed 0 --epochs 5 --batch-size 32 --learning-rate 0.0003 \
  --device cuda --threads 2 \
  --checkpoint-sha256 da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63 \
  --embeddings-sha256 d4c8f2301ef0eb3213eac9de4c8a13c32a7e5d43efa8da292ab9d230d9e54a90 \
  --evaluate-test

PYTHONHASHSEED=0 .venv/bin/python research/attention/train_pair_arm.py \
  --repo upstream/hint --assets assets --output runs/continuation_s0/attention_s0 \
  --arm attention --seed 0 --epochs 5 --batch-size 32 --learning-rate 0.0003 \
  --device cuda --threads 2 \
  --checkpoint-sha256 da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63 \
  --embeddings-sha256 d4c8f2301ef0eb3213eac9de4c8a13c32a7e5d43efa8da292ab9d230d9e54a90 \
  --evaluate-test

PYTHONHASHSEED=0 .venv/bin/python research/conditioned/train_conditioned.py \
  --repo upstream/hint --assets assets --output runs/conditioned_s0 \
  --control-run runs/continuation_s0/original_s0 \
  --arm conditioned --seed 0 --epochs 5 --batch-size 32 --learning-rate 0.0003 \
  --device cuda --threads 2 \
  --checkpoint-sha256 da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63 \
  --embeddings-sha256 d4c8f2301ef0eb3213eac9de4c8a13c32a7e5d43efa8da292ab9d230d9e54a90 \
  --evaluate-test
```

The conditioned runner enforces its source lock and CUDA settings. Run sequentially. The original experiment was one seed and the second design followed earlier test inspection; repeating the commands does not undo that exploratory history.

The source archives include scripts for recalculating tables from private saved prediction CSVs. Those CSVs include labels and are not part of this public release. Do not select hyperparameters or thresholds using the test set.

## 4. Fresh Colab baseline archive

`research/baseline/scripts/train_hint.py` expects `/content/clinical-trial-outcome-prediction` and writes `/content/hint_evidence/training/hint_p2_sN`. It builds a fresh HINT outcome model with the authors' ADMET initialization and accepts the seed as a positional argument. Completed recorded runs used CPU; the archived final script auto-selects CUDA if available. To reproduce the CPU setup on a GPU VM, explicitly disable CUDA for that process and record this choice.

The final script seeds after imports and saves the GRAM mapping. The earlier seed-0 run used an older path; an exact source snapshot for that earlier path is not asserted. Config files retain stale CUDA prose, so the actual `device` field and runtime accounting govern interpretation. Fresh baseline runs and matched continuations use different initialization and validation-loss aggregation. Compare them in separate tables.

## 5. Serve your resulting checkpoints

On a CPU host, use a separate Python 3.11/3.12 environment and install torch 2.4.1 from the CPU wheel index plus `model_service/requirements.txt`.

After obtaining the three run folders (with metrics, configs, and prediction CSVs) and the exact original assets:

```bash
mkdir -p data/predictions
cp assets/phase_II_train.csv assets/phase_II_valid.csv assets/phase_II_test.csv data/predictions/
python scripts/package-model.py --results-root runs --assets data/predictions
cp assets/phase_II.ckpt data/predictions/authors_phase_II.ckpt
cp runs/continuation_s0/original_s0/best_state.pt data/predictions/control_state.pt
cp runs/continuation_s0/attention_s0/best_state.pt data/predictions/attention_state.pt
```

The exporter checks alignment and hashes, and records metrics for **your supplied runs**. It does not claim those runs have the published Verdia scores. Only the sentence-attention and original models run live; the conditioned model appears in the comparison table.

Download the pinned BioBERT snapshot into the serving environment's Hugging Face cache once:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'dmis-lab/biobert-base-cased-v1.1',
    revision='924f12e0c3db7f156a765ad53fb6b11e7afedbc8',
    allow_patterns=['config.json', 'vocab.txt', 'pytorch_model.bin',
                    'tokenizer_config.json', 'special_tokens_map.json'],
)
PY
python model_service/serve.py
```

The worker binds to `127.0.0.1:8791` and Node calls it locally. It needs the original checkpoint, the two trained states, `benchmark.json`, `disease-map.json`, and `membership.json`. The raw training CSVs and cached sentence vectors are not needed on the deployment host. Keep all model assets outside public static directories. Start the Node app separately as described in the root README.

`deploy/` contains the actual Azure configuration as an example. Adapt its user, paths, hostname, and service limits for another machine. Do not expose the model worker's internal port.

## 6. Review the report

`paper/verdia.tex` contains the standalone report and embedded annotated figure. No PDF was exported as part of this repository publication. The report and root README are the current interpretation; files in `docs/legacy/` and archived scripts retain historical wording.
