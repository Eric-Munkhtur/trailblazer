# Experiments and evidence

This directory contains code and recorded outputs from the completed HackRU work. It is not a claim that either modification beats HINT.

## Three distinct comparisons

1. **Authors' supplied predictions vs our evaluation of their supplied checkpoint.** They differ; the released checkpoint is not asserted to be the exact model that generated every published score.
2. **Fresh outcome-model training in Colab.** Three completed CPU runs, initialized with the authors' ADMET molecular pretraining. Seed 0 used an earlier seeding path; seeds 1 and 2 used the later fix. Do not call this a uniformly seeded replication of the paper's 30-run result.
3. **Matched checkpoint continuations on the MSI.** Original, sentence attention, and conditioned attention use the same starting checkpoint, fixed sentence vectors, data splits, and five-epoch recipe. These are the comparisons relevant to our changes.

| Matched arm | ROC-AUC | AP | F1 at 0.5 |
|---|---:|---:|---:|
| Control | 0.632018 | 0.682925 | 0.605844 |
| Sentence attention | 0.632309 | 0.683349 | 0.605592 |
| Conditioned attention | 0.630982 | 0.682502 | 0.596026 |

All selected epoch 4 (epochs 0–5 considered). The sentence-attention AUC difference is +0.000290; its descriptive paired bootstrap interval includes zero. Read [the diagnosis](DIAGNOSIS.md) for the measured effects and limits.

Fresh-training mean ROC-AUC was 0.636138 (population SD 0.004160). This is a separate reference experiment, not evidence that our attention beats a fresh-trained baseline. Full values are in [results](results/) and [aggregate summary](provenance/results_summary.json).

## Source layout and provenance

- `attention/`: actual matched-continuation runner, extension, and results-table script. Runner SHA-256 matches the recorded configuration. Early comments saying it was not yet run are preserved as part of the original byte-identical source.
- `conditioned/`: exact locked source snapshot. Do not edit these files in place and expect the recorded lock to pass. Its launch script contains historical MSI paths; use the explicit commands in the reproduction guide.
- `baseline/scripts/`: GLM's final archived Colab scripts. They use `/content` paths. The final training script is not proof of the exact earlier seed-0 source: that source version was not separately archived here. Its CUDA deviation prose is stale; the recorded completed runs used CPU. A code bug does not establish how an authors' historical checkpoint was saved. The generator's subset metadata description also retains stale full-cache wording. These are documented archive limitations.
- `analysis/`: original post-hoc diagnosis scripts with historical MSI paths. Adapt paths for a new environment; record any changes. They require private run outputs, checkpoints, and the licensed source data.
- `results/`: recorded metrics, configurations, epoch histories, and MSI environment snapshots.
- `provenance/`: data/cache identities, numerical-equivalence evidence, environment records, source hashes, and runtime accounting.

## Hardware

- **Colab T4:** BioBERT cache generation and interrupted training attempts. Interrupted runs are excluded from final results.
- **Colab CPU VMs:** three completed fresh HINT outcome-model training runs.
- **MSI GTX 1050 Ti, 4 GB:** matched continuation experiments; PyTorch 2.4.1 + CUDA 12.1, Python 3.11.
- **Azure 2 vCPU / 4 GB:** website and live CPU inference; no training.

The original runtime accounting report is archived with an `.original.md` suffix. Its infrastructure timings and compute-unit usage are estimates; it does not replace the recorded metrics or the paper's corrected interpretation. Colab and MSI used different cache files; their recorded hashes must not be silently interchanged.

## Access and repeatability

[Reproduction instructions](../docs/REPRODUCING.md) describe prerequisites and the commands. No new experiment was launched when creating this repository. We publish aggregate evidence and source, while leaving original TOP rows, full per-trial predictions, sentence caches, and trained weights out of this release. Consequently exact reevaluation of these saved weights requires the separately retained artifacts; public hashes establish identity but do not establish scientific validity or substitute for those files.

The conditioned design followed earlier test inspection, so it is exploratory. Neither modification was repeated over multiple seeds. The full release has not been independently reproduced from a clean clone.
