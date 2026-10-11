# Final experiment: drug/disease-conditioned eligibility attention

## Decision and scope

This is one final exploratory architecture attempt, authorized after the first
attention experiment's test results were inspected. It is not an untouched-test
confirmatory experiment, a formal preregistration, or a claim of novel attention
machinery. The accompanying lock file records UTC time and hashes before launch.
There will be one seed, one fixed configuration, no ensemble, no parameter sweep,
and no architecture changes based on this attempt's test result.

## Prior evidence and hypothesis

The original continuation control achieved test ROC-AUC 0.6320184475, AP
0.6829250175 and F1@0.5 0.6058437686. Global sentence attention achieved
0.6323085394, 0.6833485441 and 0.6055919096 respectively: essentially tied.
These results motivated this second architecture and must remain in the report.

Diagnostic inspection showed nonzero learned global scorers and nonuniform
weights on 64 validation trials. Validation prediction correlation with the
control was 0.9988057583, mean absolute probability difference 0.0071686096.
This does not prove why performance was similar, nor guarantee a better variant.

Hypothesis: drug and disease representations can identify trial-specific
eligibility information before pooling discards it. HINT already combines the
modalities downstream; the proposed change adds conditional selection upstream.

## Fixed architecture

- Drug MPNN and disease GRAM remain the original encoders, each outputting 50 dims.
- Concatenate their outputs, apply LayerNorm(100) without trainable affine terms.
- Project each frozen 768-d sentence vector into a shared 32-d key space.
- Separate linear inclusion/exclusion queries map the 100-d context to 32 dims.
- Scores are key dot query divided by sqrt(32), softmax within each unpadded group.
- Pool original 768-d sentence values; keep the original 1536-to-50 FC/ReLU.
- Query weights and biases start at zero, giving uniform pooling initially.
- The key projection uses PyTorch's default Linear initialization with seed 0.
- Added parameters: 768*32 + 2*(100*32+32) = 31,040.
- All outcome-model parameters train; cached BioBERT embeddings remain frozen.
- Missing data, parsing, graph edges, graph attention, and downstream head unchanged.
- PrimeKG is not an input to this model. Sentence weights are not causal evidence.

## Fixed training and comparison

Start from the authors' original Phase II checkpoint, not either continued model.
Reuse the exact MSI assets and comparison control from continuation_s0/original_s0.
Train 5 epochs, seed 0, batch 32, Adam lr 0.0003, weight decay 0, BCEWithLogitsLoss,
2 CPU threads, CUDA on the MSI GTX 1050 Ti. Same software and loader generators.
Reseed after imports and model installation. No extra pretraining or warmup.

Select lowest sample-averaged validation BCE across epochs 0 through 5.
The epoch-0 validation loss must agree with the original control within 1e-6;
otherwise abort before optimization and document an implementation mismatch.
Validate all control asset hashes and training settings before optimization.
Save best and last tensor checkpoints and epoch logs. In addition to the exact
base checkpoint, export disease-code mapping for independent reconstruction.

Evaluate the selected checkpoint on the test split once at the end. Write a marker
before starting test prediction so an interrupted evaluation cannot silently be
repeated in the same run. No test-based threshold selection; threshold remains 0.5.
Only completed prediction artifacts support a reported test result.

## Reporting

Primary metric: continuous ROC-AUC; secondary: continuous AP and F1 at 0.5.
Report all three experiments, absolute differences, and the single-seed limitation.
For a descriptive target fixed before launch, a gain of at least 0.01 absolute
ROC-AUC over the matched control, with no AP decrease, counts as noticeable here.
This target is not a significance test. Smaller gains are reported numerically
without calling them meaningful; a negative outcome is retained and reported.

Preserve the configuration, source snapshot, source/data/checkpoint hashes,
environment freeze, logs, per-trial validation/test predictions, and final metrics.
Do not publish protected assets or credentials. Upstream HINT/TOP restrictions
continue to apply; this package does not grant an MIT license over upstream work.

## Time and failure policy

Expected training time is roughly 30-45 minutes based on the preceding 29-minute
arms; it is not guaranteed. Numerical/runtime failures stop the run. Any repair
must be documented; no silent retry with a different model or hyperparameters.
This experiment does not suspend the website integration and deployment objective.
