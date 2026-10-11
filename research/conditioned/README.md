# Final HINT extension attempt

Read `EXPERIMENT_PLAN.md` for the frozen hypothesis, recipe, limitations, and
reporting rule. `experiment-lock.json` records the pre-launch time and source
hashes. This package is separate from the first experiment; its training code
and model results are retained unchanged.

## Files

- `hint_conditioned/protocol.py`: the pooling module and HINT subclass junction.
- `train_conditioned.py`: fixed continuation runner, control checks, audit output.
- `launch_final.sh`: one sequential run and comparison, with no automatic retry.
- `rebuild_pair_results.py`: metrics from predictions only; no checkpoint selection.
- `THIRD_PARTY.md`: upstream provenance and restrictions.

## Why these choices

1. Condition pooling on existing drug/disease embeddings because eligibility
   relevance may depend on treatment and indication. No new data source is needed.
2. Use a 32-d attention space to limit the extra parameters (31,040) and compute.
   It is a fixed engineering choice, not a validation-tuned optimum.
3. Share sentence keys to reduce parameters; use separate inclusion/exclusion
   queries because the groups play different roles.
4. Normalize context without learned affine parameters to control input scale.
   Use scaled dot products to keep scores numerically manageable.
5. Keep raw sentence vectors as values and retain the original FC/ReLU so the
   downstream network starts from the same representation.
6. Initialize queries at zero to preserve uniform mean pooling. The key matrix
   starts nonzero; query gradients can be learned immediately, with key/context
   gradients through attention becoming active once queries move away from zero.
7. Retain full-model training at the previous rate and budget to use the completed
   original continuation as a matched control. Do not add a higher learning rate,
   freezing strategy, or longer budget at the same time.
8. Preserve the exact base object's GRAM mapping and inherited graph forward to
   avoid rebuilding disease indices or changing unrelated architecture.
9. Select on validation BCE including epoch 0, then report all test metrics even
   when the result disappoints. Prior test inspection makes this exploratory.

## MSI invocation

Install this package as `/home/edo/hint-attention/final_conditioned/` and run
`bash final_conditioned/launch_final.sh` from `/home/edo/hint-attention`.
The existing venv, pinned upstream checkout, and read-only assets are reused.
The runner verifies source hashes, input hashes, baseline settings, parameter
count, and the epoch-0 validation-loss agreement as part of training startup.

## Outputs

`runs/conditioned_s0/` contains config, environment freeze, source snapshot,
disease mapping, initialization check, epoch history, best/last tensor states,
selected-checkpoint validation/test predictions, metric JSON, and a one-time
test-evaluation marker. Console and status files sit alongside the run directory.
The comparison includes the original and global-attention experiments too.

An output directory is never overwritten. Checkpoints contain model weights,
not optimizer/RNG resume state; interrupted optimization is not automatically
resumed or restarted. Record any failure before deciding on recovery.

## Loading for inference

1. Load the trusted original full-model checkpoint on CPU (verify its hash).
2. Call `install_conditioned(model, seed=0)`.
3. Load `best_state.pt` using `weights_only=True`, then strictly load `state_dict`.
4. Set the model and ordinary device/adjacency fields to the inference device,
   preserving GRAM lookup behavior, and call `model.eval()`.
5. Use the exact upstream input parsing and cached embedding generation contract.

For sentence weights, encode drugs and diseases once, concatenate the outputs,
and call `model.protocol_encoder.forward_with_attention(criteria, context)`.
Weights correspond to the retained sentence-cache rows, with inclusion/exclusion
fallbacks unchanged. They express aggregation weights, not signed contributions,
physical mechanisms, causal explanations, or clinical recommendations.

No standalone test suite was added or run for this package. Startup invariants
and validation are part of the authorized training experiment.
