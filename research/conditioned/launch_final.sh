#!/usr/bin/env bash
# Exactly one fixed architecture run; no automatic retry or sweep.
set -euo pipefail
cd /home/edo/hint-attention
final_run=/home/edo/hint-attention/runs/conditioned_s0
final_status="${final_run}.status"
if [[ -e "$final_run" || -e "$final_status" ]]; then
  echo 'This final attempt already exists. Inspect it; do not silently rerun.' >&2
  exit 1
fi
(set -o noclobber; printf 'starting final conditioned attempt\n' > "$final_status")
trap 'printf "failed: inspect conditioned_s0 console/driver logs\n" > "$final_status"' ERR
printf 'running final conditioned attempt\n' > "$final_status"
.venv/bin/python -u final_conditioned/train_conditioned.py \
  --repo upstream --assets assets --output "$final_run" \
  --control-run runs/continuation_s0/original_s0 \
  --arm conditioned --seed 0 --epochs 5 --batch-size 32 \
  --learning-rate 0.0003 --device cuda --threads 2 \
  --checkpoint-sha256 da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63 \
  --embeddings-sha256 d4c8f2301ef0eb3213eac9de4c8a13c32a7e5d43efa8da292ab9d230d9e54a90 \
  --evaluate-test > "${final_run}.console.log" 2>&1
.venv/bin/python final_conditioned/rebuild_pair_results.py --runs-root runs
printf 'completed final conditioned attempt and comparison\n' > "$final_status"
