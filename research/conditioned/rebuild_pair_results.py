"""Recalculate comparison metrics from saved per-trial test predictions.

Example: python rebuild_pair_results.py --runs-root runs/continuation_s0
Does not load a model, train, or select checkpoints. Run after the experiments.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    results, missing, comparison_reference, reference_labels = [], [], None, None
    for config_path in sorted(root.rglob("config.json")):
        config = json.loads(config_path.read_text())
        predictions = config_path.parent / "predictions_test.csv"
        completed_metrics = config_path.parent / "metrics.json"
        if not predictions.is_file() or not completed_metrics.is_file():
            missing.append(str(config_path.parent.relative_to(root)))
            continue
        with predictions.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("Empty predictions: " + str(predictions))
        nctids = [r["nctid"] for r in rows]
        if len(set(nctids)) != len(nctids):
            raise ValueError("Duplicate trial predictions: " + str(predictions))
        y = [int(r["label"]) for r in rows]
        p = [float(r["probability"]) for r in rows]
        if any(label not in (0, 1) for label in y):
            raise ValueError("Non-binary labels in " + str(predictions))
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p):
            raise ValueError("Invalid probabilities in " + str(predictions))
        if len(rows) != config["split_sizes"]["test"]:
            raise ValueError("Prediction count differs from the recorded test split.")
        if any(r["arm"] != config["arm"] or int(r["seed"]) != config["seed"] for r in rows):
            raise ValueError("Prediction identifiers do not match the run configuration.")
        comparable = {k: config[k] for k in (
            "input_sha256", "upstream_commit", "epochs", "batch_size", "learning_rate",
            "optimizer", "weight_decay", "selection", "initialization", "device", "torch",
        )}
        label_map = dict(zip(nctids, y))
        if comparison_reference is None:
            comparison_reference, reference_labels = comparable, label_map
        elif comparable != comparison_reference or label_map != reference_labels:
            raise ValueError("Runs have different data/settings; do not present a matched comparison.")
        results.append({
            "run": str(config_path.parent.relative_to(root)), "arm": config["arm"],
            "seed": config["seed"], "n": len(y), "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "f1_at_0.5": float(f1_score(y, [int(v >= 0.5) for v in p], zero_division=0)),
            "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        })
    report = {"completed_runs": results, "incomplete_or_unevaluated_runs": missing,
              "note": "Per-run measurements; no claim of significance or prospective validity."}
    (root / "comparison_recalculated.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# HINT continuation comparison", "", "Recalculated from saved test predictions.", "",
             "| Arm | Seed | N | ROC-AUC | AP | F1 at 0.5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append("| {arm} | {seed} | {n} | {roc_auc:.6f} | {average_precision:.6f} | {f1:.6f} |".format(**r, f1=r["f1_at_0.5"]))
    if missing:
        lines += ["", "Incomplete or unevaluated runs: " + ", ".join(missing) + "."]
    if not results:
        lines += ["", "No completed test evaluations are available yet."]
    lines += ["", "Published HINT results and GLM's fresh training runs are separate reference experiments."]
    (root / "comparison_recalculated.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
