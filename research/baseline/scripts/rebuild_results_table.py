"""
Standalone metric regeneration: rebuilds ALL reported metrics + the results table from saved
per-example prediction CSVs only (no model, no data files, no retraining).
Usage: python rebuild_results_table.py <evidence_dir>
  evidence_dir must contain */preds_test.csv (+ preds_valid.csv where saved)
Verifies recomputed values against each run's saved metrics.json (tol 1e-9) when present,
and writes RESULTS_TABLE.md + regenerated_metrics.json.
"""
import sys, os, json, glob, csv
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score, confusion_matrix)

ev = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []

def metrics_from_preds(p, y, threshold=0.5):
    p = np.asarray(p, float); y = np.asarray(y, int); pred = (p >= threshold).astype(int)
    predbin = pred.tolist()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision_continuous": float(average_precision_score(y, p)),
        "f1@0.5": float(f1_score(y, pred)),
        "precision@0.5": float(precision_score(y, pred)),
        "recall@0.5": float(recall_score(y, pred)),
        "accuracy@0.5": float(accuracy_score(y, pred)),
        "confusion_matrix@0.5": confusion_matrix(y, pred).tolist(),
        "prauc_binarized_upstream": float(average_precision_score(y, predbin)),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }

def load_preds(path):
    nctids, labels, probs = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            nctids.append(r["nctid"]); labels.append(int(r["label"])); probs.append(float(r["probability"]))
    assert len(nctids) == len(set(nctids)), f"duplicate nctids in {path}"
    return nctids, labels, probs

runs = []
for pred_file in sorted(glob.glob(f"{ev}/**/preds_test.csv", recursive=True)):
    run_dir = os.path.dirname(pred_file)
    run_id = None
    with open(pred_file, newline="") as f:
        for r in csv.DictReader(f):
            run_id = r.get("run_id", os.path.basename(run_dir)); break
    nctids, y, p = load_preds(pred_file)
    m = metrics_from_preds(p, y)
    m["run_id"] = run_id; m["dir"] = run_dir
    # verify against saved metrics.json if present
    mj = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mj):
        saved = json.load(open(mj))
        node = saved.get("test", {}).get("independent", {}) or saved.get("independent_evaluator", {})
        # map authors_shipped schema (average_precision) to our key
        if "average_precision" in node and "average_precision_continuous" not in node:
            node = dict(node, average_precision_continuous=node["average_precision"])
        ok = abs(node.get("roc_auc", -9) - m["roc_auc"]) < 1e-9 and \
             abs(node.get("average_precision_continuous", -9) - m["average_precision_continuous"]) < 1e-9
        m["matches_saved_metrics"] = bool(ok)
    runs.append(m)

# aggregate training seeds
seeds = [r for r in runs if str(r.get("run_id","")).startswith("hint_p2_s")]
agg = None
if len(seeds) >= 2:
    agg = {k: {"mean": float(np.mean([s[k] for s in seeds])), "std": float(np.std([s[k] for s in seeds]))}
           for k in ["roc_auc", "average_precision_continuous", "f1@0.5", "prauc_binarized_upstream"]}
    agg["n_runs"] = len(seeds)

out = {"runs": runs, "training_seed_aggregate": agg}
json.dump(out, open(f"{ev}/regenerated_metrics.json", "w"), indent=2)

lines = ["# HINT Phase II baseline — regenerated results table (from saved predictions only)", "",
         "| run_id | n | ROC-AUC | AP (continuous) | F1@0.5 | 'PR-AUC' upstream (binarized) | matches saved |",
         "|---|---|---|---|---|---|---|"]
for r in runs:
    lines.append(f"| {r['run_id']} | {r['n']} | {r['roc_auc']:.4f} | {r['average_precision_continuous']:.4f} | "
                 f"{r['f1@0.5']:.4f} | {r['prauc_binarized_upstream']:.4f} | {r.get('matches_saved_metrics','n/a')} |")
if agg:
    lines += ["", f"Training runs aggregate over {agg['n_runs']} seeds (mean ± population std):",
        f"- ROC-AUC {agg['roc_auc']['mean']:.4f} ± {agg['roc_auc']['std']:.4f}",
        f"- AP (continuous) {agg['average_precision_continuous']['mean']:.4f} ± {agg['average_precision_continuous']['std']:.4f}",
        f"- F1@0.5 {agg['f1@0.5']['mean']:.4f} ± {agg['f1@0.5']['std']:.4f}",
        f"- 'PR-AUC' upstream {agg['prauc_binarized_upstream']['mean']:.4f} ± {agg['prauc_binarized_upstream']['std']:.4f}"]
lines += ["", "Paper Table 2 (Patterns 2022, 30 runs): PR-AUC 0.629±0.009, F1 0.620±0.008, ROC-AUC 0.645±0.006 —",
          "computed with the UPSTREAM evaluator (binarized 'PR-AUC'); not directly comparable to continuous AP."]
table = "\n".join(lines)
open(f"{ev}/RESULTS_TABLE.md", "w").write(table)
print(table)
