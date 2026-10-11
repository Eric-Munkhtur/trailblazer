"""
Step 2/3: Score the authors' SHIPPED test predictions (results/phase_II_test.csv) with:
  (a) our independent evaluator: ROC-AUC + AP from continuous probs; F1/precision/recall/confusion @0.5; also @val-free default
  (b) upstream evaluator semantics: 'PR-AUC' = average_precision_score on BINARIZED preds; F1@0.5; ROC-AUC continuous
This validates the independent evaluator against a fixed prediction set and quantifies the
upstream PR-AUC underestimation, WITHOUT needing the model or embedding cache.
Writes: /content/hint_evidence/authors_shipped_predictions/{metrics.json,predictions.csv}
"""
import csv, json, os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score, confusion_matrix)

REPO = "/content/clinical-trial-outcome-prediction"
OUT = "/content/hint_evidence/authors_shipped_predictions"
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)

def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

# authors' predictions file: header has 'prediction' where original has 'label'
_, pred_rows = load_csv("results/phase_II_test.csv")
nct2pred = {}
dup_preds = []
for r in pred_rows:
    if r[0] in nct2pred: dup_preds.append(r[0])
    nct2pred[r[0]] = float(r[3])

# official test split with labels
_, test_rows = load_csv("data/phase_II_test.csv")
nctids, labels = [], []
missing_pred = []
for r in test_rows:
    if r[0] not in nct2pred:
        missing_pred.append(r[0]); continue
    nctids.append(r[0]); labels.append(int(r[3]))
preds = [nct2pred[n] for n in nctids]
y = np.array(labels); p = np.array(preds)

def independent_eval(p, y, threshold=0.5):
    pred_cls = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "f1@0.5": float(f1_score(y, pred_cls)),
        "precision@0.5": float(precision_score(y, pred_cls)),
        "recall@0.5": float(recall_score(y, pred_cls)),
        "accuracy@0.5": float(accuracy_score(y, pred_cls)),
        "confusion_matrix@0.5": confusion_matrix(y, pred_cls).tolist(),
        "n": int(len(y)), "n_pos": int(y.sum()), "frac_pos": float(y.mean()),
    }

def upstream_eval(p, y, threshold=0.5):
    """Replicates HINT/model.py Interaction.evaluation exactly (line 77-107)."""
    pred_cls = list(map(lambda x: 0 if x < threshold else 1, p))
    return {
        "roc_auc_continuous": float(roc_auc_score(y, p)),
        "prauc_binarized_UPSTREAM": float(average_precision_score(y, pred_cls)),
        "f1@0.5": float(f1_score(y, pred_cls)),
        "precision@0.5": float(precision_score(y, pred_cls)),
        "recall@0.5": float(recall_score(y, pred_cls)),
        "accuracy@0.5": float(accuracy_score(y, pred_cls)),
    }

ind = independent_eval(p, y)
up = upstream_eval(p, y)
res = {
    "experiment": "authors' shipped test predictions (results/phase_II_test.csv) scored by us",
    "source_file_sha256_note": "see data_integrity_report for split hashes; predictions file hashed below",
    "n_test_rows": len(test_rows), "n_scored": len(y),
    "n_missing_prediction_in_authors_file": missing_pred,
    "duplicate_nctids_in_authors_file": dup_preds,
    "independent_evaluator": ind,
    "upstream_evaluator_replication": up,
    "paper_table2_phaseII": {"PR-AUC": "0.629±0.009", "F1": "0.620±0.008", "ROC-AUC": "0.645±0.006",
                              "note": "paper 'PR-AUC' corresponds to upstream binarized-AP metric"},
}
import hashlib
h = hashlib.sha256()
with open("results/phase_II_test.csv","rb") as f:
    for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
res["authors_predictions_sha256"] = h.hexdigest()

with open(f"{OUT}/metrics.json", "w") as f: json.dump(res, f, indent=2)
with open(f"{OUT}/predictions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["nctid","label","probability","predicted_class@0.5","run_id"])
    for n, yy, pp in zip(nctids, y, p):
        w.writerow([n, int(yy), float(pp), int(pp >= 0.5), "authors_shipped"])
print(json.dumps(res, indent=2))
