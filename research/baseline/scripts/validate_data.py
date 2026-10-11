"""
Step 2: Validate TOP Phase II benchmark inputs.
- counts + class balance per split
- duplicate NCT IDs + overlap between splits
- missing/invalid molecules (SMILES), icdcodes
- SHA-256 of original split files
- which fields the predictor consumes (from HINT/dataloader.py, verified by code audit)
Writes: /content/hint_evidence/data_integrity/data_integrity_report.json
"""
import csv, json, os, hashlib, sys
from collections import Counter

REPO = "/content/clinical-trial-outcome-prediction"
OUT = "/content/hint_evidence/data_integrity"
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_split(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    return header, rows

report = {"upstream_commit": os.popen("git rev-parse HEAD").read().strip(), "splits": {}}
all_ids = {}
for split in ["train", "valid", "test"]:
    path = f"data/phase_II_{split}.csv"
    header, rows = load_split(path)
    nctids = [r[0] for r in rows]
    labels = [int(r[3]) for r in rows]
    dup = {k: v for k, v in Counter(nctids).items() if v > 1}
    # invalid smiles: rdkit parse failure or empty
    bad_smiles_rows, empty_smiles_rows = [], []
    try:
        from rdkit import Chem
        for r in rows:
            smi_txt = r[8]
            if smi_txt in ("[]", "", "['']"):
                empty_smiles_rows.append(r[0]); continue
            inner = smi_txt[1:-1]
            smiles = [s.strip()[1:-1] for s in inner.split(",")]
            if all(Chem.MolFromSmiles(s) is None for s in smiles):
                bad_smiles_rows.append(r[0])
    except Exception as e:
        bad_smiles_rows = [f"rdkit check failed: {e}"]
    empty_icd = [r[0] for r in rows if r[6] in ("[]", "", '[""]')]
    report["splits"][split] = {
        "file": path, "rows": len(rows),
        "labels_1": sum(labels), "labels_0": len(labels) - sum(labels),
        "frac_1": round(sum(labels)/len(labels), 4),
        "duplicate_nctids": dup,
        "n_unique_nctids": len(set(nctids)),
        "rows_all_smiles_unparseable": bad_smiles_rows,
        "rows_empty_smiles": empty_smiles_rows,
        "rows_empty_icdcodes": empty_icd,
        "sha256": sha256(path),
    }
    all_ids[split] = set(nctids)

report["overlap"] = {
    "train∩valid": len(all_ids["train"] & all_ids["valid"]),
    "train∩test": len(all_ids["train"] & all_ids["test"]),
    "valid∩test": len(all_ids["valid"] & all_ids["test"]),
}
report["fields_consumed_by_predictor"] = {
    "nctid (row[0])": "identifier only",
    "label (row[3])": "training/eval target",
    "icdcodes (row[6])": "disease encoder input (GRAM)",
    "smiless (row[8])": "molecule encoder input (MPNN)",
    "criteria (row[9])": "protocol encoder input (BioBERT sentence embeddings)",
    "status/why_stop/phase/drugs": "NOT consumed by HINT/dataloader.csv_three_feature_2_dataloader (verified by code audit)",
}
report["paper_table1_phaseII"] = {"train": 4004, "valid": 445, "test": 1653,
    "success": [1925, 196, 918], "failure": [2079, 249, 735],
    "note": "Patterns 2022 Table 1; repo files have +1 row per split - difference investigated below"}
# investigate the +1: compare to paper counts via label balance
for split in ["train", "valid", "test"]:
    s = report["splits"][split]
    print(split, s["rows"], "pos:", s["labels_1"], "neg:", s["labels_0"], "dups:", s["duplicate_nctids"])

with open(f"{OUT}/data_integrity_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nreport saved:", f"{OUT}/data_integrity_report.json")
