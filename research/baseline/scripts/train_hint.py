"""
Experiment: 'original HINT trained by us' (Phase II).

Faithful replication of upstream recipe (HINT/learn_phaseII.py + Interaction.learn, model.py L210-234):
  - MPNN(mpnn_hidden_size=50, mpnn_depth=3); GRAM(embedding_dim=50); Protocol_Embedding(output_dim=50, highway_num=3)
  - HINTModel(global_embed_size=50, highway_num_layer=2, gnn_hidden_size=50, epoch=5, lr=3e-4, weight_decay=0)
  - init_pretrain(admet_model) from shipped save_model/admet_model.ckpt (molecule encoder weights)
  - Adam(lr=3e-4, wd=0); BCEWithLogitsLoss; batch 32; train shuffle=True; 5 epochs
  - checkpoint selection: lowest validation loss (sum of batch losses), incl. epoch-0 baseline, deepcopy per upstream
DELIBERATE DEVIATIONS from upstream (documented in report):
  1. device: cuda instead of hardcoded cpu (learn_phaseII.py)
  2. best model restored into caller via load_state_dict (upstream `self = deepcopy(best_model)` is a no-op bug:
     the SAVED upstream checkpoint is the final-epoch model, while its in-learn test metrics use the best model)
  3. test set evaluated ONCE at the end by our independent evaluator (+ upstream-style metrics for comparison);
     never used for selection
  4. seeds set via CLI arg (upstream hardcodes torch.manual_seed(0) at import)
Usage: python train_hint.py <seed>
Writes: /content/hint_evidence/training/run_s<seed>_<ts>/{config.json, log.json, metrics.json,
        preds_valid.csv, preds_test.csv, best_model_state.pt, final_model_state.pt}
"""
import os, sys, json, time, csv, random
import numpy as np
import torch

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RUN_ID = f"hint_p2_s{SEED}"  # fixed name so an interrupted run can resume
REPO = "/content/clinical-trial-outcome-prediction"
os.chdir(REPO); sys.path.insert(0, ".")
OUT = f"/content/hint_evidence/training/{RUN_ID}"
os.makedirs(OUT, exist_ok=True)

t_start = time.time()
from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.molecule_encode import MPNN
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel
from copy import deepcopy
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score, confusion_matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "run_id": RUN_ID, "seed": SEED, "device": str(device),
    "upstream_commit": os.popen("git rev-parse HEAD").read().strip(),
    "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
    "seeding": "torch/np/random seeded AFTER HINT imports (import-time reseeding bug fixed)",
    "recipe": {"arch": "HINTModel", "mpnn_hidden_size": 50, "mpnn_depth": 3,
               "gram_embedding_dim": 50, "protocol_output_dim": 50, "protocol_highway_num": 3,
               "global_embed_size": 50, "highway_num_layer": 2, "gnn_hidden_size": 50,
               "epoch": 5, "lr": 3e-4, "weight_decay": 0, "batch_size": 32,
               "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
               "init_pretrain": "save_model/admet_model.ckpt (shipped ADMET model provides molecule_encoder)",
               "selection": "lowest validation loss (sum of batch BCE losses), evaluated before training and after each epoch"},
    "deviations_from_upstream": [
        "device=cuda (upstream hardcodes cpu)",
        "best-model weights restored correctly via load_state_dict (upstream self=deepcopy(best_model) bug)",
        "test evaluated once at end; not used in selection",
        "torch.manual_seed(seed) via CLI set AFTER imports (upstream fixes 0 at import; import-time reseeding would equalize all seeds)",
        "per-epoch resumable checkpointing; GRAM mapping saved with weights",
    ],
}
json.dump(config, open(f"{OUT}/config.json", "w"), indent=2)

# Seed AFTER imports: HINT modules re-seed torch/numpy at import time (model.py manual_seed(0),
# module.py manual_seed(4)/np.random.seed(1)), which would silently override any earlier seed.
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
PYTHONHASHSEED_NOTE = os.environ.get("PYTHONHASHSEED", "<unset>")

train_loader = csv_three_feature_2_dataloader("data/phase_II_train.csv", shuffle=True, batch_size=32)
valid_loader = csv_three_feature_2_dataloader("data/phase_II_valid.csv", shuffle=False, batch_size=32)
test_loader = csv_three_feature_2_dataloader("data/phase_II_test.csv", shuffle=False, batch_size=32)

mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
admet_model = torch.load("save_model/admet_model.ckpt", weights_only=False, map_location="cpu")
admet_model = admet_model.to(device); admet_model.set_device(device)

model = HINTModel(molecule_encoder=mpnn_model, disease_encoder=gram_model,
                  protocol_encoder=protocol_model, device=device,
                  global_embed_size=50, highway_num_layer=2, prefix_name="phase_II",
                  gnn_hidden_size=50, epoch=5, lr=3e-4, weight_decay=0)
model.init_pretrain(admet_model)
model = model.to(device)
model.molecule_encoder.set_device(device)

opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
loss_fn = torch.nn.BCEWithLogitsLoss()

# resume from per-epoch checkpoint if this run was interrupted
start_epoch, resumed = 0, False
if os.path.exists(f"{OUT}/checkpoint_resume.pt"):
    ck = torch.load(f"{OUT}/checkpoint_resume.pt", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
    best_valid = ck["best_valid_loss"]; best_state = ck["best_state"]; start_epoch = ck["epoch"] + 1
    resumed = True
    print(f"[{RUN_ID}] RESUMED from epoch {ck['epoch']} (best_valid {best_valid:.4f})", flush=True)

@torch.no_grad()
def valid_loss(model, loader):
    model.eval()
    total = 0.0
    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in loader:
        label_vec = label_vec.to(device)
        out = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
        total += loss_fn(out, label_vec.float()).item()
    model.train()
    return total

log = {"valid_loss_after_epoch": [], "epoch_seconds": [], "train_batch_losses_last_epoch": []}
if not resumed:
    best_valid = valid_loss(model, valid_loader)
    best_state = deepcopy(model.state_dict())
    log["valid_loss_before_training"] = best_valid
    print(f"[{RUN_ID}] initial valid loss {best_valid:.4f}", flush=True)
for ep in range(start_epoch, 5):
    t0 = time.time()
    model.train()
    batch_losses = []
    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in train_loader:
        label_vec = label_vec.to(device)
        out = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
        loss = loss_fn(out, label_vec.float())
        batch_losses.append(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    vl = valid_loss(model, valid_loader)
    log["valid_loss_after_epoch"].append(vl)
    log["epoch_seconds"].append(round(time.time() - t0, 1))
    log["train_batch_losses_last_epoch"] = [round(x, 4) for x in batch_losses[-20:]]
    if vl < best_valid:
        best_valid = vl; best_state = deepcopy(model.state_dict())
        log["best_epoch"] = ep
    # crash-resilient: save resumable state every epoch
    torch.save({"epoch": ep, "model": model.state_dict(), "best_state": best_state,
                "optimizer": opt.state_dict(), "best_valid_loss": best_valid,
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                "py_rng": random.getstate()},
               f"{OUT}/checkpoint_resume.pt")
    print(f"[{RUN_ID}] epoch {ep}: valid_loss={vl:.4f} (best {best_valid:.4f} @ep {log.get('best_epoch','-1')}) {log['epoch_seconds'][-1]}s", flush=True)

torch.save({"state_dict": best_state, "best_valid_loss": best_valid, "config": config}, f"{OUT}/best_model_state.pt")
# GRAM disease-code lookup is built from unordered sets (code2index order varies per process);
# saving it makes the state_dict reloadable in a fresh process.
torch.save({"code2index": gram_model.code2index, "all_code_lst": gram_model.all_code_lst,
            "padding_matrix": gram_model.padding_matrix, "mask_matrix": gram_model.mask_matrix,
            "icdcode2ancestor": {k: list(v) for k, v in gram_model.icdcode2ancestor.items()}},
           f"{OUT}/gram_mapping.pt")
torch.save({"state_dict": model.state_dict(), "final_valid_loss": vl}, f"{OUT}/final_model_state.pt")

# restore best (correct fix of upstream bug) and evaluate ONCE
model.load_state_dict(best_state)
model.eval()

@torch.no_grad()
def predict(loader):
    nctids, labels, probs = [], [], []
    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in loader:
        out = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
        probs.extend(torch.sigmoid(out).cpu().tolist())
        labels.extend(label_vec.tolist()); nctids.extend(nctid_lst)
    return nctids, labels, probs

def independent_eval(p, y, threshold=0.5):
    p = np.array(p); y = np.array(y); pred = (p >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision_continuous": float(average_precision_score(y, p)),
            "f1@0.5": float(f1_score(y, pred)), "precision@0.5": float(precision_score(y, pred)),
            "recall@0.5": float(recall_score(y, pred)), "accuracy@0.5": float(accuracy_score(y, pred)),
            "confusion_matrix@0.5": confusion_matrix(y, pred).tolist(), "n": int(len(y)), "n_pos": int(y.sum())}

def upstream_eval(p, y, threshold=0.5):
    pred = [0 if x < threshold else 1 for x in p]
    return {"roc_auc_continuous": float(roc_auc_score(y, p)),
            "prauc_binarized_UPSTREAM": float(average_precision_score(y, pred)),
            "f1@0.5": float(f1_score(y, pred))}

metrics = {"run_id": RUN_ID, "seed": SEED, "best_epoch": log.get("best_epoch"),
           "best_valid_loss": best_valid, "log": log, "elapsed_seconds": round(time.time()-t_start, 1)}
for name, loader in [("valid", valid_loader), ("test", test_loader)]:
    nctids, labels, probs = predict(loader)
    metrics[name] = {"independent": independent_eval(probs, labels), "upstream_replication": upstream_eval(probs, labels)}
    with open(f"{OUT}/preds_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["nctid","label","probability","predicted_class@0.5","run_id","seed"])
        for n, l, p in zip(nctids, labels, probs): w.writerow([n, int(l), float(p), int(p >= 0.5), RUN_ID, SEED])
    print(name, json.dumps(metrics[name], indent=2), flush=True)

json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
json.dump(log, open(f"{OUT}/log.json", "w"), indent=2)
print(f"[{RUN_ID}] DONE in {metrics['elapsed_seconds']}s", flush=True)
