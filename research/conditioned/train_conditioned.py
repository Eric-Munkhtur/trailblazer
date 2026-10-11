"""One final, fixed exploratory conditioned-attention run; see EXPERIMENT_PLAN.md.
Derived from the completed matched MSI continuation runner.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import shutil
import time


COMMIT = "8dc0497f23fdb84e2905da7655924a91e6e79798"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=("conditioned",), required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--embeddings-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Opt in only once this comparison's recipe is locked.")
    args = parser.parse_args()
    for value in (args.checkpoint_sha256, args.embeddings_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            parser.error("Supply complete 64-character SHA-256 hashes from GLM.")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive.")

    repo, assets, output = (p.resolve() for p in (args.repo, args.assets, args.output))
    if output.exists():
        parser.error("Output directory already exists; use a unique run directory.")
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != COMMIT:
        raise RuntimeError("Unexpected upstream commit: " + source_commit)
    source_patch = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", "HINT"], text=True
    )
    if source_patch:
        raise RuntimeError("This runner expects unmodified upstream HINT sources.")

    required = ["phase_II.ckpt", "sentence2embedding.pkl", "icdcode2ancestor_dict.pkl"]
    required += ["phase_II_" + split + ".csv" for split in ("train", "valid", "test")]
    for name in required:
        if not (assets / name).is_file():
            raise FileNotFoundError(assets / name)
    print("Hashing shared input assets...", flush=True)
    hashes = {name: sha256(assets / name) for name in required}
    for name, expected in (("phase_II.ckpt", args.checkpoint_sha256),
                           ("sentence2embedding.pkl", args.embeddings_sha256)):
        if hashes[name] != expected.lower():
            raise RuntimeError("Asset hash mismatch: " + name)

    control_run = args.control_run.resolve()
    control_config = json.loads((control_run / "config.json").read_text())
    control_history = json.loads((control_run / "history.json").read_text())
    control_metrics = json.loads((control_run / "metrics.json").read_text())
    if control_config["arm"] != "original" or "test" not in control_metrics:
        raise RuntimeError("Expected the completed original continuation control.")
    expected_settings = {"seed": args.seed, "epochs": args.epochs,
                         "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                         "input_sha256": hashes, "upstream_commit": source_commit,
                         "device": args.device, "optimizer": "Adam", "weight_decay": 0}
    for name, value in expected_settings.items():
        if control_config[name] != value:
            raise RuntimeError("Control mismatch: " + name)
    source_root = Path(__file__).resolve().parent
    lock_path = source_root / "experiment-lock.json"
    lock = json.loads(lock_path.read_text())
    for name, expected in lock["source_sha256"].items():
        if sha256(source_root / name) != expected:
            raise RuntimeError("Locked source changed: " + name)
    fixed = lock["settings"]
    for name in ("seed", "epochs", "batch_size", "learning_rate", "threads", "device"):
        if getattr(args, name) != fixed[name]:
            raise RuntimeError("Predeclared setting changed: " + name)
    if not args.evaluate_test:
        raise RuntimeError("This fixed run includes one final test evaluation.")

    # Importing HINT's loader loads its cache from data/ relative to cwd.
    # A private runtime directory supplies symlinks without modifying assets.
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in list(lock["source_sha256"]) + ["experiment-lock.json"]:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, target)
    runtime = output / "runtime"
    (runtime / "data").mkdir(parents=True)
    for name in required:
        if name != "phase_II.ckpt":
            (runtime / "data" / name).symlink_to(assets / name)
    os.chdir(runtime)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    from HINT.dataloader import csv_three_feature_2_dataloader
    from HINT.model import HINTModel  # Import before setting random seeds.
    from hint_conditioned import install_conditioned

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback.")
    device = torch.device(args.device)
    # The hash establishes identity, not trust: only use the authors'/GLM's
    # reviewed checkpoint. Full-model pickle loading executes Python code.
    model = torch.load(assets / "phase_II.ckpt", map_location="cpu", weights_only=False)
    if not isinstance(model, HINTModel):
        raise TypeError("Checkpoint is not the expected original HINTModel.")
    original_parameter_count = sum(p.numel() for p in model.parameters())
    install_conditioned(model, seed=args.seed)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if sum(p.numel() for p in model.parameters()) - original_parameter_count != 31040:
        raise RuntimeError("Unexpected added parameter count.")
    if str(torch.__version__) != control_config["torch"]:
        raise RuntimeError("PyTorch version differs from the control.")
    model.to(device)
    for module in model.modules():
        if hasattr(module, "device"):
            module.device = device
    # Upstream adjacency is an ordinary attribute, not a registered buffer.
    model.adj = model.adj.to(device)
    # These upstream lookup tables can stay on CPU; forward moves selected rows.
    model.disease_encoder.padding_matrix = model.disease_encoder.padding_matrix.cpu()
    model.disease_encoder.mask_matrix = model.disease_encoder.mask_matrix.cpu()

    # HINT imports contain manual_seed calls. Reseed AFTER all imports/loading.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    loaders = {}
    for split in ("train", "valid", "test"):
        loader = csv_three_feature_2_dataloader(
            str(assets / ("phase_II_" + split + ".csv")),
            shuffle=(split == "train"), batch_size=args.batch_size,
        )
        # Separate generators keep validation iteration out of training RNG.
        generator = torch.Generator().manual_seed(args.seed)
        loader.generator = generator
        if split == "train":
            loader.sampler.generator = generator
        loaders[split] = loader

    config = {
        "arm": args.arm, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "Adam", "weight_decay": 0,
        "selection": "lowest sample-averaged validation BCE, including epoch 0",
        "initialization": "continuation from authors' immutable Phase II checkpoint",
        "split_sizes": {s: len(loader.dataset) for s, loader in loaders.items()},
        "input_sha256": hashes, "upstream_commit": source_commit,
        "device": str(device), "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "test_evaluation_enabled": args.evaluate_test,
        "argv": sys.argv, "started_at_unix": time.time(),
        "runner_sha256": sha256(Path(__file__)),
        "attention_module_sha256": sha256(source_root / "hint_conditioned/protocol.py"),
        "architecture": lock["architecture"],
        "added_parameters": 31040,
        "exploratory_after_prior_test_inspection": True,
        "control_run": str(control_run),
        "control_config_sha256": sha256(control_run / "config.json"),
        "experiment_lock_sha256": sha256(lock_path),
        "cpu_threads": args.threads,
        "differences_from_glm_replication": [
            "Continuation from a shared checkpoint, not new outcome-model initialization",
            "Seeds applied after imports/loading; dedicated data-loader generators",
            "Sample-averaged validation loss, not sum of batch means; identical in both arms",
            "MSI software/hardware recorded separately from Colab",
        ],
    }
    write_json(output / "config.json", config)
    write_json(output / "disease_mapping.json", {
        "code2index": {str(k): int(v) for k, v in model.disease_encoder.code2index.items()},
        "icdcode2ancestor": model.disease_encoder.icdcode2ancestor,
        "base_checkpoint_sha256": hashes["phase_II.ckpt"],
        "reload": "Load immutable base model; install_conditioned; strict state_dict load",
    })
    (output / "environment.txt").write_text(subprocess.check_output(
        ["uv", "pip", "freeze", "--python", sys.executable], text=True
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def predict(loader):
        model.eval()
        total_loss, count, rows = 0.0, 0, []
        for nctids, labels, smiles, codes, criteria in loader:
            target = labels.to(device).float()
            logits = model(smiles, codes, criteria).view(-1)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation output.")
            total_loss += loss_fn(logits, target).item() * len(nctids)
            count += len(nctids)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            rows.extend(zip(nctids, labels.tolist(), probabilities))
        if count == 0:
            raise RuntimeError("Empty evaluation split.")
        return total_loss / count, rows

    def save_checkpoint(filename, epoch, validation_loss):
        path = output / filename
        temp = path.with_suffix(".tmp")
        torch.save({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "epoch": epoch, "validation_loss": validation_loss, "config": config,
        }, temp)
        temp.replace(path)

    def save_predictions(split, rows):
        with open(output / ("predictions_" + split + ".csv"), "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["nctid", "label", "probability", "predicted_class", "arm", "seed"])
            for nctid, label, probability in rows:
                writer.writerow([nctid, int(label), probability, int(probability >= 0.5),
                                 args.arm, args.seed])
        y = [int(row[1]) for row in rows]
        p = [row[2] for row in rows]
        pred = [int(value >= 0.5) for value in p]
        return {
            "n": len(y), "n_positive": sum(y), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
            "f1_at_0.5": f1_score(y, pred, zero_division=0),
            "precision_at_0.5": precision_score(y, pred, zero_division=0),
            "recall_at_0.5": recall_score(y, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }

    start = time.monotonic()
    print("Evaluating epoch-0 validation loss...", flush=True)
    best_loss, _ = predict(loaders["valid"])
    control_initial = control_history[0]["validation_loss"]
    initial_delta = abs(best_loss - control_initial)
    write_json(output / "initialization_check.json", {
        "control_epoch0_validation_bce": control_initial,
        "candidate_epoch0_validation_bce": best_loss,
        "absolute_difference": initial_delta, "tolerance": 1e-6,
    })
    if initial_delta > 1e-6:
        raise RuntimeError("Initial validation loss differs from the matched control.")
    print(json.dumps({"initialization_validation_bce_difference": initial_delta,
                      "added_parameters": 31040}), flush=True)
    best_epoch = 0
    save_checkpoint("best_state.pt", 0, best_loss)
    history = [{"epoch": 0, "validation_loss": best_loss}]
    write_json(output / "history.json", history)

    for epoch in range(1, args.epochs + 1):
        model.train()
        synchronize()
        epoch_start = time.monotonic()
        total_loss, total_n = 0.0, 0
        for batch_index, (nctids, labels, smiles, codes, criteria) in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(smiles, codes, criteria).view(-1)
            loss = loss_fn(logits, labels.to(device).float())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(nctids)
            total_n += len(nctids)
            if batch_index == 1 or batch_index % 10 == 0:
                synchronize()
                print(json.dumps({"epoch": epoch, "batch": batch_index,
                                  "batches": len(loaders["train"]),
                                  "mean_seconds_per_batch": (time.monotonic() - epoch_start) / batch_index,
                                  "training_loss": total_loss / total_n}), flush=True)
        synchronize()
        train_seconds = time.monotonic() - epoch_start
        validation_start = time.monotonic()
        validation_loss, _ = predict(loaders["valid"])
        synchronize()
        record = {"epoch": epoch, "train_loss": total_loss / total_n,
                  "validation_loss": validation_loss, "train_seconds": train_seconds,
                  "validation_seconds": time.monotonic() - validation_start}
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint("best_state.pt", epoch, best_loss)
        save_checkpoint("last_state.pt", epoch, validation_loss)

    # This package writes tensor state dictionaries; the upstream input alone
    # needs trusted full-model unpickling. Config uses plain JSON-compatible types.
    best = torch.load(output / "best_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"], strict=True)
    metrics = {"arm": args.arm, "seed": args.seed, "best_epoch": best_epoch,
               "best_validation_loss": best_loss}
    splits = ["valid", "test"] if args.evaluate_test else ["valid"]
    for split in splits:
        if split == "test":
            with open(output / "test_evaluation_started.json", "x") as marker:
                json.dump({"at_unix": time.time(), "selected_epoch": best_epoch,
                           "checkpoint_sha256": sha256(output / "best_state.pt"),
                           "selection": config["selection"]}, marker, indent=2)
        _, rows = predict(loaders[split])
        metrics[split] = save_predictions(split, rows)
    metrics["parameter_diagnostics"] = {
        name: {"l2": float(parameter.detach().norm().cpu()),
               "nonzero": int(torch.count_nonzero(parameter.detach()).cpu())}
        for name, parameter in model.protocol_encoder.named_parameters()
        if "query" in name or "sentence_key" in name
    }
    metrics["elapsed_seconds"] = time.monotonic() - start
    metrics["best_checkpoint_sha256"] = sha256(output / "best_state.pt")
    if device.type == "cuda":
        metrics["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "metrics.json", metrics)
    if sha256(assets / "phase_II.ckpt") != hashes["phase_II.ckpt"]:
        raise RuntimeError("Starting checkpoint changed during the experiment.")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
