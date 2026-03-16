"""
compare.py
==========
LSST Time Series Classification — Unified Comparison Runner.

Runs both experimental pipelines sequentially and produces a consolidated
comparison report:

    Experiment A  BaselineFCN       (Wang et al., 2017 — trained from scratch)
    Experiment B  Chronos + SVM     (amazon/chronos-t5-small, RBF kernel probe)

Outputs
-------
  results/metrics/comparison_summary.json    — machine-readable
  results/metrics/comparison_report.txt      — human-readable ASCII table
  results/checkpoints/best_baseline_fcn.pt   — best FCN checkpoint
  results/logs/*_history.json                — training curves (FCN only)

Usage
-----
    python compare.py
    python compare.py --fcn_epochs 500 --results_dir experiments/run1
    python compare.py --skip_fcn          # only run Chronos experiment
    python compare.py --skip_chronos      # only run FCN
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chronos import ChronosPipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from baseline_model import BaselineFCN
from data_utils import compute_class_weights, get_dataloaders, load_lsst
from train_chronos import extract_embeddings, multi_pool_features


# =============================================================================
# Shared utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_dirs(base: str = "results") -> dict:
    dirs = {
        "base":        base,
        "checkpoints": os.path.join(base, "checkpoints"),
        "logs":        os.path.join(base, "logs"),
        "metrics":     os.path.join(base, "metrics"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Experiment A — BaselineFCN
# =============================================================================

def run_baseline_fcn(
    X_train, y_train, X_test, y_test, le,
    device, dirs,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 40,
    val_split: float = 0.15,
    seed: int = 42,
) -> dict:
    print(f"\n{'#' * 70}")
    print("  EXPERIMENT A  —  BaselineFCN (Wang et al., 2017)")
    print(f"{'#' * 70}")

    set_seed(seed)
    t0 = time.time()

    n_classes = len(le.classes_)
    seq_len   = X_train.shape[1]
    n_ch      = X_train.shape[2]

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=batch_size, val_split=val_split, seed=seed,
    )
    class_weights = compute_class_weights(y_train, device)

    model = BaselineFCN(
        n_channels=n_ch, n_classes=n_classes, seq_len=seq_len
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    # --- training loop -------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler   = cosine_warmup_scheduler(
        optimizer, int(0.05 * total_steps), total_steps
    )

    best_f1 = best_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    ckpt = os.path.join(dirs["checkpoints"], "best_baseline_fcn.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tl.append(loss.item())

        model.eval()
        vl, preds, labs = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl.append(F.cross_entropy(logits, yb).item())
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(yb.cpu().numpy())

        va = accuracy_score(labs, preds)
        vf = f1_score(labs, preds, average="weighted")
        history["train_loss"].append(np.mean(tl))
        history["val_loss"].append(np.mean(vl))
        history["val_acc"].append(va)
        history["val_f1"].append(vf)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={np.mean(tl):.4f}  val_loss={np.mean(vl):.4f}  "
                f"val_acc={va:.4f}  val_f1={vf:.4f}"
            )

        if vf > best_f1:
            best_f1, best_acc = vf, va
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))

    # --- test ----------------------------------------------------------------
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().numpy())
            labs.extend(y.numpy())

    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds, average="weighted")
    elapsed = time.time() - t0

    print(f"\n  Test Accuracy={acc:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")

    report_dict = classification_report(
        labs, preds,
        target_names=le.classes_,
        digits=4, output_dict=True, zero_division=0,
    )
    print(classification_report(
        labs, preds,
        target_names=le.classes_,
        digits=4, zero_division=0,
    ))

    result = {
        "model": "BaselineFCN",
        "accuracy": acc,
        "f1_weighted": f1,
        "n_params": n_params,
        "training_seconds": elapsed,
        "report": report_dict,
    }
    with open(os.path.join(dirs["metrics"], "baseline_fcn_results.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    with open(os.path.join(dirs["logs"], "baseline_fcn_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    return result


# =============================================================================
# Experiment B — Chronos + SVM
# =============================================================================

def run_chronos_svm(
    X_train, y_train, X_test, y_test, le,
    device, dirs,
    batch_size: int = 128,
) -> dict:
    print(f"\n{'#' * 70}")
    print("  EXPERIMENT B  —  Chronos + SVM (amazon/chronos-t5-small)")
    print(f"{'#' * 70}")

    t0 = time.time()

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=str(device),
        torch_dtype=torch.float32,
    )
    pipeline.model.eval()

    print("\n  Extracting embeddings ...")
    train_emb = extract_embeddings(X_train, pipeline, batch_size=batch_size, label="[train]")
    test_emb  = extract_embeddings(X_test,  pipeline, batch_size=batch_size, label="[test]")

    print(f"\n  Embedding shape : {train_emb.shape}  (N, C, patches, D)")

    X_tr = multi_pool_features(train_emb)
    X_te = multi_pool_features(test_emb)
    print(f"  Feature dimension after multi-pool : {X_tr.shape[1]:,}")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    t_clf = time.time()
    print("\n  Training SVC (RBF kernel) ...")
    clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")
    elapsed_clf = time.time() - t_clf

    print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  ({elapsed_clf:.0f}s)")
    print(classification_report(
        y_test, preds,
        target_names=[str(c) for c in le.classes_],
        digits=4, zero_division=0,
    ))

    report_dict = classification_report(
        y_test, preds,
        target_names=[str(c) for c in le.classes_],
        digits=4, output_dict=True, zero_division=0,
    )
    result = {
        "model":            "Chronos + SVM",
        "accuracy":         acc,
        "f1_weighted":      f1,
        "feature_dim":      int(X_tr.shape[1]),
        "training_seconds": elapsed_clf,
        "report":           report_dict,
    }
    with open(os.path.join(dirs["metrics"], "chronos_plus_svm_results.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"\n  Total Chronos pipeline time : {time.time() - t0:.0f}s")
    return {"Chronos + SVM": result}


# =============================================================================
# Consolidated report
# =============================================================================

def print_comparison(all_results: dict, dirs: dict) -> None:
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    header = f"  {'Model':<30}  {'Accuracy':>10}  {'F1 (w.)':>10}"
    divider = "  " + "-" * 56
    print(header)
    print(divider)

    rows = []
    for name, r in all_results.items():
        acc = r.get("accuracy", float("nan"))
        f1  = r.get("f1_weighted", float("nan"))
        rows.append((name, acc, f1))
        print(f"  {name:<30}  {acc:>10.4f}  {f1:>10.4f}")

    print(f"{'=' * 70}")

    # Machine-readable summary
    summary = {name: {"accuracy": r["accuracy"], "f1_weighted": r["f1_weighted"]}
               for name, r in all_results.items()}
    path_json = os.path.join(dirs["metrics"], "comparison_summary.json")
    with open(path_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Human-readable ASCII table
    lines = [
        "LSST Time Series Classification — Experiment Comparison",
        "=" * 56,
        f"{'Model':<30}  {'Accuracy':>10}  {'F1 (weighted)':>13}",
        "-" * 56,
    ]
    for name, acc, f1 in sorted(rows, key=lambda x: -x[1]):
        lines.append(f"{name:<30}  {acc:>10.4f}  {f1:>13.4f}")
    lines.append("=" * 56)

    path_txt = os.path.join(dirs["metrics"], "comparison_report.txt")
    with open(path_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n  Saved -> {path_json}")
    print(f"  Saved -> {path_txt}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FCN baseline + Chronos probes and compare on LSST"
    )
    parser.add_argument("--fcn_epochs",    type=int,   default=300)
    parser.add_argument("--fcn_batch",     type=int,   default=64)
    parser.add_argument("--fcn_lr",        type=float, default=1e-3)
    parser.add_argument("--fcn_patience",  type=int,   default=40)
    parser.add_argument("--chronos_batch", type=int,   default=128)
    parser.add_argument("--results_dir",   type=str,   default="results")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--skip_fcn",      action="store_true",
                        help="Skip Experiment A (BaselineFCN)")
    parser.add_argument("--skip_chronos",  action="store_true",
                        help="Skip Experiment B (Chronos + SVM)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")
    dirs = setup_dirs(args.results_dir)
    set_seed(args.seed)

    X_train, y_train, X_test, y_test, le = load_lsst(normalise=True)

    all_results = {}

    if not args.skip_fcn:
        r = run_baseline_fcn(
            X_train, y_train, X_test, y_test, le,
            device, dirs,
            epochs=args.fcn_epochs,
            batch_size=args.fcn_batch,
            lr=args.fcn_lr,
            patience=args.fcn_patience,
            seed=args.seed,
        )
        all_results["BaselineFCN"] = r

    if not args.skip_chronos:
        chronos_results = run_chronos_svm(
            X_train, y_train, X_test, y_test, le,
            device, dirs,
            batch_size=args.chronos_batch,
        )
        all_results.update(chronos_results)

    if all_results:
        print_comparison(all_results, dirs)

    print(f"\n  All outputs -> {args.results_dir}/")


if __name__ == "__main__":
    main()
