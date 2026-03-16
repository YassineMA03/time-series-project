"""
train_baseline.py
=================
LSST Time Series Classification — Experiment A: BaselineFCN (no pretraining).

Pipeline
--------
1. Load LSST via tslearn (instance-normalised)
2. Train BaselineFCN with:
     - Sqrt-damped class-weighted CrossEntropyLoss + label smoothing 0.1
     - AdamW optimiser
     - Cosine LR schedule with 5 % linear warmup
     - Gradient clipping (max-norm 1.0)
     - Early stopping on weighted F1 (patience=40)
3. Restore best checkpoint and evaluate on the held-out test set
4. Save metrics to results/metrics/baseline_fcn_results.json

Usage
-----
    python train_baseline.py
    python train_baseline.py --epochs 500 --lr 5e-4
    python train_baseline.py --results_dir experiments/run1
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from baseline_model import BaselineFCN
from data_utils import compute_class_weights, get_dataloaders, load_lsst


# =============================================================================
# Helpers
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
# Training
# =============================================================================

def train(
    model,
    train_loader,
    val_loader,
    device,
    dirs,
    class_weights=None,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 40,
):
    print(f"\n{'=' * 70}")
    print("  TRAINING  BaselineFCN")
    print(f"{'=' * 70}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * len(train_loader)
    scheduler   = cosine_warmup_scheduler(
        optimizer, int(0.05 * total_steps), total_steps
    )

    best_val_f1 = best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    ckpt = os.path.join(dirs["checkpoints"], "best_baseline_fcn.pt")

    for epoch in range(1, epochs + 1):

        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses, preds, labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(F.cross_entropy(logits, yb).item())
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        va = accuracy_score(labels, preds)
        vf = f1_score(labels, preds, average="weighted")

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_f1"].append(vf)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={tl:.4f}  val_loss={vl:.4f}  "
                f"val_acc={va:.4f}  val_f1={vf:.4f}"
            )

        if vf > best_val_f1:
            best_val_f1, best_val_acc = vf, va
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"\n  Best val_acc={best_val_acc:.4f}  best_val_f1={best_val_f1:.4f}")

    with open(os.path.join(dirs["logs"], "baseline_fcn_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, test_loader, device, le, dirs) -> tuple[float, float]:
    print(f"\n{'=' * 70}")
    print("  TEST  BaselineFCN")
    print(f"{'=' * 70}")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")

    print(f"\n  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(
        f"\n{classification_report(labels, preds, target_names=le.classes_, digits=4, zero_division=0)}"
    )

    result = {
        "model":       "BaselineFCN",
        "accuracy":    acc,
        "f1_weighted": f1,
        "report": classification_report(
            labels, preds,
            target_names=le.classes_,
            digits=4, output_dict=True, zero_division=0,
        ),
    }
    path = os.path.join(dirs["metrics"], "baseline_fcn_results.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"  Metrics saved -> {path}")
    return acc, f1


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BaselineFCN on LSST")
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=40)
    parser.add_argument("--val_split",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--results_dir", type=str,   default="results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")
    dirs = setup_dirs(args.results_dir)
    set_seed(args.seed)

    X_train, y_train, X_test, y_test, le = load_lsst()
    n_classes = len(le.classes_)
    seq_len   = X_train.shape[1]
    n_ch      = X_train.shape[2]
    print(f"  {n_ch} channels  |  {seq_len} timesteps  |  {n_classes} classes")

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    class_weights = compute_class_weights(y_train, device)

    model = BaselineFCN(n_channels=n_ch, n_classes=n_classes, seq_len=seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  BaselineFCN parameters : {n_params:,}")

    model, _ = train(
        model, train_loader, val_loader, device, dirs,
        class_weights=class_weights,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    acc, f1 = evaluate(model, test_loader, device, le, dirs)

    print(f"\n{'=' * 70}")
    print(f"  FINAL  Accuracy={acc:.4f}  F1={f1:.4f}")
    print(f"{'=' * 70}")
    print(f"\n  All outputs -> {args.results_dir}/")


if __name__ == "__main__":
    main()
