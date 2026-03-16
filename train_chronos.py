"""
train_chronos.py
================
LSST Time Series Classification — Experiment B: Chronos Foundation Model.

Pipeline
--------
1. Load LSST (instance-normalised)
2. Extract patch-level embeddings from amazon/chronos-t5-small channel-by-channel
3. Multi-pooling feature engineering  (mean / max / std over time axis)
4. StandardScaler normalisation
5. Train SVC(kernel='rbf') probe classifier
6. Report per-class classification report and save JSON metrics

Usage
-----
    python train_chronos.py
    python train_chronos.py --results_dir experiments/run1
    python train_chronos.py --batch_size 256
"""

import argparse
import json
import os

import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_lsst


# =============================================================================
# Helpers
# =============================================================================

def setup_dirs(base: str = "results") -> dict:
    dirs = {
        "base":    base,
        "metrics": os.path.join(base, "metrics"),
        "logs":    os.path.join(base, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# =============================================================================
# Chronos embedding extraction
# =============================================================================

def load_chronos(device: torch.device) -> ChronosPipeline:
    """Load amazon/chronos-t5-small from HuggingFace Hub."""
    print("\nLoading amazon/chronos-t5-small ...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=str(device),
        torch_dtype=torch.float32,
    )
    pipeline.model.eval()
    return pipeline


def extract_embeddings(
    X: np.ndarray,
    pipeline: ChronosPipeline,
    batch_size: int = 128,
    label: str = "",
) -> torch.Tensor:
    """
    Extract Chronos patch embeddings for a multivariate time-series array.

    Parameters
    ----------
    X          : np.ndarray, shape (N, T, C)
    pipeline   : ChronosPipeline (chronos-t5-small)
    batch_size : inference mini-batch size

    Returns
    -------
    embeddings : torch.Tensor, shape (N, C, T', D)
                 T' = number of patches, D = model hidden size (512 for t5-small)
    """
    N, T, C = X.shape
    channel_embeddings = []

    for c in range(C):
        print(f"  {label}  channel {c + 1}/{C} ...")
        batches = []
        for start in range(0, N, batch_size):
            batch = torch.tensor(
                X[start : start + batch_size, :, c], dtype=torch.float32
            )  # (B, T)  — stays on CPU for the Chronos tokenizer
            with torch.no_grad():
                emb, _ = pipeline.embed(batch)   # (B, T', D)
            batches.append(emb.cpu())
        channel_embeddings.append(torch.cat(batches, dim=0))  # (N, T', D)

    # Stack over channels: (N, C, T', D)
    return torch.stack(channel_embeddings, dim=1)


# =============================================================================
# Feature engineering
# =============================================================================

def multi_pool_features(emb: torch.Tensor) -> np.ndarray:
    """
    Convert patch embeddings to a fixed-length feature vector via multi-pooling.

    Temporal statistics (mean, max, std) are computed over the patch axis and
    concatenated per channel, then flattened.

        (N, C, T', D) -> (N, C * 3 * D) = (N, C * 1536) for t5-small

    Using three pooling operators captures different aspects of the temporal
    distribution without requiring sequence-length alignment.
    """
    mean_pool = emb.mean(dim=2)         # (N, C, D)
    max_pool, _ = emb.max(dim=2)        # (N, C, D)
    std_pool  = emb.std(dim=2)          # (N, C, D)

    combined = torch.cat([mean_pool, max_pool, std_pool], dim=2)  # (N, C, 3D)
    return combined.reshape(combined.shape[0], -1).numpy()        # (N, C * 3D)


# =============================================================================
# Classifier
# =============================================================================

def train_svm(X_train, y_train):
    print("  Training SVC (RBF kernel) ...")
    clf = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_clf(clf, X_test, y_test, le, label: str, dirs: dict):
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")
    report_str = classification_report(
        y_test, preds,
        target_names=[str(c) for c in le.classes_],
        digits=4, zero_division=0,
    )

    print(f"\n  {'=' * 60}")
    print(f"  {label}")
    print(f"  {'=' * 60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(f"\n{report_str}")

    result = {
        "model":       label,
        "accuracy":    acc,
        "f1_weighted": f1,
        "report": classification_report(
            y_test, preds,
            target_names=[str(c) for c in le.classes_],
            digits=4, output_dict=True, zero_division=0,
        ),
    }
    fname = label.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_results.json"
    path  = os.path.join(dirs["metrics"], fname)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"  Metrics saved -> {path}")
    return acc, f1


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chronos embedding + ML probe on LSST"
    )
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device : {device}")
    dirs = setup_dirs(args.results_dir)

    # ── Data ─────────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, le = load_lsst(normalise=True)

    # ── Chronos embeddings ───────────────────────────────────────────────────
    pipeline = load_chronos(device)

    print("\nExtracting embeddings for training set ...")
    train_emb = extract_embeddings(
        X_train, pipeline, batch_size=args.batch_size, label="[train]"
    )
    print("Extracting embeddings for test set ...")
    test_emb = extract_embeddings(
        X_test, pipeline, batch_size=args.batch_size, label="[test]"
    )

    print(f"\n  Embedding shape (train) : {train_emb.shape}")   # (N, C, T', D)

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\nBuilding multi-pool feature vectors ...")
    X_train_feat = multi_pool_features(train_emb)
    X_test_feat  = multi_pool_features(test_emb)
    print(f"  Feature dimension : {X_train_feat.shape[1]:,}")

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    # Save scaler params for reproducibility
    scaler_info = {
        "mean_shape": list(scaler.mean_.shape),
        "feature_dim": int(X_train_feat.shape[1]),
    }
    with open(os.path.join(dirs["logs"], "chronos_scaler_info.json"), "w") as fh:
        json.dump(scaler_info, fh)

    # ── Classifiers ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING CLASSIFIER")
    print("=" * 70)

    svm = train_svm(X_train_feat, y_train)
    acc_svm, f1_svm = evaluate_clf(
        svm, X_test_feat, y_test, le, "Chronos + SVM", dirs
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Chronos + SVM  |  Accuracy={acc_svm:.4f}  F1={f1_svm:.4f}")
    print(f"{'=' * 70}")
    print(f"\n  All outputs -> {args.results_dir}/")


if __name__ == "__main__":
    main()
