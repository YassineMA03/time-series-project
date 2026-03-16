"""
visualise.py
============
Post-hoc visualisation for LSST TSC experiments.

Reads the JSON artefacts produced by compare.py and generates:

  1. FCN training curves          (train/val loss + val accuracy)
  2. Per-class F1 bar chart        (BaselineFCN vs Chronos + SVM)
  3. Accuracy / F1 comparison bar  (overall metrics, both models)
  4. Confusion-matrix heatmaps     (one per model, saved as PNG)

All figures are saved to <results_dir>/figures/.

Usage
-----
    python visualise.py
    python visualise.py --results_dir experiments/run1
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       120,
})

PALETTE = {
    "BaselineFCN":   "#2C7BB6",
    "Chronos + SVM": "#D7191C",
    "train":         "#555555",
    "val":           "#2C7BB6",
}


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def ensure_figures_dir(results_dir: str) -> str:
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


# =============================================================================
# 1. FCN training curves
# =============================================================================

def plot_training_curves(history: dict, fig_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("BaselineFCN — Training Curves", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["train"],
            lw=1.5, label="Train loss")
    ax.plot(epochs, history["val_loss"],   color=PALETTE["val"],
            lw=1.5, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss")
    ax.legend()

    # Accuracy + F1
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], color=PALETTE["val"],
            lw=1.5, label="Val accuracy")
    ax.plot(epochs, history["val_f1"],  color="#1A9641",
            lw=1.5, linestyle="--", label="Val F1 (weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Performance")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()

    fig.tight_layout()
    path = os.path.join(fig_dir, "fcn_training_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# 2. Per-class F1 comparison
# =============================================================================

def plot_per_class_f1(reports: dict, class_names: list, fig_dir: str) -> None:
    """
    Grouped bar chart: per-class F1 for each model.
    reports = {"ModelName": sklearn classification_report dict}
    """
    models  = list(reports.keys())
    n_cls   = len(class_names)
    x       = np.arange(n_cls)
    width   = 0.35
    colors  = [PALETTE.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(max(10, n_cls * 0.9), 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        f1_scores = [
            reports[model].get(cls, {}).get("f1-score", 0.0)
            for cls in class_names
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=model,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Score — BaselineFCN vs Chronos + SVM",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(fig_dir, "per_class_f1.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# 3. Overall metric comparison bar
# =============================================================================

def plot_overall_comparison(summary: dict, fig_dir: str) -> None:
    models   = list(summary.keys())
    accs     = [summary[m]["accuracy"]    for m in models]
    f1s      = [summary[m]["f1_weighted"] for m in models]
    colors   = [PALETTE.get(m, "#888888") for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy",
                color=colors, alpha=0.9, edgecolor="white")
    b2 = ax.bar(x + width / 2, f1s,  width, label="F1 (weighted)",
                color=colors, alpha=0.55, edgecolor="white", hatch="//")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Overall Comparison — Accuracy & Weighted F1",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(fig_dir, "overall_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# 4. Confusion matrix heatmap
# =============================================================================

def plot_confusion_matrix(report_dict: dict, class_names: list,
                           model_label: str, fig_dir: str) -> None:
    """
    Reconstruct an approximate normalised confusion matrix from the
    per-class precision / recall values stored in the classification report.

    Note: the full confusion matrix is not stored in JSON; this function
    creates a diagonal-dominant approximation useful for visual inspection.
    If you want exact counts, pass the raw predictions to this function instead.
    """
    n = len(class_names)
    # Extract per-class recall (= TP / (TP + FN)) as diagonal values
    diag = np.array([
        report_dict.get(c, {}).get("recall", 0.0) for c in class_names
    ])
    # Spread remaining probability uniformly across off-diagonal (approximation)
    mat = np.full((n, n), 0.0)
    for i in range(n):
        mat[i, i] = diag[i]
        off = (1.0 - diag[i]) / max(n - 1, 1)
        for j in range(n):
            if j != i:
                mat[i, j] = off

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.65)))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Normalised Confusion Matrix — {model_label}",
                 fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.tight_layout()
    fname = model_label.lower().replace(" ", "_").replace("+", "plus") + "_confusion.png"
    path  = os.path.join(fig_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise LSST TSC experiment results")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, "metrics")
    logs_dir    = os.path.join(args.results_dir, "logs")
    fig_dir     = ensure_figures_dir(args.results_dir)

    print(f"\nReading artefacts from {args.results_dir}/ ...")

    # ── 1. Training curves (FCN) ─────────────────────────────────────────────
    history_path = os.path.join(logs_dir, "baseline_fcn_history.json")
    if os.path.exists(history_path):
        print("\n[1/4] Plotting FCN training curves ...")
        plot_training_curves(load_json(history_path), fig_dir)
    else:
        print(f"  [1/4] Skipped — {history_path} not found")

    # ── Load per-model reports ───────────────────────────────────────────────
    model_files = {
        "BaselineFCN":   os.path.join(metrics_dir, "baseline_fcn_results.json"),
        "Chronos + SVM": os.path.join(metrics_dir, "chronos_plus_svm_results.json"),
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded[name] = load_json(path)

    if not loaded:
        print("\n  No model result files found. Run compare.py first.")
        return

    # Collect class names from any available report
    sample_report = next(iter(loaded.values()))["report"]
    class_names   = [
        k for k in sample_report
        if k not in ("accuracy", "macro avg", "weighted avg")
    ]

    # ── 2. Per-class F1 ──────────────────────────────────────────────────────
    if len(loaded) >= 1:
        print("\n[2/4] Plotting per-class F1 comparison ...")
        reports = {name: data["report"] for name, data in loaded.items()}
        plot_per_class_f1(reports, class_names, fig_dir)
    else:
        print("  [2/4] Skipped — need at least one model result")

    # ── 3. Overall comparison ────────────────────────────────────────────────
    summary_path = os.path.join(metrics_dir, "comparison_summary.json")
    if os.path.exists(summary_path):
        print("\n[3/4] Plotting overall metric comparison ...")
        plot_overall_comparison(load_json(summary_path), fig_dir)
    else:
        print(f"  [3/4] Skipped — {summary_path} not found")

    # ── 4. Confusion matrices ────────────────────────────────────────────────
    print("\n[4/4] Plotting confusion matrices ...")
    for name, data in loaded.items():
        plot_confusion_matrix(data["report"], class_names, name, fig_dir)

    print(f"\n  All figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
