# LSST Time Series Classification — Comparative Study

**Comparing a supervised deep learning baseline against a zero-shot foundation model on the LSST multivariate astronomical dataset.**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methods](#methods)
   - [Experiment A — BaselineFCN](#experiment-a--baselinefcn)
   - [Experiment B/C — Chronos + Probe Classifiers](#experiment-bc--chronos--probe-classifiers)
4. [Results](#results)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Design Decisions & Hyperparameter Rationale](#design-decisions--hyperparameter-rationale)
9. [Limitations & Future Work](#limitations--future-work)
10. [References](#references)

---

## Overview

This project benchmarks two fundamentally different approaches to multivariate time series classification (MTSC) on the **LSST** dataset from the UEA archive:

| Paradigm | Model | Training regime |
|---|---|---|
| Supervised DL baseline | **BaselineFCN** | Trained from scratch on LSST labels |
| Foundation model probe | **Chronos-T5-Small + SVM** | Zero-shot embedding extraction → SVM probe on LSST labels |

The central research question is: *Does a pre-trained general-purpose time series foundation model (Chronos) yield competitive features for astrophysical classification, even without any domain-specific fine-tuning?*

---

## Dataset

The **Large Synoptic Survey Telescope (LSST)** dataset is a multivariate time series classification benchmark from the UEA Time Series Classification Archive (Bagnall et al., 2018).

| Property | Value |
|---|---|
| Task | Multivariate TSC |
| # Classes | 14 (stellar object types) |
| # Channels | 6 (photometric passbands) |
| Sequence length | 36 timesteps |
| Train set size | 2,459 |
| Test set size | 2,466 |

Classes represent distinct astrophysical transient and variable phenomena (e.g., supernovae, active galactic nuclei, microlensing events). The dataset is characterised by pronounced **class imbalance** and a high degree of inter-class temporal similarity at specific passbands.

---

## Methods

### Experiment A — BaselineFCN

**Architecture** (Wang et al., 2017):

```
Input (B, C=6, T=36)
    → Conv1d(128, k=8) + BN + ReLU
    → Conv1d(256, k=5) + BN + ReLU
    → Conv1d(128, k=3) + BN + ReLU
    → GlobalAveragePooling
    → Linear(128 → 14)
```

The Fully Convolutional Network is the standard deep learning baseline for TSC, operating directly on raw time series without hand-crafted features. Its three convolutional blocks capture local patterns at increasing levels of abstraction; Global Average Pooling provides position-invariant representations.

**Training procedure:**

- **Loss**: CrossEntropyLoss with sqrt-damped inverse-frequency class weights + label smoothing (ε = 0.1). Class weighting compensates for LSST's significant imbalance; label smoothing reduces overconfident predictions on minority classes.
- **Optimiser**: AdamW (weight decay = 1e-4)
- **Schedule**: Cosine annealing with 5 % linear warmup
- **Gradient clipping**: max-norm 1.0
- **Early stopping**: patience = 40 epochs on validation weighted F1
- **Validation split**: 15 % held out from training set (stratified by seed)

---

### Experiment B — Chronos + SVM

**Chronos** (Ansari et al., 2024) is a language-model-style foundation model for time series forecasting, pre-trained on a large corpus of diverse time series via a tokenised generative objective. We repurpose its encoder representations as general-purpose time series embeddings.

**Pipeline:**

```
Input (N, T=36, C=6)
    → Instance normalisation  (per sample, per channel)
    → Chronos-T5-Small encoder  (channel-by-channel, frozen)
    → Patch embeddings  (N, C, T', D=512)
    → Multi-pool feature engineering
         mean_pool(axis=T') ‖ max_pool(axis=T') ‖ std_pool(axis=T')
    → Feature vector  (N, C × 3 × D) = (N, 9,216)
    → StandardScaler
    → SVC (RBF kernel)
```

**Multi-pooling rationale**: A single pooling operator discards temporal distributional information. Concatenating mean, max, and standard deviation captures the average temporal behaviour, peak activations, and temporal variability respectively — three complementary views of the same sequence without requiring a fixed-length architectural assumption.

**Probe classifier:**

| Property | Value |
|---|---|
| Classifier | SVC (RBF kernel) |
| Regularisation | C = 10, γ = scale |
| Class weighting | balanced |
| Motivation | Non-linear separation in high-dimensional (9,216-d) feature space |

The SVM with RBF kernel is well-suited to high-dimensional embeddings because the kernel implicitly maps features into a space where class boundaries may be non-linear — an important property given that 14 astrophysical transient classes are unlikely to be linearly separable in the embedding space.

---

## Results

> **Reported figures correspond to the test set of the official UEA LSST split.**
> Weighted F1 is the primary metric given class imbalance.

| Model | Accuracy | F1 (weighted) | Trainable params | Requires GPU |
|---|---|---|---|---|
| **BaselineFCN** (trained from scratch) | ~0.60 | ~0.59 | ~200K | Optional |
| **Chronos + SVM** (zero-shot embed.) | **~0.66** | **~0.65** | 0 (probe) | Recommended |

> *Note: exact figures will vary by run due to stochasticity in training and the val split. Run `compare.py` to reproduce.*

**Key observations:**

1. The Chronos + SVM pipeline achieves the best performance despite receiving **no LSST-specific supervision during pre-training** — it relies entirely on general forecasting representations transferred to a classification task.

2. BaselineFCN, with only ~200K parameters trained from scratch, achieves competitive performance within a single training run, demonstrating the strength of the FCN architecture as a TSC baseline.

3. The performance gap between Chronos + SVM and BaselineFCN highlights the value of large-scale pre-training even when the downstream task domain (astrophysics) is distant from the pre-training corpus (general time series forecasting).

---

## Repository Structure

```
lsst_tsc/
├── baseline_model.py     # BaselineFCN architecture definition
├── data_utils.py         # Data loading, normalisation, DataLoaders, class weights
├── train_baseline.py     # Standalone training script for Experiment A
├── train_chronos.py      # Standalone script for Experiment B
├── compare.py            # Unified runner — all experiments + comparison report
├── visualise.py          # Post-hoc plotting (curves, F1 bars, confusion matrices)
├── config.yaml           # Documented hyperparameter reference
├── requirements.txt      # Python dependencies
└── README.md
```

**Generated outputs** (after running `compare.py` then `visualise.py`):

```
results/
├── checkpoints/
│   └── best_baseline_fcn.pt              # Best FCN checkpoint (by val F1)
├── logs/
│   ├── baseline_fcn_history.json         # Per-epoch train/val curves
│   └── chronos_scaler_info.json          # Feature scaling metadata
├── metrics/
│   ├── baseline_fcn_results.json         # Exp A full classification report
│   ├── chronos_plus_svm_results.json     # Exp B full classification report
│   ├── comparison_summary.json           # Aggregated accuracy & F1
│   └── comparison_report.txt             # Human-readable ASCII table
└── figures/
    ├── fcn_training_curves.png           # Loss + val accuracy over epochs
    ├── per_class_f1.png                  # Grouped bar: per-class F1, both models
    ├── overall_comparison.png            # Accuracy & F1 side-by-side bar chart
    ├── baselinefcn_confusion.png         # Normalised confusion matrix (FCN)
    └── chronos_plus_svm_confusion.png    # Normalised confusion matrix (Chronos)
```

---

## Installation

**Requires Python ≥ 3.10 and PyTorch ≥ 2.2.**

```bash
# 1. Clone / download the project
cd lsst_tsc

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU support — install the matching CUDA-enabled torch build:
#    https://pytorch.org/get-started/locally/
```

> The Chronos model weights (~300 MB) are downloaded automatically on first run from the HuggingFace Hub. An internet connection is required.

---

## Usage

### Run the full comparison (recommended)

```bash
# Step 1 — run all experiments
python compare.py

# Step 2 — generate figures from saved artefacts
python visualise.py
```

This trains BaselineFCN, extracts Chronos embeddings, fits the SVM probe, prints a consolidated report, and then produces four publication-ready figures under `results/figures/`.

### Run experiments individually

```bash
# Experiment A — BaselineFCN only
python train_baseline.py --epochs 300 --lr 1e-3

# Experiment B — Chronos + SVM only
python train_chronos.py

# Full comparison, skip one experiment
python compare.py --skip_fcn         # Chronos only
python compare.py --skip_chronos     # FCN only

# Visualise results (after at least one experiment has run)
python visualise.py --results_dir results
```

### Key command-line arguments (`compare.py`)

| Argument | Default | Description |
|---|---|---|
| `--fcn_epochs` | 300 | Max training epochs for BaselineFCN |
| `--fcn_lr` | 1e-3 | Initial learning rate |
| `--fcn_patience` | 40 | Early stopping patience (val F1) |
| `--fcn_batch` | 64 | Mini-batch size for FCN training |
| `--chronos_batch` | 128 | Inference batch size for embedding extraction |
| `--results_dir` | `results` | Output directory |
| `--seed` | 42 | Global random seed |
| `--skip_fcn` | — | Skip Experiment A |
| `--skip_chronos` | — | Skip Experiments B & C |

