"""
data_utils.py
=============
Shared data-loading and preprocessing utilities for LSST TSC experiments.

Provides:
  - load_lsst()          : Download/cache LSST via tslearn, encode labels
  - instance_norm()      : Per-instance z-score normalisation
  - get_dataloaders()    : Train/val/test DataLoaders (Conv1d-ready)
  - compute_class_weights: Sqrt-damped inverse-frequency weights
"""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tslearn.datasets import UCR_UEA_datasets


# =============================================================================
# Normalisation
# =============================================================================

def instance_norm(X: np.ndarray) -> np.ndarray:
    """
    Per-instance z-score normalisation along the time axis.
    Shape: (N, T, C)  ->  (N, T, C)
    Each sample is centred/scaled independently (astrophysical flux variability
    spans orders of magnitude across object classes).
    """
    mean = X.mean(axis=1, keepdims=True)   # (N, 1, C)
    std  = X.std(axis=1, keepdims=True) + 1e-5
    return (X - mean) / std


# =============================================================================
# Dataset loading
# =============================================================================

def load_lsst(normalise: bool = True):
    """
    Load the LSST multivariate time-series dataset from the UEA archive.

    Parameters
    ----------
    normalise : bool
        Apply instance-wise z-score normalisation (recommended).

    Returns
    -------
    X_train : np.ndarray, shape (N_train, T, C)
    y_train : np.ndarray, shape (N_train,)   — integer labels
    X_test  : np.ndarray, shape (N_test,  T, C)
    y_test  : np.ndarray, shape (N_test,)
    le      : sklearn.LabelEncoder  — maps integers back to class names
    """
    print("Loading LSST dataset from UEA archive via tslearn ...")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    # Convert masked arrays / ensure float32
    X_train = np.nan_to_num(np.array(X_train, dtype=np.float32))
    X_test  = np.nan_to_num(np.array(X_test,  dtype=np.float32))

    if normalise:
        X_train = instance_norm(X_train)
        X_test  = instance_norm(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    print(
        f"  Train : {X_train.shape}  |  Test : {X_test.shape}  |  "
        f"Classes : {len(le.classes_)}"
    )
    return X_train, y_train, X_test, y_test, le


# =============================================================================
# DataLoaders  (for PyTorch models — transposes to (B, C, T))
# =============================================================================

def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    batch_size: int = 64,
    val_split:  float = 0.15,
    seed: int = 42,
):
    """
    Create train / validation / test DataLoaders.

    Input arrays are expected in (B, T, C) format (tslearn convention).
    Tensors are transposed to (B, C, T) as required by PyTorch Conv1d.

    Parameters
    ----------
    val_split : fraction of training data used for validation

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_train))
    n_val = max(1, int(len(X_train) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def make_loader(X, y, shuffle: bool) -> DataLoader:
        # (B, T, C) -> (B, C, T)
        Xt = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        yt = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(X_train[train_idx], y_train[train_idx], shuffle=True)
    val_loader   = make_loader(X_train[val_idx],   y_train[val_idx],   shuffle=False)
    test_loader  = make_loader(X_test,             y_test,             shuffle=False)

    print(
        f"  DataLoaders  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Class weights
# =============================================================================

def compute_class_weights(
    y_train: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Sqrt-damped inverse-frequency class weights.

    Full inverse frequency over-emphasises rare classes; square-root damping
    provides a softer trade-off between majority and minority performance.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts / counts.sum()
    w    = 1.0 / np.sqrt(freq)
    w    = w / w.sum() * len(classes)   # normalise so mean == 1
    return torch.tensor(w, dtype=torch.float32).to(device)
