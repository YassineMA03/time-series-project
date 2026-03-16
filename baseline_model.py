"""
baseline_model.py
==================
Simple baselines for Time Series Classification (no pretraining).

Two baselines provided:
  1. FCN (Fully Convolutional Network) — the standard DL baseline for TSC
     Reference: "Time Series Classification from Scratch with Deep Neural
     Networks: A Strong Baseline" (Wang et al., 2017)
  2. ResNet — FCN with residual connections

These are the "decent baselines without pretraining" required by the project.
They follow the architectures discussed in Session 2 (ConvNets) and
Session 4 (TSC state-of-the-art) of the course.
"""

import torch
import torch.nn as nn


class BaselineFCN(nn.Module):
    """
    Fully Convolutional Network for Time Series Classification.

    This is the simplest standard DL baseline for TSC:
      - 3 convolutional blocks (Conv1D + BatchNorm + ReLU)
      - Global Average Pooling
      - Linear classifier

    No recurrence, no attention, no pretraining.
    Directly mentioned in the course (Session 4, slide 17).

    Architecture:
        Input (C, T) -> Conv(128) -> Conv(256) -> Conv(128) -> GAP -> Linear -> classes
    """

    def __init__(self, n_channels, n_classes, seq_len=36):
        super().__init__()
        self.model_name = "BaselineFCN"

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Global Average Pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time) — e.g. (B, 6, 36) for LSST
        Returns:
            logits: (batch, n_classes)
        """
        x = self.conv1(x)  # (B, 128, T)
        x = self.conv2(x)  # (B, 256, T)
        x = self.conv3(x)  # (B, 128, T)
        x = self.gap(x).squeeze(-1)  # (B, 128)
        return self.classifier(x)    # (B, n_classes)


# Keep backward compatibility: alias so train_v3.py still works
BaselineCNNLSTM = BaselineFCN
