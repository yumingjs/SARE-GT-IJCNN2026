"""Shared utilities for SARE-GT training and evaluation.

Provides reproducibility helpers, logging setup, the warmup-cosine learning
rate scheduler, and standard evaluation metrics.
"""

import math
import os
import time
import random
import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, cohen_kappa_score,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(experiment_name: str = "sare_gt", log_dir: str = "logs"):
    """Create a logger that writes to both console and a timestamped file.

    Args:
        experiment_name: Prefix for the log file name.
        log_dir: Directory to store log files.

    Returns:
        tuple: ``(logger, log_file_path)``
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Warmup followed by cosine-annealing learning rate schedule.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs (int): Number of linear warmup epochs.
        total_epochs (int): Total number of training epochs.
        base_lr (float): Peak learning rate (reached at end of warmup).
        min_lr (float): Minimum learning rate.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        """Update the learning rate and return its current value."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = ((epoch - self.warmup_epochs)
                        / (self.total_epochs - self.warmup_epochs))
            lr = (self.min_lr
                  + (self.base_lr - self.min_lr)
                  * 0.5 * (1 + math.cos(math.pi * progress)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, num_classes: int = 9):
    """Compute classification metrics including QWK.

    Args:
        y_true: Ground-truth labels (array-like).
        y_pred: Predicted labels (array-like).
        num_classes: Number of ordinal rating classes.

    Returns:
        dict with keys ``accuracy``, ``recall_macro``, ``f1_macro``, ``qwk``.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(
            y_true, y_pred, weights="quadratic",
            labels=list(range(num_classes)),
        ),
    }
