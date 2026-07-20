"""Classification performance and calibration metrics (numpy inputs)."""
from __future__ import annotations
from typing import Dict
import numpy as np


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """probs: (N,2) softmax outputs, labels: (N,). Positive class = 1 (abnormal)."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    preds = probs.argmax(axis=1)
    p1 = probs[:, 1]
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(labels, p1))
    except ValueError:
        out["auroc"] = float("nan")   # single-class batch
    return out


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """Expected Calibration Error (ECE) and Brier score. Lower is better for both."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

    p1 = probs[:, 1]
    brier = float(np.mean((p1 - labels) ** 2))
    return {"ece": float(ece), "brier": brier}
