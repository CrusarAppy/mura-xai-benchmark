"""Classification performance and calibration metrics (numpy inputs)."""
from __future__ import annotations
from typing import Dict
import numpy as np


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold maximising Youden's J = sensitivity + specificity - 1
    (equivalently argmax(TPR - FPR) on the ROC curve). Returns 0.5 if undefined."""
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, thr = roc_curve(labels, scores)
    except ValueError:
        return 0.5
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    # roc_curve can emit an inf at the first point; clamp to a valid probability
    if not np.isfinite(t):
        return 0.5
    return float(min(max(t, 0.0), 1.0))


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """probs: (N,2) softmax outputs, labels: (N,). Positive class = 1 (abnormal).

    Reports threshold-free discrimination (AUROC, AUPRC) plus threshold-dependent
    metrics at BOTH the default 0.5 and the Youden-J optimum (suffix `_youden`),
    per proposal Section 3.9.1.
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score)
    p1 = probs[:, 1]
    preds = (p1 >= 0.5).astype(int)
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
    try:
        out["auprc"] = float(average_precision_score(labels, p1))
    except ValueError:
        out["auprc"] = float("nan")

    # Youden-J operating point
    thr = youden_threshold(labels, p1)
    preds_y = (p1 >= thr).astype(int)
    out["threshold_youden"] = float(thr)
    out["precision_youden"] = float(precision_score(labels, preds_y, zero_division=0))
    out["recall_youden"] = float(recall_score(labels, preds_y, zero_division=0))
    out["f1_youden"] = float(f1_score(labels, preds_y, zero_division=0))
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
