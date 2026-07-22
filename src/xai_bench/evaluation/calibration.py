"""Calibration analysis: reliability-diagram data and post-hoc temperature scaling.

Complements `metrics.calibration_metrics` (ECE/Brier). Proposal Section 3.9.2 commits to
reliability diagrams and to examining the effect of temperature scaling on calibration.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


def _ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> List[Dict]:
    """Per-bin (confidence, accuracy, count) for plotting a reliability diagram."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "count": cnt,
            "confidence": float(conf[mask].mean()) if cnt else float("nan"),
            "accuracy": float(correct[mask].mean()) if cnt else float("nan"),
        })
    return rows


def temperature_scale(logits: np.ndarray, labels: np.ndarray,
                      max_iter: int = 100, n_bins: int = 15) -> Dict[str, float]:
    """Fit a single temperature T>0 by minimising NLL on (logits, labels), then report
    ECE before and after scaling. Returns {temperature, ece_before, ece_after}.

    logits: (N,2) pre-softmax scores. Pure-numpy 1-D search (no torch dependency here).
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _nll(T):
        p = _softmax(logits / T)
        p_true = p[np.arange(len(labels)), labels]
        return -np.mean(np.log(np.clip(p_true, 1e-12, 1.0)))

    # coarse-to-fine 1-D search over T in [0.05, 10]
    grid = np.linspace(0.05, 10.0, 200)
    best_T = float(grid[int(np.argmin([_nll(T) for T in grid]))])
    lo, hi = max(0.05, best_T - 0.1), best_T + 0.1
    fine = np.linspace(lo, hi, 200)
    best_T = float(fine[int(np.argmin([_nll(T) for T in fine]))])

    def _ece(p):
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == labels).astype(np.float64)
        return _ece_from_conf(conf, correct, n_bins)

    ece_before = _ece(_softmax(logits))
    ece_after = _ece(_softmax(logits / best_T))
    return {"temperature": best_T, "ece_before": ece_before, "ece_after": ece_after}
