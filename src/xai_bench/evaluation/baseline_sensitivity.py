"""Baseline-sensitivity diagnostic for attribution methods (proposal Section 3.7).

Radiograph attributions depend on the reference baseline. This measures how much an
attribution method's map moves when the baseline is switched between zero / blur / mean,
on a small subset of images. Higher agreement -> more baseline-robust.

Intended as a dedicated subset pass, NOT part of the per-config sweep row, because it
regenerates each attribution three times.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else float("nan")


def baseline_sensitivity(build_method, model, target_layer, x,
                         baselines: List[str] = ("zero", "blur", "mean"),
                         target_class: int = 1) -> Dict[str, float]:
    """Generate the attribution for `x` under each baseline and report the mean pairwise
    Spearman agreement between the resulting maps. `build_method(model, target_layer,
    baseline=...)` must return a callable explainer.

    Returns {baseline_sensitivity_corr, n_pairs}. corr near 1 = baseline-robust.
    """
    maps = []
    for b in baselines:
        ex = build_method(model, target_layer, baseline=b)
        maps.append(np.asarray(ex(x, target_class=target_class)))
        if hasattr(ex, "remove"):
            ex.remove()
    corrs = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            corrs.append(_spearman(maps[i], maps[j]))
    corrs = [c for c in corrs if np.isfinite(c)]
    return {
        "baseline_sensitivity_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "n_pairs": len(corrs),
    }
