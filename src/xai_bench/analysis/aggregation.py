"""Multi-criteria aggregation and ranking (proposal Section 3.8.1).

Turns a per-configuration metric table into a defensible ranking via:
  min-max normalisation (direction-aware) -> Pareto non-dominance -> TOPSIS + weighted-sum
  + Borda -> Kendall's tau agreement between schemes -> weight-sensitivity analysis.

Input `df` has one row per alternative (e.g. an XAI method, or a method x backbone cell)
and one column per metric. Metric direction (higher/lower is better) is taken from
METRIC_DIRECTIONS unless overridden.
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

# True = higher is better, False = lower is better.
METRIC_DIRECTIONS: Dict[str, bool] = {
    # performance / calibration (model level)
    "accuracy": True, "precision": True, "recall": True, "f1": True,
    "auroc": True, "auprc": True,
    "ece": False, "brier": False, "ece_temp_scaled": False,
    # faithfulness
    "insertion_auc": True, "insertion_auc_mean": True,
    "increase_in_confidence": True,
    "deletion_auc": False, "deletion_auc_mean": False, "average_drop": False,
    # efficiency
    "runtime_s_per_explanation": False, "gpu_mem_mb_per_explanation": False,
    # robustness / agreement (higher stability/overlap = better)
    "robust_ssim_noise_lo": True, "robust_ssim_noise_hi": True,
    "robust_spearman_noise_lo": True, "robust_spearman_noise_hi": True,
    "iou_mean": True, "dice_mean": True, "ssim_mean": True, "spearman_mean": True,
}


def _directions(cols: Sequence[str], overrides: Dict[str, bool] | None) -> List[bool]:
    d = dict(METRIC_DIRECTIONS)
    if overrides:
        d.update(overrides)
    missing = [c for c in cols if c not in d]
    if missing:
        raise KeyError(f"no direction known for {missing}; pass `directions=` overrides")
    return [d[c] for c in cols]


def normalize(df: pd.DataFrame, cols: Sequence[str],
              directions: Dict[str, bool] | None = None) -> pd.DataFrame:
    """Min-max each column to [0,1] with lower-is-better columns inverted, so that in the
    output higher always means better. Constant columns map to 0.5."""
    dirs = _directions(cols, directions)
    out = pd.DataFrame(index=df.index)
    for c, higher_better in zip(cols, dirs):
        x = df[c].astype(float).to_numpy()
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi - lo < 1e-12:
            out[c] = 0.5
            continue
        z = (x - lo) / (hi - lo)
        out[c] = z if higher_better else (1.0 - z)
    return out


def pareto_front(df: pd.DataFrame, cols: Sequence[str],
                 directions: Dict[str, bool] | None = None) -> np.ndarray:
    """Boolean mask of non-dominated rows on the benefit-normalised matrix."""
    M = normalize(df, cols, directions).to_numpy()
    n = len(M)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                nd[i] = False
                break
    return nd


def topsis(df: pd.DataFrame, cols: Sequence[str], weights: Sequence[float] | None = None,
           directions: Dict[str, bool] | None = None) -> np.ndarray:
    """TOPSIS closeness-to-ideal in [0,1] (higher = better). Operates on the benefit-
    normalised matrix, then applies vector normalisation and weights."""
    B = normalize(df, cols, directions).to_numpy()
    w = np.ones(len(cols)) / len(cols) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    denom = np.sqrt((B ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    V = (B / denom) * w
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    denom2 = d_best + d_worst
    denom2[denom2 == 0] = 1.0
    return d_worst / denom2


def weighted_sum(df: pd.DataFrame, cols: Sequence[str], weights: Sequence[float] | None = None,
                 directions: Dict[str, bool] | None = None) -> np.ndarray:
    B = normalize(df, cols, directions).to_numpy()
    w = np.ones(len(cols)) / len(cols) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return (B * w).sum(axis=1)


def borda(df: pd.DataFrame, cols: Sequence[str],
          directions: Dict[str, bool] | None = None) -> np.ndarray:
    """Borda points: per criterion, rank alternatives (best gets m-1), then sum."""
    B = normalize(df, cols, directions).to_numpy()
    m = B.shape[0]
    points = np.zeros(m)
    for j in range(B.shape[1]):
        order = np.argsort(B[:, j])          # ascending; worst first
        pts = np.empty(m)
        pts[order] = np.arange(m)            # best gets m-1
        points += pts
    return points


def kendall_tau_between(rankings: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """Pairwise Kendall's tau between ranking schemes (each a score vector; higher=better)."""
    from scipy.stats import kendalltau
    names = list(rankings)
    T = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            tau, _ = kendalltau(rankings[a], rankings[b])
            T.loc[a, b] = float(tau)
    return T


def rank_methods(df: pd.DataFrame, cols: Sequence[str], id_col: str = "xai_method",
                 weights: Sequence[float] | None = None,
                 directions: Dict[str, bool] | None = None) -> pd.DataFrame:
    """Full ranking table: TOPSIS, weighted-sum, Borda scores + ranks + Pareto flag,
    plus the Kendall's tau agreement between the three schemes (attached as .attrs)."""
    ids = df[id_col].tolist()
    top = topsis(df, cols, weights, directions)
    ws = weighted_sum(df, cols, weights, directions)
    bd = borda(df, cols, directions)
    pf = pareto_front(df, cols, directions)
    out = pd.DataFrame({
        id_col: ids,
        "topsis": top, "topsis_rank": (-top).argsort().argsort() + 1,
        "weighted_sum": ws, "weighted_sum_rank": (-ws).argsort().argsort() + 1,
        "borda": bd, "borda_rank": (-bd).argsort().argsort() + 1,
        "pareto_optimal": pf,
    }).sort_values("topsis", ascending=False).reset_index(drop=True)
    out.attrs["kendall_tau"] = kendall_tau_between(
        {"topsis": top, "weighted_sum": ws, "borda": bd})
    return out


def weight_sensitivity(df: pd.DataFrame, cols: Sequence[str],
                       weight_schemes: Dict[str, Sequence[float]], id_col: str = "xai_method",
                       directions: Dict[str, bool] | None = None) -> pd.DataFrame:
    """TOPSIS rank of each alternative under several weight schemes, plus rank stability
    (mean rank and rank standard deviation across schemes)."""
    ids = df[id_col].tolist()
    table = pd.DataFrame({id_col: ids})
    rank_cols = []
    for name, w in weight_schemes.items():
        sc = topsis(df, cols, w, directions)
        rank = (-sc).argsort().argsort() + 1
        table[f"rank_{name}"] = rank
        rank_cols.append(f"rank_{name}")
    table["rank_mean"] = table[rank_cols].mean(axis=1)
    table["rank_std"] = table[rank_cols].std(axis=1, ddof=0)
    return table.sort_values("rank_mean").reset_index(drop=True)
