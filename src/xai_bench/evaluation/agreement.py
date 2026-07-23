"""Explanation-agreement metrics (proposal Section 3.9.7).

Quantifies how much two saliency maps agree, using region overlap on energy-thresholded
masks (IoU, Dice) and correlation on the continuous maps (SSIM, Spearman). A driver
computes pairwise agreement across explainability methods (fixed backbone) and across
architectures (fixed method).

All inputs are numpy saliency maps in [0, 1], shape (H, W) or (N, H, W).
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


def binarise_topk(sal: np.ndarray, k_percent: float = 20.0) -> np.ndarray:
    """Energy-based percentile threshold: keep the smallest set of pixels whose summed
    saliency reaches k% of the total mass. Returns a 0/1 mask of the same shape.

    Chosen over fixed absolute thresholds (saliency scales differ across methods) and
    Otsu (saliency histograms are typically unimodal), per proposal Section 3.9.7.
    """
    single = sal.ndim == 2
    arr = sal[None] if single else sal
    n = arr.shape[0]
    flat = arr.reshape(n, -1).astype(np.float64)
    flat = np.clip(flat, 0, None)
    masks = np.zeros_like(flat)
    frac = float(k_percent) / 100.0
    for i in range(n):
        row = flat[i]
        order = np.argsort(row)[::-1]
        csum = np.cumsum(row[order])
        total = csum[-1]
        if total <= 0:
            continue
        cutoff = int(np.searchsorted(csum, frac * total)) + 1
        masks[i, order[:cutoff]] = 1.0
    masks = masks.reshape(arr.shape)
    return masks[0] if single else masks


def iou(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    a = a_mask.astype(bool); b = b_mask.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def dice(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    a = a_mask.astype(bool); b = b_mask.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else float("nan")


def ssim_map(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    try:
        return float(structural_similarity(a.astype(np.float64), b.astype(np.float64), data_range=1.0))
    except Exception:
        return float("nan")


def spearman_map(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else float("nan")


def agreement_pair(a: np.ndarray, b: np.ndarray, k_percent: float = 20.0) -> Dict[str, float]:
    """All four agreement metrics for one pair of (H,W) maps."""
    ma, mb = binarise_topk(a, k_percent), binarise_topk(b, k_percent)
    return {"iou": iou(ma, mb), "dice": dice(ma, mb),
            "ssim": ssim_map(a, b), "spearman": spearman_map(a, b)}


def pairwise_agreement(maps: Dict[str, np.ndarray], k_percent: float = 20.0) -> List[Dict]:
    """maps: {name: (N,H,W)}. For every unordered pair of names, average each agreement
    metric over the N images. Returns one row per pair.
    """
    names = list(maps)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_stack, b_stack = maps[names[i]], maps[names[j]]
            n = min(a_stack.shape[0], b_stack.shape[0])
            acc = {"iou": [], "dice": [], "ssim": [], "spearman": []}
            for t in range(n):
                r = agreement_pair(a_stack[t], b_stack[t], k_percent)
                for kk in acc:
                    acc[kk].append(r[kk])
            row = {"a": names[i], "b": names[j], "n": n, "k_percent": k_percent}
            for kk, v in acc.items():
                vv = [x for x in v if np.isfinite(x)]
                row[f"{kk}_mean"] = float(np.mean(vv)) if vv else float("nan")
            rows.append(row)
    return rows
