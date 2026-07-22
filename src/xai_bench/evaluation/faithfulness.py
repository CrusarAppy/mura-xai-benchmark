"""Faithfulness metrics: Deletion / Insertion AUC, Average Drop, Increase in Confidence.

A `predict_prob(x)->(N,)` callable returns the target-class probability for a batch of
normalized image tensors. Saliency maps are (N,H,W) in [0,1].

- Deletion AUC (lower is better): remove most-important pixels first; confidence should fall fast.
- Insertion AUC (higher is better): add most-important pixels to a baseline; confidence should rise fast.
- Average Drop (lower is better) and Increase in Confidence (higher is better): keep only the
  salient region (thresholded) and compare confidence to the original.
"""
from __future__ import annotations
from typing import Callable, Dict
import numpy as np

# NumPy 2.0 renamed np.trapz -> np.trapezoid; support both.
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:  # NumPy < 2.0
    _trapz = np.trapz


def _blur_baseline(x):
    import torch
    import torch.nn.functional as F
    k = 25
    ch = x.shape[1]
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * 8.0 ** 2)); g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).to(x.device).view(1, 1, k, k).repeat(ch, 1, 1, 1)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kernel, groups=ch)


def make_baseline(x, kind: str = "blur"):
    """Construct a perturbation baseline for x (N,C,H,W).

    - blur: Gaussian-blurred input (removes high-frequency evidence but keeps low-freq layout)
    - mean: each image replaced by its per-channel spatial mean (a flat, information-free image)
    - zero: all zeros (in ImageNet-normalised space this is ~the dataset mean image)
    """
    import torch
    if kind == "blur":
        return _blur_baseline(x)
    if kind == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    if kind == "zero":
        return torch.zeros_like(x)
    raise ValueError(f"unknown baseline {kind!r} (use blur|mean|zero)")


def deletion_insertion(predict_prob: Callable, x, sal, steps: int = 100,
                       baseline: str = "blur") -> Dict[str, float]:
    """Return mean Deletion and Insertion AUC over the batch under the given baseline."""
    import torch
    n, _, H, W = x.shape
    base = make_baseline(x, baseline)
    sal_t = torch.as_tensor(sal, dtype=torch.float32, device=x.device).view(n, -1)
    order = torch.argsort(sal_t, dim=1, descending=True)   # most important first
    total = H * W
    ks = np.linspace(0, total, steps + 1).astype(int)

    del_curves = np.zeros((n, len(ks)))
    ins_curves = np.zeros((n, len(ks)))
    x_flat = x.view(n, x.shape[1], -1)
    base_flat = base.view(n, x.shape[1], -1)

    del_img = x.clone().view(n, x.shape[1], -1)
    ins_img = base.clone().view(n, x.shape[1], -1)
    prev = 0
    for j, k in enumerate(ks):
        if k > prev:
            idx = order[:, prev:k]                      # (n, k-prev)
            for c in range(x.shape[1]):
                del_img[:, c].scatter_(1, idx, base_flat[:, c].gather(1, idx))
                ins_img[:, c].scatter_(1, idx, x_flat[:, c].gather(1, idx))
            prev = k
        with torch.no_grad():
            del_curves[:, j] = predict_prob(del_img.view(n, x.shape[1], H, W))
            ins_curves[:, j] = predict_prob(ins_img.view(n, x.shape[1], H, W))

    xs = ks / total
    del_auc = _trapz(del_curves, xs, axis=1)
    ins_auc = _trapz(ins_curves, xs, axis=1)
    return {"deletion_auc": float(del_auc.mean()), "insertion_auc": float(ins_auc.mean())}


def average_drop_increase(predict_prob: Callable, x, sal, threshold: float = 0.5) -> Dict[str, float]:
    """Keep only the salient region (sal>=threshold) and compare confidence to original."""
    import torch
    n = x.shape[0]
    mask = (torch.as_tensor(sal, dtype=torch.float32, device=x.device) >= threshold).float()
    mask = mask.unsqueeze(1)                              # (n,1,H,W)
    masked = x * mask
    with torch.no_grad():
        o = predict_prob(x); m = predict_prob(masked)
    o = np.asarray(o); m = np.asarray(m)
    avg_drop = np.mean(np.maximum(0.0, o - m) / (o + 1e-8))
    inc_conf = np.mean((m > o).astype(np.float64))
    return {"average_drop": float(avg_drop), "increase_in_confidence": float(inc_conf)}
