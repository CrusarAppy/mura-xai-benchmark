"""Robustness and sanity-check evaluation (proposal Section 3.9.4).

Two families:
  1. Input robustness — perturb the image (Gaussian noise, brightness/contrast) and measure
     how stable the saliency map is (SSIM + Spearman vs. the clean map), reported per
     severity level rather than pooled.
  2. Sanity checks (Adebayo et al., 2018) — progressively randomise the model's learned
     parameters (cascading and independent) and a label-randomisation control; a faithful
     method's map should diverge as parameters are destroyed. Divergence is 1 - similarity.

Explainers follow the project interface: `explainer(x, target_class=1) -> (N,H,W)` numpy in [0,1].
"""
from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np

from .agreement import ssim_map, spearman_map


def _map_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    n = min(a.shape[0], b.shape[0])
    s = [ssim_map(a[i], b[i]) for i in range(n)]
    r = [spearman_map(a[i], b[i]) for i in range(n)]
    s = [v for v in s if np.isfinite(v)]; r = [v for v in r if np.isfinite(v)]
    return {"ssim": float(np.mean(s)) if s else float("nan"),
            "spearman": float(np.mean(r)) if r else float("nan")}


# ---------- input perturbations ----------

def add_gaussian_noise(x, sigma_frac: float):
    """Additive Gaussian noise with sigma = sigma_frac * per-image dynamic range."""
    import torch
    dr = (x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True))
    return x + (sigma_frac * dr) * torch.randn_like(x)


def adjust_brightness_contrast(x, amount: float):
    """Bounded contrast/brightness shift: scale around the per-image mean by (1+amount)."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    return (x - mean) * (1.0 + amount) + mean * (1.0 + amount)


def input_robustness(explainer: Callable, x, target_class: int = 1,
                     noise_sigmas=(0.05, 0.10), bc_amounts=(0.10, 0.20)) -> Dict[str, float]:
    """Stability of the saliency map under two severities each of noise and brightness/contrast.
    Returns similarity (SSIM/Spearman) of perturbed vs clean maps, per severity level.
    """
    import torch
    with torch.no_grad():
        pass
    clean = np.asarray(explainer(x, target_class=target_class))
    out: Dict[str, float] = {}
    for tag, sig in zip(("lo", "hi"), noise_sigmas):
        xp = add_gaussian_noise(x, sig)
        sim = _map_similarity(clean, np.asarray(explainer(xp, target_class=target_class)))
        out[f"robust_ssim_noise_{tag}"] = sim["ssim"]
        out[f"robust_spearman_noise_{tag}"] = sim["spearman"]
    for tag, amt in zip(("lo", "hi"), bc_amounts):
        xp = adjust_brightness_contrast(x, amt)
        sim = _map_similarity(clean, np.asarray(explainer(xp, target_class=target_class)))
        out[f"robust_ssim_bc_{tag}"] = sim["ssim"]
        out[f"robust_spearman_bc_{tag}"] = sim["spearman"]
    return out


# ---------- sanity checks (Adebayo et al.) ----------

def _param_modules(model):
    """Modules with learnable weights, in forward order (Conv/Linear/Norm)."""
    import torch.nn as nn
    mods = [m for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm))
            and any(p.requires_grad for p in m.parameters(recurse=False))]
    return mods


def _reinit(module):
    import torch.nn as nn
    for p in module.parameters(recurse=False):
        if p.dim() > 1:
            nn.init.kaiming_normal_(p)
        else:
            nn.init.zeros_(p)


def sanity_check(build_explainer_for_model: Callable, model, x, target_class: int = 1,
                 mode: str = "cascading", n_steps: int = 5) -> List[Dict]:
    """Randomise learned parameters top-down and measure explanation divergence.

    build_explainer_for_model(model) -> explainer callable (rebuilt each step because the
    hooks/target layer must bind to the mutated model). mode:
      - 'cascading': accumulate randomisation from the output layer downward
      - 'independent': randomise only one block at each step (model reset between steps)
    Returns rows: {mode, fraction_randomised, ssim, spearman} where low similarity = good
    (the explanation depends on the learned weights).
    """
    import copy
    import numpy as np

    ref_model = copy.deepcopy(model).eval()
    ref = np.asarray(build_explainer_for_model(ref_model)(x, target_class=target_class))

    work = copy.deepcopy(model).eval()
    mods = _param_modules(work)[::-1]          # output side first
    if not mods:
        return []
    steps = np.linspace(1, len(mods), min(n_steps, len(mods))).astype(int)
    rows = []
    for cut in steps:
        if mode == "independent":
            work = copy.deepcopy(model).eval()
            mm = _param_modules(work)[::-1]
            _reinit(mm[cut - 1])
        else:  # cascading
            for m in _param_modules(work)[::-1][:cut]:
                _reinit(m)
        cur = np.asarray(build_explainer_for_model(work)(x, target_class=target_class))
        sim = _map_similarity(ref, cur)
        rows.append({"mode": mode, "fraction_randomised": float(cut / len(mods)),
                     "ssim": sim["ssim"], "spearman": sim["spearman"]})
    return rows


def label_randomization_control(build_explainer_for_model: Callable, model_trained,
                                model_random, x, target_class: int = 1) -> Dict[str, float]:
    """Compare explanations from the trained model vs an untrained (random-label proxy)
    model. Low similarity indicates the explanation reflects learned structure."""
    ref = np.asarray(build_explainer_for_model(model_trained)(x, target_class=target_class))
    rnd = np.asarray(build_explainer_for_model(model_random)(x, target_class=target_class))
    return _map_similarity(ref, rnd)
