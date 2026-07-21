"""Reusable pipeline steps shared by the single-run and sweep scripts."""
from __future__ import annotations
from typing import Dict


def score_explainer(model, method: str, va_df, cfg: Dict, device, image_size: int) -> Dict:
    """Generate explanations for `method` on a validation subset and return
    faithfulness + runtime metrics. Honors `quick_debug` and per-method subset caps
    (Score-CAM is far heavier, so `explain.eval_subset_scorecam` can shrink it).
    """
    import numpy as np
    import torch
    from .data import MuraDataset
    from .models import target_layer_for
    from .explainers import build_explainer
    from .evaluation import deletion_insertion, average_drop_increase
    from .utils import Timer

    quick = bool(cfg["train"].get("quick_debug"))
    ex = cfg["explain"]
    k = int(ex.get(f"eval_subset_{method}", ex["eval_subset"]))
    k = min(k, len(va_df))
    steps = int(cfg["faithfulness"]["steps"])
    if quick:
        k = min(k, 8)
        steps = min(steps, 20)

    exp_bs = min(int(cfg["train"]["batch_size"]), 8)
    sub = MuraDataset(va_df.iloc[:k].reset_index(drop=True), image_size=image_size, train=False)
    loader = torch.utils.data.DataLoader(sub, batch_size=exp_bs, shuffle=False)
    cam = build_explainer(method, model, target_layer_for(model))

    def predict_prob(xb):
        with torch.no_grad():
            return torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(it, **kw):
            return it

    fa = cfg["faithfulness"]
    acc = {"deletion_auc": [], "insertion_auc": [], "average_drop": [], "increase_in_confidence": []}
    runtimes = []
    for xb, _ in tqdm(loader, desc=method):
        xb = xb.to(device)
        with Timer() as tm:
            sal = cam(xb, target_class=1)
        runtimes.append(tm.seconds / xb.shape[0])
        di = deletion_insertion(predict_prob, xb, sal, steps=steps, baseline=fa["baseline"])
        ad = average_drop_increase(predict_prob, xb, sal, threshold=float(fa["keep_threshold"]))
        acc["deletion_auc"].append(di["deletion_auc"])
        acc["insertion_auc"].append(di["insertion_auc"])
        acc["average_drop"].append(ad["average_drop"])
        acc["increase_in_confidence"].append(ad["increase_in_confidence"])
    cam.remove()

    out = {kk: float(np.mean(v)) for kk, v in acc.items()}
    out["n_explained"] = k
    out["runtime_s_per_explanation"] = float(np.mean(runtimes))
    return out
