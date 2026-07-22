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
    # Dual-baseline faithfulness (proposal 3.9.3): primary = configured baseline (blur),
    # secondary = dataset-mean, so cross-family rankings can be read across baselines.
    primary = fa.get("baseline", "blur")
    second = "mean" if primary != "mean" else "blur"
    acc = {"deletion_auc": [], "insertion_auc": [],
           "deletion_auc_mean": [], "insertion_auc_mean": [],
           "average_drop": [], "increase_in_confidence": []}
    runtimes = []
    gpu_mem = []
    use_cuda = device.type == "cuda"
    for xb, _ in tqdm(loader, desc=method):
        xb = xb.to(device)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        with Timer() as tm:
            sal = cam(xb, target_class=1)
        runtimes.append(tm.seconds / xb.shape[0])
        if use_cuda:
            gpu_mem.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2) / xb.shape[0])
        di = deletion_insertion(predict_prob, xb, sal, steps=steps, baseline=primary)
        di2 = deletion_insertion(predict_prob, xb, sal, steps=steps, baseline=second)
        ad = average_drop_increase(predict_prob, xb, sal, threshold=float(fa["keep_threshold"]))
        acc["deletion_auc"].append(di["deletion_auc"])
        acc["insertion_auc"].append(di["insertion_auc"])
        acc["deletion_auc_mean"].append(di2["deletion_auc"])
        acc["insertion_auc_mean"].append(di2["insertion_auc"])
        acc["average_drop"].append(ad["average_drop"])
        acc["increase_in_confidence"].append(ad["increase_in_confidence"])
    cam.remove()

    out = {kk: float(np.mean(v)) for kk, v in acc.items()}
    out["n_explained"] = k
    out["runtime_s_per_explanation"] = float(np.mean(runtimes))
    out["gpu_mem_mb_per_explanation"] = float(np.mean(gpu_mem)) if gpu_mem else float("nan")
    out["faithfulness_baseline_primary"] = primary
    out["faithfulness_baseline_secondary"] = second
    return out
