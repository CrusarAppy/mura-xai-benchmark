#!/usr/bin/env python
"""Phase 1 orchestration: train DenseNet121, run Grad-CAM on a subset, compute
performance + calibration + faithfulness + runtime, and log one reproducible row.

Usage:
    python scripts/run_experiment.py --config configs/densenet_gradcam.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.utils import load_config, get_device, Timer          # noqa: E402
from xai_bench.seeds import set_seed                                 # noqa: E402
from xai_bench.data import (build_mura_index, make_folds, MuraDataset, class_weights)  # noqa: E402
from xai_bench.models import build_backbone, target_layer_for, train_model, evaluate_logits  # noqa: E402
from xai_bench.explainers import build_explainer                     # noqa: E402
from xai_bench.evaluation import (classification_metrics, calibration_metrics,
                                  deletion_insertion, average_drop_increase)  # noqa: E402
from xai_bench.reporting import append_result                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- Data (patient-level CV) ----
    d = cfg["data"]
    train_all = build_mura_index(d["mura_root"], split="train", regions=d.get("regions", "all"))
    train_all = make_folds(train_all, n_folds=int(d["n_folds"]), seed=seed)
    fold = int(d["fold"])
    tr_df = train_all[train_all["fold"] != fold].reset_index(drop=True)
    va_df = train_all[train_all["fold"] == fold].reset_index(drop=True)
    print(f"Train images: {len(tr_df)} | Val images: {len(va_df)} (fold {fold})")

    img = int(d["image_size"])
    train_ds = MuraDataset(tr_df, image_size=img, train=True)
    val_ds = MuraDataset(va_df, image_size=img, train=False)

    # ---- Model + training ----
    m = cfg["model"]
    model = build_backbone(m["backbone"], num_classes=int(m["num_classes"]), pretrained=bool(m["pretrained"]))
    ckpt = Path(cfg["output"]["checkpoints_dir"]) / f"{cfg['experiment']['name']}.pt"
    train_model(model, train_ds, val_ds, cfg, device, class_weight=class_weights(tr_df), ckpt_path=ckpt)

    # ---- Classification + calibration on validation ----
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    probs, labels = evaluate_logits(model, val_loader, device)
    perf = classification_metrics(probs, labels)
    calib = calibration_metrics(probs, labels)
    print("Performance:", perf, "\nCalibration:", calib)

    # ---- Explanation subset ----
    quick = bool(cfg["train"].get("quick_debug"))
    k = min(int(cfg["explain"]["eval_subset"]), len(val_ds))
    steps = int(cfg["faithfulness"]["steps"])
    if quick:                                  # keep the smoke test fast
        k = min(k, 8)
        steps = min(steps, 20)
    exp_bs = min(int(cfg["train"]["batch_size"]), 8)   # smaller batch -> less peak memory
    sub = MuraDataset(va_df.iloc[:k].reset_index(drop=True), image_size=img, train=False)
    sub_loader = torch.utils.data.DataLoader(sub, batch_size=exp_bs, shuffle=False)

    cam = build_explainer(cfg["explain"]["method"], model, target_layer_for(model))

    def predict_prob(xb):
        with torch.no_grad():
            return torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(it, **kw):
            return it

    fa = cfg["faithfulness"]
    del_ins = {"deletion_auc": [], "insertion_auc": []}
    adic = {"average_drop": [], "increase_in_confidence": []}
    runtimes = []
    print(f"Generating {cfg['explain']['method']} explanations + faithfulness on {k} images "
          f"(steps={steps}) ...")
    for xb, _ in tqdm(sub_loader, desc="explain+faithfulness"):
        xb = xb.to(device)
        with Timer() as tm:
            sal = cam(xb, target_class=1)          # explain the 'abnormal' class
        runtimes.append(tm.seconds / xb.shape[0])
        di = deletion_insertion(predict_prob, xb, sal, steps=steps, baseline=fa["baseline"])
        ad = average_drop_increase(predict_prob, xb, sal, threshold=float(fa["keep_threshold"]))
        for kk in del_ins: del_ins[kk].append(di[kk])
        for kk in adic: adic[kk].append(ad[kk])
    cam.remove()

    row = {
        "experiment": cfg["experiment"]["name"],
        "backbone": m["backbone"], "xai_method": cfg["explain"]["method"],
        "fold": fold, "seed": seed, "n_val": len(va_df), "n_explained": k,
        **perf, **calib,
        "deletion_auc": float(np.mean(del_ins["deletion_auc"])),
        "insertion_auc": float(np.mean(del_ins["insertion_auc"])),
        "average_drop": float(np.mean(adic["average_drop"])),
        "increase_in_confidence": float(np.mean(adic["increase_in_confidence"])),
        "runtime_s_per_explanation": float(np.mean(runtimes)),
    }
    append_result(cfg["output"]["results_csv"], row)
    print("Logged result:", row)
    print(f"Results -> {cfg['output']['results_csv']}")


if __name__ == "__main__":
    main()
