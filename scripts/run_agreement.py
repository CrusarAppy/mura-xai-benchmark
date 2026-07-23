#!/usr/bin/env python
"""Phase B driver: explanation-agreement analysis (proposal 3.9.7).

For a fixed set of validation images, generate saliency maps for every method on every
backbone (reusing the sweep's checkpoints), then compute:
  * method x method agreement for each backbone (do methods converge on the same evidence?)
  * architecture x architecture agreement for each method (does the backbone change the map?)
across energy-threshold levels k in {10, 20, 30}% for the sensitivity analysis.

Usage:
    python scripts/run_agreement.py --config configs/sweep.yaml [--n-images 40]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.utils import load_config, get_device                     # noqa: E402
from xai_bench.seeds import set_seed                                    # noqa: E402
from xai_bench.data import build_mura_index, make_folds, MuraDataset    # noqa: E402
from xai_bench.models import build_backbone, target_layer_for          # noqa: E402
from xai_bench.explainers import build_explainer                       # noqa: E402
from xai_bench.evaluation import pairwise_agreement                    # noqa: E402
from xai_bench.reporting import append_result                          # noqa: E402


def _load_model(backbone, ckpt, cfg, device):
    import torch
    model = build_backbone(backbone, num_classes=int(cfg["model"]["num_classes"]),
                           pretrained=bool(cfg["model"]["pretrained"]))
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    return model.to(device).eval()


def _gen_maps(model, methods, x):
    import numpy as np
    maps = {}
    for m in methods:
        ex = build_explainer(m, model, target_layer_for(model))
        maps[m] = np.asarray(ex(x, target_class=1))
        ex.remove()
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region", required=True,
                    help="anatomical region to analyse, e.g. XR_WRIST (matches sweep checkpoints)")
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config); device = get_device(); print("Device:", device)
    d = cfg["data"]; sw = cfg["sweep"]; img = int(d["image_size"])
    set_seed(args.seed)
    index = make_folds(build_mura_index(d["mura_root"], "train", [args.region]),
                       n_folds=int(d["n_folds"]), seed=args.seed)
    va = index[index["fold"] == args.fold].reset_index(drop=True).iloc[:args.n_images]
    ds = MuraDataset(va.reset_index(drop=True), image_size=img, train=False)
    xb = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    methods = list(sw["methods"]); backbones = list(sw["backbones"])
    ck = Path(cfg["output"]["checkpoints_dir"])
    out_csv = cfg["output"].get("agreement_csv", "results/agreement.csv")
    ks = [10.0, 20.0, 30.0]

    maps_by_backbone = {}
    for bb in backbones:
        ckpt = ck / f"{bb}_{args.region}_f{args.fold}_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for {bb}: {ckpt}"); continue
        model = _load_model(bb, ckpt, cfg, device)
        maps = _gen_maps(model, methods, xb)
        maps_by_backbone[bb] = maps
        for k in ks:                              # method x method, fixed backbone
            for row in pairwise_agreement(maps, k_percent=k):
                append_result(out_csv, {"scope": "method_x_method", "region": args.region,
                                        "backbone": bb, "method_a": row["a"],
                                        "method_b": row["b"], **row})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # architecture x architecture, fixed method
    for m in methods:
        per_arch = {bb: maps_by_backbone[bb][m] for bb in maps_by_backbone if m in maps_by_backbone[bb]}
        if len(per_arch) < 2:
            continue
        for k in ks:
            for row in pairwise_agreement(per_arch, k_percent=k):
                append_result(out_csv, {"scope": "arch_x_arch", "region": args.region,
                                        "method": m, "backbone_a": row["a"],
                                        "backbone_b": row["b"], **row})

    print("Agreement analysis ->", out_csv)


if __name__ == "__main__":
    main()
