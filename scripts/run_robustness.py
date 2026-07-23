#!/usr/bin/env python
"""Phase B driver: robustness + sanity checks (proposal 3.9.4).

For each backbone x method (reusing the sweep's checkpoints), on a small validation subset:
  * input robustness  -> saliency stability (SSIM/Spearman) under Gaussian noise and
    brightness/contrast, reported per severity level (appended to results/robustness.csv)
  * sanity check      -> cascading parameter randomisation; explanation divergence vs the
    trained model should increase as weights are destroyed (results/sanity.csv)

Usage:
    python scripts/run_robustness.py --config configs/sweep.yaml [--n-images 16]
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
from xai_bench.evaluation import input_robustness, sanity_check        # noqa: E402
from xai_bench.reporting import append_result                          # noqa: E402


def _load_model(backbone, ckpt, cfg, device):
    import torch
    model = build_backbone(backbone, num_classes=int(cfg["model"]["num_classes"]),
                           pretrained=bool(cfg["model"]["pretrained"]))
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region", required=True,
                    help="anatomical region to analyse, e.g. XR_WRIST (matches sweep checkpoints)")
    ap.add_argument("--sanity-steps", type=int, default=5)
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
    rob_csv = cfg["output"].get("robustness_csv", "results/robustness.csv")
    san_csv = cfg["output"].get("sanity_csv", "results/sanity.csv")

    for bb in backbones:
        ckpt = ck / f"{bb}_{args.region}_f{args.fold}_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for {bb}: {ckpt}"); continue
        model = _load_model(bb, ckpt, cfg, device)
        for m in methods:
            print(f"  {bb} x {m}")
            ex = build_explainer(m, model, target_layer_for(model))
            rob = input_robustness(ex, xb, target_class=1)
            ex.remove()
            append_result(rob_csv, {"region": args.region, "backbone": bb, "xai_method": m,
                                    "n_images": int(xb.shape[0]), **rob})

            def _build(mdl, _m=m):
                return build_explainer(_m, mdl, target_layer_for(mdl))
            for row in sanity_check(_build, model, xb, target_class=1,
                                    mode="cascading", n_steps=args.sanity_steps):
                append_result(san_csv, {"region": args.region, "backbone": bb,
                                        "xai_method": m, **row})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Robustness ->", rob_csv, "| Sanity ->", san_csv)


if __name__ == "__main__":
    main()
