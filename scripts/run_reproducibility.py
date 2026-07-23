#!/usr/bin/env python
"""Phase D driver: the H4 reproducibility experiment (proposal H4 / Section 3.10.6).

Tests whether a STANDARDISED protocol produces lower-variance explanation metrics than an
UNCONTROLLED one. Reusing one trained checkpoint (so no retraining), it generates the same
faithfulness metric R times under two regimes:

  * controlled   — only the random seed varies; target layer, explainer sample budget, and
                   image subset are fixed (the standardized protocol).
  * uncontrolled — seed AND nuisance factors vary: CAM target layer / attribution sample
                   budget and the evaluated image subset (an unstandardised protocol).

It then compares the between-run variance with Levene + Bartlett and a bootstrap CI on the
variance ratio. A ratio > 1 (var_uncontrolled > var_controlled) supports H4.

Note: this isolates explanation-side variance on a fixed model; training-side variance is a
heavier extension. Requires a checkpoint from the sweep.

Usage:
    python scripts/run_reproducibility.py --config configs/sweep.yaml \
        --region XR_WRIST --backbone densenet121 --method integrated_gradients --runs 10
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
from xai_bench.evaluation import deletion_insertion                    # noqa: E402
from xai_bench.analysis import variance_ratio_test                     # noqa: E402
from xai_bench.reporting import append_result                          # noqa: E402


def _predict_prob(model):
    import torch

    def f(xb):
        with torch.no_grad():
            return torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--metric", default="deletion_auc")
    ap.add_argument("--out", default="results/h4_variance.csv")
    args = ap.parse_args()

    import numpy as np
    import torch

    cfg = load_config(args.config); device = get_device(); print("Device:", device)
    d = cfg["data"]; img = int(d["image_size"])
    set_seed(args.seed)
    index = make_folds(build_mura_index(d["mura_root"], "train", [args.region]),
                       n_folds=int(d["n_folds"]), seed=args.seed)
    va = index[index["fold"] == args.fold].reset_index(drop=True)

    ckpt = Path(cfg["output"]["checkpoints_dir"]) / f"{args.backbone}_{args.region}_f{args.fold}_s{args.seed}.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint: {ckpt} (run the sweep for this region first)")
    model = build_backbone(args.backbone, num_classes=int(cfg["model"]["num_classes"]),
                           pretrained=bool(cfg["model"]["pretrained"]))
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    predict = _predict_prob(model)

    fixed_ds = MuraDataset(va.iloc[:args.n_images].reset_index(drop=True), image_size=img, train=False)
    fixed_x = torch.stack([fixed_ds[i][0] for i in range(len(fixed_ds))]).to(device)
    steps = int(cfg["faithfulness"]["steps"])
    base = cfg["faithfulness"].get("baseline", "blur")

    def _score(x, layer, extra):
        ex = build_explainer(args.method, model, layer, **extra)
        sal = ex(x, target_class=1)
        if hasattr(ex, "remove"):
            ex.remove()
        return deletion_insertion(predict, x, sal, steps=steps, baseline=base)[args.metric]

    default_layer = target_layer_for(model)
    controlled, uncontrolled = [], []
    for r in range(args.runs):
        # controlled: fixed layer / subset / budget; only the seed changes
        set_seed(1000 + r)
        controlled.append(_score(fixed_x, default_layer, {}))

        # uncontrolled: vary subset, and (for attribution methods) the sample budget
        set_seed(2000 + r)
        k = args.n_images
        start = int(torch.randint(0, max(1, len(va) - k), (1,)).item()) if len(va) > k else 0
        sub = MuraDataset(va.iloc[start:start + k].reset_index(drop=True), image_size=img, train=False)
        xu = torch.stack([sub[i][0] for i in range(len(sub))]).to(device)
        extra = {}
        if args.method in ("integrated_gradients",):
            extra = {"n_steps": int(np.random.choice([16, 32, 64]))}
        elif args.method in ("shap",):
            extra = {"n_samples": int(np.random.choice([16, 32, 64]))}
        uncontrolled.append(_score(xu, default_layer, extra))

    res = variance_ratio_test(controlled, uncontrolled)
    row = {"region": args.region, "backbone": args.backbone, "xai_method": args.method,
           "metric": args.metric, "runs": args.runs, **res}
    append_result(args.out, row)
    print("\nH4 variance comparison:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"\nController var={res['var_controlled']:.2e}  uncontrolled var={res['var_uncontrolled']:.2e}")
    print(f"Levene p={res.get('levene_p')}, ratio={res.get('variance_ratio')}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
