#!/usr/bin/env python
"""Full benchmark sweep: all backbones x all XAI methods in one command.

Each backbone is trained ONCE per (fold, seed) and the trained model is reused for
every explainer -- no wasted training. Every (backbone, method, fold, seed) result is
appended as one row to `output.results_csv`. Checkpoints are reused across runs unless
`sweep.force_retrain: true`, so the sweep is resume-friendly (re-run after a crash and
it skips already-trained backbones).

Usage:
    python scripts/run_sweep.py --config configs/sweep.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.utils import load_config, get_device                    # noqa: E402
from xai_bench.seeds import set_seed                                   # noqa: E402
from xai_bench.data import (build_mura_index, make_folds, MuraDataset,  # noqa: E402
                           class_weights, expand_regions)
from xai_bench.models import build_backbone, train_model, collect_probs_logits  # noqa: E402
from xai_bench.evaluation import (classification_metrics, calibration_metrics,  # noqa: E402
                                  temperature_scale)
from xai_bench.reporting import append_result                         # noqa: E402
from xai_bench.pipeline import score_explainer                        # noqa: E402


def _parse_int_list(s):
    return [int(x) for x in str(s).replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", default=None,
                    help="override sweep.folds, e.g. '0,1,2' (for splitting across Kaggle sessions)")
    ap.add_argument("--seeds", default=None, help="override sweep.seeds, e.g. '42,7'")
    ap.add_argument("--regions", default=None,
                    help="override data.regions, e.g. 'XR_WRIST,XR_SHOULDER' or 'all' "
                         "(per-anatomy: one model trained per region)")
    args = ap.parse_args()

    import torch
    from xai_bench.reporting.env_capture import capture_environment

    cfg = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")
    env = capture_environment()   # D3: version/hardware provenance stamped on every row

    d = cfg["data"]
    sw = cfg["sweep"]
    img = int(d["image_size"])
    backbones = list(sw["backbones"])
    methods = list(sw["methods"])
    folds = _parse_int_list(args.folds) if args.folds else list(sw["folds"])
    seeds = _parse_int_list(args.seeds) if args.seeds else list(sw["seeds"])
    region_cfg = args.regions.split(",") if args.regions and args.regions != "all" else \
        ("all" if args.regions == "all" else d.get("regions", "all"))
    regions = expand_regions(region_cfg)   # D1: explicit per-anatomy list
    force_retrain = bool(sw.get("force_retrain", False))

    total = len(regions) * len(backbones) * len(methods) * len(folds) * len(seeds)
    print(f"Sweep plan: {len(regions)} regions x {len(backbones)} backbones x "
          f"{len(methods)} methods x {len(folds)} folds x {len(seeds)} seeds = {total} rows")
    print(f"Regions: {regions}")
    done = 0

    for region in regions:
        for seed in seeds:
            set_seed(seed)
            cfg["experiment"]["seed"] = seed
            index = build_mura_index(d["mura_root"], split="train", regions=[region])
            index = make_folds(index, n_folds=int(d["n_folds"]), seed=seed)

            for fold in folds:
                cfg["data"]["fold"] = fold
                tr_df = index[index["fold"] != fold].reset_index(drop=True)
                va_df = index[index["fold"] == fold].reset_index(drop=True)
                print(f"\n[{region} | seed {seed} | fold {fold}] "
                      f"train={len(tr_df)}  val={len(va_df)}")

                for backbone in backbones:
                    cfg["model"]["backbone"] = backbone
                    set_seed(seed)  # identical init/order for every backbone
                    model = build_backbone(
                        backbone,
                        num_classes=int(cfg["model"]["num_classes"]),
                        pretrained=bool(cfg["model"]["pretrained"]),
                    )
                    ckpt = (Path(cfg["output"]["checkpoints_dir"]) /
                            f"{backbone}_{region}_f{fold}_s{seed}.pt")

                    if ckpt.exists() and not force_retrain:
                        print(f"  [load] {backbone}: reusing {ckpt}")
                        state = torch.load(ckpt, map_location=device)
                        model.load_state_dict(state["model_state"] if "model_state" in state else state)
                        model.to(device)
                    else:
                        print(f"  [train] {backbone} ({region}, fold {fold}, seed {seed})")
                        train_model(
                            model,
                            MuraDataset(tr_df, image_size=img, train=True),
                            MuraDataset(va_df, image_size=img, train=False),
                            cfg, device,
                            class_weight=class_weights(tr_df),
                            ckpt_path=ckpt,
                        )

                    # ---- model-level metrics (once, shared by all explainers) ----
                    metrics_df = va_df.iloc[:64] if cfg["train"].get("quick_debug") else va_df
                    val_loader = torch.utils.data.DataLoader(
                        MuraDataset(metrics_df.reset_index(drop=True), image_size=img, train=False),
                        batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                    )
                    probs, logits, labels = collect_probs_logits(model, val_loader, device)
                    perf = classification_metrics(probs, labels)
                    calib = calibration_metrics(probs, labels)
                    temp = temperature_scale(logits, labels)   # 3.9.2
                    calib["ece_temp_scaled"] = temp["ece_after"]
                    calib["temperature"] = temp["temperature"]
                    print(f"    {backbone}: AUROC={perf.get('auroc', float('nan')):.3f} "
                          f"AUPRC={perf.get('auprc', float('nan')):.3f} "
                          f"ECE={calib.get('ece', float('nan')):.3f}->{temp['ece_after']:.3f} "
                          f"(T={temp['temperature']:.2f})")

                    for method in methods:
                        done += 1
                        print(f"  === [{done}/{total}] {region} {backbone} x {method} ===")
                        mx = score_explainer(model, method, va_df, cfg, device, img)
                        row = {
                            "experiment": cfg["experiment"]["name"],
                            "region": region,
                            "backbone": backbone, "xai_method": method,
                            "fold": fold, "seed": seed, "n_val": len(va_df),
                            **perf, **calib, **mx, **env,
                        }
                        append_result(cfg["output"]["results_csv"], row)

                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    print(f"\nSweep complete. {done} rows -> {cfg['output']['results_csv']}")


if __name__ == "__main__":
    main()
