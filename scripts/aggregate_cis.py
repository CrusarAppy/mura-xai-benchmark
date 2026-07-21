#!/usr/bin/env python
"""Aggregate per-fold sweep results into mean +/- 95% CI per (backbone, method).

Handles the multi-session Kaggle workflow: pass several CSVs (one per session) and
they're concatenated. Duplicate (backbone, method, fold, seed) rows are de-duplicated
(last one wins), so re-running a fold is safe.

Usage:
    python scripts/aggregate_cis.py results/sweep_results.csv [more.csv ...]
    python scripts/aggregate_cis.py results/*.csv --out results/summary_cis.csv
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

KEY = ["backbone", "xai_method", "fold", "seed"]
GROUP = ["backbone", "xai_method"]
METRICS = ["auroc", "ece", "brier", "accuracy", "f1",
           "deletion_auc", "insertion_auc", "average_drop",
           "increase_in_confidence", "runtime_s_per_explanation"]
# metric -> is a higher value better? (for the readable printout only)
HIGHER_BETTER = {"auroc": True, "accuracy": True, "f1": True,
                 "insertion_auc": True, "increase_in_confidence": True,
                 "ece": False, "brier": False, "deletion_auc": False,
                 "average_drop": False, "runtime_s_per_explanation": False}

# t critical values (two-sided 95%) for small dof, so scipy isn't required.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042}


def _t95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if dof in _T95:
        return _T95[dof]
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, dof))
    except Exception:
        # nearest tabulated dof, else normal approx
        keys = [k for k in _T95 if k <= dof]
        return _T95[max(keys)] if keys else 1.96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="one or more sweep result CSVs")
    ap.add_argument("--out", default="results/summary_cis.csv")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    frames = [pd.read_csv(p) for p in args.csvs]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last")
    print(f"Loaded {before} rows from {len(args.csvs)} file(s) -> {len(df)} unique (backbone,method,fold,seed).")

    metrics = [m for m in METRICS if m in df.columns]
    rows = []
    for (bb, method), g in df.groupby(GROUP):
        rec = {"backbone": bb, "xai_method": method, "n": len(g),
               "folds": ",".join(map(str, sorted(g["fold"].unique())))}
        for m in metrics:
            vals = g[m].dropna().to_numpy(dtype=float)
            n = len(vals)
            mean = float(np.mean(vals)) if n else float("nan")
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            ci = _t95(n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
            rec[f"{m}_mean"] = mean
            rec[f"{m}_ci95"] = ci
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(GROUP).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # readable printout: mean +/- ci per key metric
    show = ["auroc", "ece", "deletion_auc", "insertion_auc",
            "average_drop", "increase_in_confidence"]
    show = [m for m in show if m in metrics]
    print(f"\nSummary (mean +/- 95% CI over folds)  ->  {args.out}\n")
    hdr = f"{'backbone':15} {'method':20} {'n':>2}  " + "  ".join(f"{m:>22}" for m in show)
    print(hdr)
    for _, r in out.iterrows():
        cells = []
        for m in show:
            cells.append(f"{r[f'{m}_mean']:.3f}+/-{r[f'{m}_ci95']:.3f}".rjust(22))
        print(f"{r['backbone']:15} {r['xai_method']:20} {int(r['n']):>2}  " + "  ".join(cells))

    # best method per backbone by a simple faithfulness signal (low deletion + high insertion)
    if "deletion_auc" in metrics and "insertion_auc" in metrics:
        print("\nBest method per backbone (insertion_mean - deletion_mean, higher = better):")
        out["_faith"] = out["insertion_auc_mean"] - out["deletion_auc_mean"]
        for bb, g in out.groupby("backbone"):
            best = g.loc[g["_faith"].idxmax()]
            print(f"  {bb:15} -> {best['xai_method']} "
                  f"(ins {best['insertion_auc_mean']:.3f} - del {best['deletion_auc_mean']:.3f} "
                  f"= {best['_faith']:.3f})")


if __name__ == "__main__":
    main()
