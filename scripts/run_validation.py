#!/usr/bin/env python
"""Phase D driver: Layer-7 benchmark validation (proposal 3.3.2).

Reads the sweep results and writes a benchmark-validity report (internal validity,
construct validity via inter-metric correlation, and reproducibility/stability via
within-configuration variance and ICC).

Usage:
    python scripts/run_validation.py --results results/sweep_results.csv \
        [--out results/validation_report.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.analysis.validation import validate_benchmark            # noqa: E402

DEFAULT_METRICS = ["auroc", "auprc", "ece", "deletion_auc", "insertion_auc",
                   "average_drop", "increase_in_confidence", "runtime_s_per_explanation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep_results.csv")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--out", default="results/validation_report.json")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.results)
    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in df.columns]
    report = validate_benchmark(df, metrics)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    iv = report["internal_validity"]
    print(f"Rows: {report['n_rows']} | configs: {report['n_configs']}")
    print(f"Internal validity consistent: {iv.get('overall_consistent')}")
    rp = report["construct_validity"]["redundant_pairs"]
    print(f"Redundant metric pairs (|r|>=0.95): "
          f"{[(p['a'], p['b'], p['r']) for p in rp] or 'none'}")
    print("Stability (ICC1 per metric):")
    for m, s in report["stability"].items():
        print(f"  {m:28} ICC={s['icc1']}  within-var={s['mean_within_config_variance']}")
    print(f"\nValidation report -> {args.out}")


if __name__ == "__main__":
    main()
