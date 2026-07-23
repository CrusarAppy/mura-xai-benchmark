#!/usr/bin/env python
"""Phase C driver: multi-criteria aggregation & ranking (proposal 3.8.1).

Reads the sweep results (optionally merged with agreement/robustness summaries), averages
each metric per XAI method (across backbones/folds/seeds), and produces:
  * results/ranking.csv          — TOPSIS / weighted-sum / Borda scores + ranks + Pareto flag
  * results/ranking_sensitivity.csv — TOPSIS rank under equal / faithfulness / efficiency weights
  * prints the Kendall's tau agreement between the three ranking schemes

Usage:
    python scripts/run_aggregation.py --results results/sweep_results.csv \
        [--metrics deletion_auc insertion_auc average_drop increase_in_confidence runtime_s_per_explanation]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.analysis import (rank_methods, weight_sensitivity, METRIC_DIRECTIONS)  # noqa: E402

# Default criteria = explanation-quality + efficiency (the properties that distinguish
# XAI methods; model-level auroc/ece are shared by all methods on a backbone).
DEFAULT_METRICS = ["deletion_auc", "insertion_auc", "average_drop",
                   "increase_in_confidence", "runtime_s_per_explanation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep_results.csv")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--id-col", default="xai_method")
    ap.add_argument("--out-ranking", default="results/ranking.csv")
    ap.add_argument("--out-sensitivity", default="results/ranking_sensitivity.csv")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.results)
    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in df.columns]
    if not metrics:
        raise SystemExit(f"none of the default metrics present in {args.results}")
    print("Ranking on metrics:", metrics)

    agg = df.groupby(args.id_col)[metrics].mean().reset_index()

    ranking = rank_methods(agg, metrics, id_col=args.id_col)
    Path(args.out_ranking).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out_ranking, index=False)

    # weight schemes: equal, faithfulness-emphasis, efficiency-emphasis
    def _weights(emphasis, factor=3.0):
        w = [factor if _is(emphasis, m) else 1.0 for m in metrics]
        return w

    def _is(emphasis, m):
        if emphasis == "faithfulness":
            return m in ("deletion_auc", "insertion_auc", "average_drop", "increase_in_confidence")
        if emphasis == "efficiency":
            return m in ("runtime_s_per_explanation", "gpu_mem_mb_per_explanation")
        return False

    schemes = {"equal": [1.0] * len(metrics),
               "faithfulness": _weights("faithfulness"),
               "efficiency": _weights("efficiency")}
    sens = weight_sensitivity(agg, metrics, schemes, id_col=args.id_col)
    sens.to_csv(args.out_sensitivity, index=False)

    print("\n== Ranking (TOPSIS order) ==")
    print(ranking.to_string(index=False))
    print("\n== Kendall's tau between schemes ==")
    print(ranking.attrs["kendall_tau"].to_string())
    print("\n== Weight sensitivity (TOPSIS ranks) ==")
    print(sens.to_string(index=False))
    print(f"\nSaved -> {args.out_ranking} , {args.out_sensitivity}")


if __name__ == "__main__":
    main()
