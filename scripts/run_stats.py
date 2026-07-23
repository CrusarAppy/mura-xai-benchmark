#!/usr/bin/env python
"""Phase C driver: inferential statistics (proposal 3.10).

For each explanation-quality metric, tests whether XAI methods differ significantly using
blocks = backbone x region x fold (regions as datasets, per Demsar): Shapiro-Wilk on the
per-method distributions, Friedman omnibus + Kendall's W effect size, Nemenyi post-hoc with
a critical-difference diagram, and pairwise Wilcoxon + Holm.

Usage:
    python scripts/run_stats.py --results results/sweep_results.csv \
        [--metrics deletion_auc insertion_auc ...] [--figdir results/figures]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_bench.analysis import (shapiro_per_metric, friedman_test, nemenyi_posthoc,  # noqa: E402
                                wilcoxon_holm, critical_difference_diagram, METRIC_DIRECTIONS)
from xai_bench.reporting import append_result                          # noqa: E402

DEFAULT_METRICS = ["deletion_auc", "insertion_auc", "average_drop",
                   "increase_in_confidence", "runtime_s_per_explanation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep_results.csv")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--treatment", default="xai_method")
    ap.add_argument("--out", default="results/stats_summary.csv")
    ap.add_argument("--figdir", default="results/figures")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.results)
    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in df.columns]
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        Path(args.out).unlink()

    for metric in metrics:
        higher = METRIC_DIRECTIONS.get(metric, True)
        fr = friedman_test(df, metric, treatment=args.treatment)
        if "error" in fr:
            print(f"[{metric}] skipped: {fr['error']} "
                  f"(blocks={fr.get('n_blocks')}, k={fr.get('k')})")
            continue
        nem = nemenyi_posthoc(df, metric, treatment=args.treatment, higher_better=higher)
        row = {"metric": metric, "friedman_stat": fr["statistic"], "friedman_p": fr["p"],
               "kendalls_w": fr["kendalls_w"], "k": fr["k"], "n_blocks": fr["n_blocks"],
               "nemenyi_cd": nem["cd"]}
        row.update({f"meanrank__{m}": r for m, r in nem["mean_ranks"].items()})
        append_result(args.out, row)

        fig = str(Path(args.figdir) / f"cd_{metric}.png")
        critical_difference_diagram(nem["mean_ranks"], nem["cd"], fig,
                                    title=f"{metric} (Friedman p={fr['p']:.3g})")
        # pairwise Wilcoxon + Holm
        wh = wilcoxon_holm(df, metric, treatment=args.treatment)
        wh.insert(0, "metric", metric)
        wh.to_csv(Path(args.figdir).parent / f"wilcoxon_{metric}.csv", index=False)
        sig = "significant" if fr["p"] < 0.05 else "n.s."
        print(f"[{metric}] Friedman p={fr['p']:.4g} ({sig}), W={fr['kendalls_w']:.3f}, "
              f"CD={nem['cd']:.3f} -> {fig}")

    print(f"\nStats summary -> {args.out} ; CD diagrams -> {args.figdir}")


if __name__ == "__main__":
    main()
