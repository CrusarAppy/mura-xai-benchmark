"""Layer-7 Benchmark Validation (proposal Section 3.3.2 — the principal contribution).

Rather than evaluating the XAI methods, this validates the *benchmark itself*:

  * internal validity   — were all configurations produced under identical conditions?
                          (consistent preprocessing/seed/env stamped on every row)
  * construct validity  — are the evaluation dimensions non-redundant? (inter-metric
                          correlation; |r| ~ 1 means two metrics measure the same thing)
  * reproducibility/stability — variance and intraclass correlation of each metric across
                          folds & seeds within a configuration (feeds H4)
  * extensibility       — asserted by scripts/check_repo.py (a new explainer/metric registers
                          and runs from a clean checkout)

Pure Pandas/NumPy. Produces a JSON-serialisable report.
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd


CONFIG_KEYS = ["region", "backbone", "xai_method"]


def internal_validity(df: pd.DataFrame, invariant_cols: Sequence[str] = ("env_torch", "env_python")) -> Dict:
    """Every row should share the same environment / protocol stamp. Reports, per invariant
    column present, how many distinct values were seen (1 = internally consistent)."""
    report = {}
    for c in invariant_cols:
        if c in df.columns:
            vals = sorted(map(str, df[c].dropna().unique()))
            report[c] = {"distinct": len(vals), "values": vals, "consistent": len(vals) <= 1}
    report["overall_consistent"] = all(v["consistent"] for v in report.values()) if report else None
    return report


def construct_validity(df: pd.DataFrame, metrics: Sequence[str],
                       redundant_threshold: float = 0.95) -> Dict:
    """Pearson correlation between evaluation dimensions; flag near-collinear pairs whose
    |r| exceeds the threshold (candidates for redundancy)."""
    cols = [m for m in metrics if m in df.columns]
    corr = df[cols].corr(method="pearson")
    redundant = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= redundant_threshold:
                redundant.append({"a": cols[i], "b": cols[j], "r": float(r)})
    return {"correlation": corr.round(3).to_dict(), "redundant_pairs": redundant,
            "threshold": redundant_threshold}


def _icc1(groups: List[np.ndarray]) -> float:
    """ICC(1): between-group variance / total variance, across repeated measurements per
    configuration. Higher = more reproducible (stable across folds/seeds)."""
    groups = [g[np.isfinite(g)] for g in groups if len(g) > 0]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return float("nan")
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    n = np.mean([len(g) for g in groups])
    ms_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(all_vals) - len(groups))
    denom = ms_between + (n - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else float("nan")


def stability(df: pd.DataFrame, metrics: Sequence[str],
              config_keys: Sequence[str] = CONFIG_KEYS) -> Dict:
    """For each metric: mean within-configuration variance across folds/seeds, and ICC(1)
    treating each configuration as a group. Low variance / high ICC = reproducible."""
    keys = [k for k in config_keys if k in df.columns]
    out = {}
    for m in [m for m in metrics if m in df.columns]:
        grp = df.groupby(keys)[m]
        within_var = grp.var(ddof=1).mean()
        groups = [g[m].to_numpy() for _, g in df.groupby(keys)]
        out[m] = {"mean_within_config_variance": float(within_var) if np.isfinite(within_var) else None,
                  "icc1": _icc1(groups)}
    return out


def validate_benchmark(df: pd.DataFrame, metrics: Sequence[str]) -> Dict:
    """Full Layer-7 report combining the checks above."""
    return {
        "n_rows": int(len(df)),
        "n_configs": int(df.groupby([k for k in CONFIG_KEYS if k in df.columns]).ngroups)
        if any(k in df.columns for k in CONFIG_KEYS) else None,
        "internal_validity": internal_validity(df),
        "construct_validity": construct_validity(df, metrics),
        "stability": stability(df, metrics),
    }
