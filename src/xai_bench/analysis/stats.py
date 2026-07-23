"""Inferential statistics for the benchmark (proposal Section 3.10).

Implements the plan exactly: Shapiro-Wilk normality -> omnibus test (Friedman across the
seven anatomical regions treated as independent datasets, per Demsar) -> Nemenyi post-hoc
+ critical-difference diagram; parametric RM-ANOVA effect size where applicable; pairwise
Wilcoxon signed-rank with Holm correction; the Nadeau-Bengio corrected-resampled t-test for
cross-validation comparisons; and bootstrap confidence intervals.

Everything is NumPy/SciPy/Pandas + Matplotlib. No torch.
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd


# ---- normality -------------------------------------------------------------
def shapiro_per_metric(values: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import shapiro
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"W": float("nan"), "p": float("nan"), "n": len(v)}
    W, p = shapiro(v)
    return {"W": float(W), "p": float(p), "n": len(v)}


# ---- omnibus: Friedman across blocks --------------------------------------
def _pivot_blocks(df: pd.DataFrame, metric: str, treatment: str,
                  block_cols: Sequence[str]) -> pd.DataFrame:
    """Return a blocks x treatments matrix (mean metric per cell), complete cases only."""
    g = df.groupby(list(block_cols) + [treatment])[metric].mean().reset_index()
    wide = g.pivot_table(index=list(block_cols), columns=treatment, values=metric)
    return wide.dropna(axis=0, how="any")


def friedman_test(df: pd.DataFrame, metric: str, treatment: str = "xai_method",
                  block_cols: Sequence[str] = ("backbone", "region", "fold")) -> Dict:
    """Friedman test across blocks. Blocks default to backbone x region x fold; per Demsar
    the anatomical regions are the natural 'datasets'. Returns statistic, p, k, n_blocks,
    Kendall's W effect size, and the treatment column order used."""
    from scipy.stats import friedmanchisquare
    bc = [c for c in block_cols if c in df.columns]
    wide = _pivot_blocks(df, metric, treatment, bc)
    if wide.shape[0] < 2 or wide.shape[1] < 3:
        return {"metric": metric, "error": "need >=2 blocks and >=3 treatments",
                "n_blocks": int(wide.shape[0]), "k": int(wide.shape[1])}
    cols = list(wide.columns)
    stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in cols])
    return {"metric": metric, "statistic": float(stat), "p": float(p),
            "k": len(cols), "n_blocks": int(wide.shape[0]),
            "kendalls_w": kendalls_w(wide.to_numpy()), "treatments": cols,
            "block_cols": bc}


# critical values of the Studentized range q_alpha (alpha=0.05), infinite df, by k
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164}


def nemenyi_posthoc(df: pd.DataFrame, metric: str, treatment: str = "xai_method",
                    block_cols: Sequence[str] = ("backbone", "region", "fold"),
                    higher_better: bool = True) -> Dict:
    """Nemenyi post-hoc on Friedman ranks. Returns mean ranks, the critical difference (CD)
    at alpha=0.05, and a pairwise |rank difference| > CD significance matrix."""
    bc = [c for c in block_cols if c in df.columns]
    wide = _pivot_blocks(df, metric, treatment, bc)
    cols = list(wide.columns)
    k, n = len(cols), wide.shape[0]
    if k < 2 or n < 2:
        return {"metric": metric, "error": "insufficient data"}
    # rank within each block; best rank = 1
    vals = wide.to_numpy()
    order = -vals if higher_better else vals
    ranks = np.apply_along_axis(lambda r: pd.Series(r).rank().to_numpy(), 1, order)
    mean_ranks = ranks.mean(axis=0)
    q = _Q05.get(k, 3.2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    sig = pd.DataFrame(index=cols, columns=cols, dtype=object)
    for i in range(k):
        for j in range(k):
            sig.iloc[i, j] = bool(abs(mean_ranks[i] - mean_ranks[j]) > cd) if i != j else False
    return {"metric": metric, "mean_ranks": dict(zip(cols, mean_ranks.tolist())),
            "cd": float(cd), "k": k, "n_blocks": int(n),
            "significant_pairs": sig}


def kendalls_w(matrix: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) as a Friedman effect size. matrix is
    blocks x treatments; returns W in [0,1]."""
    n, k = matrix.shape
    ranks = np.apply_along_axis(lambda r: pd.Series(r).rank().to_numpy(), 1, matrix)
    Rj = ranks.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    denom = n ** 2 * (k ** 3 - k) / 12.0
    return float(S / denom) if denom > 0 else float("nan")


# ---- pairwise ---------------------------------------------------------------
def wilcoxon_holm(df: pd.DataFrame, metric: str, treatment: str = "xai_method",
                  block_cols: Sequence[str] = ("backbone", "region", "fold")) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests across blocks with Holm-adjusted p-values."""
    from scipy.stats import wilcoxon
    from itertools import combinations
    bc = [c for c in block_cols if c in df.columns]
    wide = _pivot_blocks(df, metric, treatment, bc)
    cols = list(wide.columns)
    rows = []
    for a, b in combinations(cols, 2):
        try:
            stat, p = wilcoxon(wide[a].to_numpy(), wide[b].to_numpy())
        except ValueError:
            stat, p = float("nan"), 1.0
        rows.append({"a": a, "b": b, "statistic": float(stat), "p_raw": float(p)})
    res = pd.DataFrame(rows)
    if len(res):
        order = res["p_raw"].to_numpy().argsort()
        m = len(res)
        adj = np.empty(m)
        run_max = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * res["p_raw"].iloc[idx]
            run_max = max(run_max, val)
            adj[idx] = min(run_max, 1.0)
        res["p_holm"] = adj
    return res


def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    denom = ss_effect + ss_error
    return float(ss_effect / denom) if denom > 0 else float("nan")


# ---- cross-validation comparison: Nadeau-Bengio ----------------------------
def corrected_resampled_ttest(diffs: Sequence[float], n_test: int, n_train: int) -> Dict:
    """Nadeau & Bengio (2003) corrected-resampled t-test for CV-fold performance
    differences. `diffs` = per-fold (score_a - score_b); the variance is inflated by
    (1/J + n_test/n_train) to account for dependence between overlapping folds."""
    from scipy.stats import t as tdist
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    J = len(d)
    if J < 2:
        return {"t": float("nan"), "p": float("nan"), "J": J}
    mean = d.mean()
    var = d.var(ddof=1)
    corr = (1.0 / J + n_test / max(n_train, 1))
    se = np.sqrt(var * corr)
    if se == 0:
        return {"t": float("nan"), "p": float("nan"), "J": J, "mean_diff": float(mean)}
    tstat = mean / se
    p = 2 * (1 - tdist.cdf(abs(tstat), df=J - 1))
    return {"t": float(tstat), "p": float(p), "J": J, "mean_diff": float(mean),
            "se_corrected": float(se)}


# ---- variance comparison (H4) ----------------------------------------------
def variance_ratio_test(controlled: Sequence[float], uncontrolled: Sequence[float],
                        n_boot: int = 10000, seed: int = 0) -> Dict:
    """Test whether the standardized (controlled) protocol yields lower variance than the
    uncontrolled one (proposal H4). Reports Levene and Bartlett p-values plus a bootstrap
    95% CI on the variance ratio var(uncontrolled)/var(controlled) (>1 => standardisation
    reduces variance)."""
    from scipy.stats import levene, bartlett
    c = np.asarray(controlled, float); c = c[np.isfinite(c)]
    u = np.asarray(uncontrolled, float); u = u[np.isfinite(u)]
    out = {"var_controlled": float(np.var(c, ddof=1)) if len(c) > 1 else float("nan"),
           "var_uncontrolled": float(np.var(u, ddof=1)) if len(u) > 1 else float("nan"),
           "n_controlled": len(c), "n_uncontrolled": len(u)}
    if len(c) < 2 or len(u) < 2:
        out.update({"levene_p": float("nan"), "bartlett_p": float("nan")}); return out
    out["levene_p"] = float(levene(c, u)[1])
    out["bartlett_p"] = float(bartlett(c, u)[1])
    out["variance_ratio"] = out["var_uncontrolled"] / out["var_controlled"] \
        if out["var_controlled"] > 0 else float("inf")
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        cb = c[rng.integers(0, len(c), len(c))]
        ub = u[rng.integers(0, len(u), len(u))]
        vc = np.var(cb, ddof=1)
        ratios.append(np.var(ub, ddof=1) / vc if vc > 0 else np.nan)
    ratios = np.asarray(ratios); ratios = ratios[np.isfinite(ratios)]
    if len(ratios):
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        out["variance_ratio_ci_lo"] = float(lo); out["variance_ratio_ci_hi"] = float(hi)
    return out


# ---- bootstrap CI -----------------------------------------------------------
def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": len(v)}


# ---- critical-difference diagram -------------------------------------------
def critical_difference_diagram(mean_ranks: Dict[str, float], cd: float, out_path: str,
                                title: str = "") -> str:
    """Draw a Demsar critical-difference diagram: methods on a rank axis with a CD bar.
    Saves to out_path (PNG) and returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(mean_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]
    kmax = max(ranks) + 0.5
    kmin = min(ranks) - 0.5

    fig, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(names)))
    ax.set_xlim(kmin, kmax)
    ax.set_ylim(0, 1)
    ax.hlines(0.8, kmin, kmax, color="black")
    for x in np.arange(np.ceil(kmin), np.floor(kmax) + 1):
        ax.vlines(x, 0.78, 0.82, color="black")
        ax.text(x, 0.86, str(int(x)), ha="center", fontsize=8)
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        y = 0.6 - i * (0.5 / max(len(names), 1))
        ax.vlines(rk, 0.6 - 0.5, 0.8, color="grey", lw=0.5)
        ax.plot([rk, rk], [0.8, y], color="C0", lw=1)
        ax.text(rk, y, f"  {nm} ({rk:.2f})", va="center", fontsize=9)
    # CD bar
    ax.hlines(0.95, kmin, kmin + cd, color="red", lw=3)
    ax.text(kmin + cd / 2, 0.98, f"CD = {cd:.2f}", ha="center", color="red", fontsize=9)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
