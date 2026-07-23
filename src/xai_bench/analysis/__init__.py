"""Post-processing analysis: multi-criteria aggregation and inferential statistics.

Operates on the results CSVs produced by the sweep / agreement / robustness drivers.
Pure NumPy/SciPy/Pandas — no torch, no GPU.
"""
from .aggregation import (METRIC_DIRECTIONS, normalize, pareto_front, topsis,
                          weighted_sum, borda, kendall_tau_between, weight_sensitivity,
                          rank_methods)
from .stats import (shapiro_per_metric, friedman_test, nemenyi_posthoc,
                    wilcoxon_holm, corrected_resampled_ttest, bootstrap_ci,
                    kendalls_w, partial_eta_squared, critical_difference_diagram,
                    variance_ratio_test)
from .validation import (internal_validity, construct_validity, stability,
                         validate_benchmark)

__all__ = [
    "METRIC_DIRECTIONS", "normalize", "pareto_front", "topsis", "weighted_sum",
    "borda", "kendall_tau_between", "weight_sensitivity", "rank_methods",
    "shapiro_per_metric", "friedman_test", "nemenyi_posthoc", "wilcoxon_holm",
    "corrected_resampled_ttest", "bootstrap_ci", "kendalls_w",
    "partial_eta_squared", "critical_difference_diagram", "variance_ratio_test",
    "internal_validity", "construct_validity", "stability", "validate_benchmark",
]
