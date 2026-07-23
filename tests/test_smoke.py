"""Smoke tests on tiny synthetic data — no MURA and no GPU required.

Torch-dependent tests skip gracefully if torch is not installed; the numpy/pandas
tests always run. Run with `pytest`.
"""
import numpy as np
import pytest


def _tiny_net():
    """Small CNN with a `.features` block so CAM hooking works like the real backbones."""
    import torch.nn as nn
    import torch

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            )
            self.head = nn.Linear(16, 2)
            self._feature_module = "features"

        def forward(self, x):
            f = self.features(x)
            p = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            return self.head(p)

    return TinyNet().eval()


def test_gradcam_shape_and_range():
    torch = pytest.importorskip("torch")
    from xai_bench.explainers import GradCAM
    from xai_bench.models.backbones import target_layer_for
    net = _tiny_net()
    cam = GradCAM(net, target_layer_for(net))
    sal = cam(torch.randn(4, 3, 32, 32), target_class=1)
    cam.remove()
    assert sal.shape == (4, 32, 32)
    assert sal.min() >= -1e-6 and sal.max() <= 1.0 + 1e-6


def test_all_explainers():
    """All six explainers (4 CAM + IG + SHAP) return a normalized (N,H,W) map."""
    torch = pytest.importorskip("torch")
    from xai_bench.explainers import EXPLAINERS
    from xai_bench.models.backbones import target_layer_for
    net = _tiny_net()
    x = torch.randn(2, 3, 32, 32)
    for name, cls in EXPLAINERS.items():
        m = cls(net, target_layer_for(net))
        sal = m(x, target_class=1)
        m.remove()
        assert sal.shape == (2, 32, 32), name
        assert sal.min() >= -1e-6 and sal.max() <= 1.0 + 1e-6, name


def test_faithfulness_runs():
    torch = pytest.importorskip("torch")
    from xai_bench.evaluation import deletion_insertion, average_drop_increase
    net = _tiny_net()
    x = torch.randn(3, 3, 32, 32)
    sal = np.random.rand(3, 32, 32).astype("float32")

    def predict_prob(xb):
        with torch.no_grad():
            return torch.softmax(net(xb), dim=1)[:, 1].numpy()

    di = deletion_insertion(predict_prob, x, sal, steps=10, baseline="zero")
    ad = average_drop_increase(predict_prob, x, sal, threshold=0.5)
    assert 0.0 <= di["deletion_auc"] <= 1.0
    assert 0.0 <= di["insertion_auc"] <= 1.0
    assert 0.0 <= ad["average_drop"] <= 1.0
    assert 0.0 <= ad["increase_in_confidence"] <= 1.0


def test_metrics():
    from xai_bench.evaluation import classification_metrics, calibration_metrics
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    labels = np.array([0, 1, 0, 1])
    perf = classification_metrics(probs, labels)
    calib = calibration_metrics(probs, labels, n_bins=5)
    assert perf["accuracy"] == 1.0
    assert 0.0 <= calib["ece"] <= 1.0 and calib["brier"] >= 0.0


def test_phaseA_classification_extras():
    """A1: AUPRC, Youden-J threshold, and *_youden metrics present and in range."""
    from xai_bench.evaluation import classification_metrics, youden_threshold
    rng = np.random.default_rng(0)
    labels = np.array([0] * 30 + [1] * 10)          # imbalanced, like MURA
    p1 = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.15, 40), 0, 1)
    probs = np.stack([1 - p1, p1], axis=1)
    m = classification_metrics(probs, labels)
    for key in ("auprc", "threshold_youden", "precision_youden", "recall_youden", "f1_youden"):
        assert key in m, key
    assert 0.0 <= m["auprc"] <= 1.0
    assert 0.0 <= m["threshold_youden"] <= 1.0
    assert youden_threshold(labels, p1) == m["threshold_youden"]


def test_phaseA_temperature_scaling():
    """A2: temperature scaling returns T>0 and does not worsen ECE on overconfident logits."""
    from xai_bench.evaluation import temperature_scale, reliability_curve
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    # deliberately overconfident logits (large magnitude) -> T>1 should help
    z1 = 4.0 * (2 * labels - 1) + rng.normal(0, 2.0, 200)
    logits = np.stack([-z1, z1], axis=1)
    out = temperature_scale(logits, labels)
    assert out["temperature"] > 0
    assert out["ece_after"] <= out["ece_before"] + 1e-6
    probs = np.stack([1 / (1 + np.exp(z1)), 1 / (1 + np.exp(-z1))], axis=1)
    curve = reliability_curve(probs, labels, n_bins=10)
    assert len(curve) == 10 and sum(r["count"] for r in curve) == 200


def test_phaseA_baselines_differ():
    """A3/A4: make_baseline produces distinct blur/mean/zero tensors."""
    torch = pytest.importorskip("torch")
    from xai_bench.evaluation import make_baseline
    x = torch.randn(2, 3, 16, 16)
    b = make_baseline(x, "blur"); m = make_baseline(x, "mean"); z = make_baseline(x, "zero")
    assert b.shape == x.shape == m.shape == z.shape
    assert float(z.abs().sum()) == 0.0
    assert not torch.allclose(b, m)
    # mean baseline is spatially constant per channel
    assert torch.allclose(m[:, :, 0, 0], m[:, :, -1, -1])


def test_phaseB_agreement_identities():
    """B1: self-agreement is perfect; disjoint maps have zero overlap; energy threshold works."""
    from xai_bench.evaluation import (binarise_topk, iou, dice, agreement_pair,
                                      pairwise_agreement)
    a = np.zeros((8, 8), dtype="float32"); a[:2, :2] = 1.0     # 4 hot pixels
    m = binarise_topk(a, k_percent=20.0)
    assert m.sum() >= 1 and m[0, 0] == 1.0
    assert iou(m, m) == 1.0 and dice(m, m) == 1.0
    b = np.zeros((8, 8), dtype="float32"); b[-2:, -2:] = 1.0   # disjoint hot region
    r = agreement_pair(a, b, k_percent=20.0)
    assert r["iou"] == 0.0 and r["dice"] == 0.0
    maps = {"m1": a[None], "m2": a[None]}                       # identical stacks
    rows = pairwise_agreement(maps, k_percent=20.0)
    assert len(rows) == 1 and rows[0]["iou_mean"] == 1.0


def test_phaseB_binarise_energy_fraction():
    """B1: energy threshold retains ~k% of the saliency MASS, not k% of pixels."""
    from xai_bench.evaluation import binarise_topk
    sal = np.zeros((10, 10), dtype="float32")
    sal[0, 0] = 9.0                      # one pixel holds 90% of the mass
    sal[1:, :] = 0.0
    sal[0, 1] = 1.0                      # total mass = 10
    mask = binarise_topk(sal, k_percent=50.0)   # 50% of mass -> just the 9.0 pixel
    assert mask[0, 0] == 1.0 and mask.sum() == 1.0


def test_phaseB_robustness_perturbations():
    """B2: perturbations preserve shape and actually change the image."""
    torch = pytest.importorskip("torch")
    from xai_bench.evaluation import add_gaussian_noise, adjust_brightness_contrast
    x = torch.randn(2, 3, 16, 16)
    xn = add_gaussian_noise(x, 0.1); xb = adjust_brightness_contrast(x, 0.2)
    assert xn.shape == x.shape and xb.shape == x.shape
    assert not torch.allclose(xn, x) and not torch.allclose(xb, x)


def test_phaseC_aggregation():
    """C1: TOPSIS/weighted-sum/Borda agree that the dominant alternative wins; Pareto+tau."""
    import pandas as pd
    from xai_bench.analysis import (rank_methods, topsis, pareto_front, weight_sensitivity,
                                    normalize)
    # method A dominates on every criterion (higher del is *worse*; lower is better)
    df = pd.DataFrame({
        "xai_method": ["A", "B", "C"],
        "insertion_auc": [0.9, 0.6, 0.5],          # higher better
        "deletion_auc": [0.1, 0.4, 0.5],           # lower better
        "runtime_s_per_explanation": [0.1, 1.0, 2.0],  # lower better
    })
    cols = ["insertion_auc", "deletion_auc", "runtime_s_per_explanation"]
    nrm = normalize(df, cols)
    assert (nrm.loc[0] >= nrm.loc[1]).all()        # A normalises to the best on all
    ranking = rank_methods(df, cols, id_col="xai_method")
    assert ranking.iloc[0]["xai_method"] == "A"
    assert bool(ranking.iloc[0]["pareto_optimal"]) is True
    # dominated C should not be Pareto-optimal
    pf = pareto_front(df, cols)
    assert pf[0] and not pf[2]
    tau = ranking.attrs["kendall_tau"]
    assert abs(tau.loc["topsis", "topsis"] - 1.0) < 1e-9
    sens = weight_sensitivity(df, cols, {"equal": [1, 1, 1], "eff": [1, 1, 3]},
                              id_col="xai_method")
    assert sens.iloc[0]["xai_method"] == "A"       # A is robust to weighting


def test_phaseC_statistics():
    """C2: Friedman detects a planted method effect; Nadeau-Bengio inflates variance."""
    import numpy as np, pandas as pd
    from xai_bench.analysis import (friedman_test, nemenyi_posthoc, corrected_resampled_ttest,
                                    bootstrap_ci)
    rng = np.random.default_rng(0)
    rows = []
    # method C is consistently better (higher insertion) across all blocks
    for region in ["wrist", "elbow", "hand", "shoulder"]:
        for fold in range(3):
            for bb in ["densenet121", "efficientnet_b0"]:
                base = {"densenet121": 0.5, "efficientnet_b0": 0.55}[bb]
                for m, bump in [("A", 0.0), ("B", 0.02), ("C", 0.15)]:
                    rows.append({"xai_method": m, "backbone": bb, "region": region,
                                 "fold": fold,
                                 "insertion_auc": base + bump + rng.normal(0, 0.01)})
    df = pd.DataFrame(rows)
    fr = friedman_test(df, "insertion_auc")
    assert fr["p"] < 0.05 and fr["k"] == 3
    nem = nemenyi_posthoc(df, "insertion_auc", higher_better=True)
    assert nem["mean_ranks"]["C"] < nem["mean_ranks"]["A"]   # lower rank = better
    # Nadeau-Bengio: corrected SE must exceed naive SE
    diffs = [0.05, 0.04, 0.06, 0.03, 0.05]
    nb = corrected_resampled_ttest(diffs, n_test=200, n_train=800)
    naive_se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    assert nb["se_corrected"] > naive_se
    ci = bootstrap_ci([0.1, 0.2, 0.15, 0.18, 0.12], seed=1)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_phaseD_variance_ratio_h4():
    """D2/H4: uncontrolled protocol has larger variance -> ratio>1, Levene detects it."""
    import numpy as np
    from xai_bench.analysis import variance_ratio_test
    rng = np.random.default_rng(0)
    controlled = 0.5 + rng.normal(0, 0.01, 20)     # tight (standardised)
    uncontrolled = 0.5 + rng.normal(0, 0.08, 20)   # loose (nuisance factors vary)
    res = variance_ratio_test(controlled, uncontrolled, n_boot=2000)
    assert res["var_uncontrolled"] > res["var_controlled"]
    assert res["variance_ratio"] > 1.0
    assert res["levene_p"] < 0.05
    assert res["variance_ratio_ci_lo"] > 0


def test_phaseD_benchmark_validation():
    """D4: internal validity, construct redundancy, and stability report shape."""
    import numpy as np, pandas as pd
    from xai_bench.analysis import validate_benchmark, construct_validity, internal_validity
    rng = np.random.default_rng(1)
    rows = []
    for region in ["wrist", "elbow"]:
        for bb in ["densenet121", "efficientnet_b0"]:
            for m in ["gradcam", "shap"]:
                for fold in range(3):
                    base = 0.9 - 0.05 * (m == "shap")
                    rows.append({"region": region, "backbone": bb, "xai_method": m,
                                 "fold": fold, "env_torch": "2.3.1", "env_python": "3.11.0",
                                 "auroc": base + rng.normal(0, 0.005),
                                 "auprc": base + rng.normal(0, 0.005),      # ~collinear with auroc
                                 "deletion_auc": 0.3 + rng.normal(0, 0.02),
                                 "runtime_s_per_explanation": 0.1 + rng.normal(0, 0.001)})
    df = pd.DataFrame(rows)
    rep = validate_benchmark(df, ["auroc", "auprc", "deletion_auc", "runtime_s_per_explanation"])
    assert rep["internal_validity"]["overall_consistent"] is True    # single env stamp
    # auroc & auprc are near-collinear here -> flagged redundant
    pairs = {(p["a"], p["b"]) for p in rep["construct_validity"]["redundant_pairs"]}
    assert ("auroc", "auprc") in pairs
    assert "auroc" in rep["stability"] and "icc1" in rep["stability"]["auroc"]


def test_patient_level_split_no_leakage():
    import pandas as pd
    from xai_bench.data.mura import make_folds
    df = pd.DataFrame({
        "filepath": [f"img{i}.png" for i in range(40)],
        "label": [i % 2 for i in range(40)],
        "patient": [f"patient{i // 4:03d}" for i in range(40)],  # 4 images per patient
        "study": ["s"] * 40, "region": ["XR_WRIST"] * 40, "split": ["train"] * 40,
    })
    folded = make_folds(df, n_folds=5, seed=1)
    per_patient_folds = folded.groupby("patient")["fold"].nunique()
    assert (per_patient_folds == 1).all()
