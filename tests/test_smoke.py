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
