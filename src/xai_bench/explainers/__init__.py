from .gradcam import GradCAM
from .gradcampp import GradCAMpp
from .scorecam import ScoreCAM
from .layercam import LayerCAM
from .integrated_gradients import IntegratedGradients
from .shap_grad import GradientSHAP

# Full six-method registry: 4 CAM-based + 2 attribution-based.
EXPLAINERS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMpp,
    "scorecam": ScoreCAM,
    "layercam": LayerCAM,
    "integrated_gradients": IntegratedGradients,
    "shap": GradientSHAP,
}


def build_explainer(name: str, model, target_layer, **kwargs):
    if name not in EXPLAINERS:
        raise KeyError(f"Unknown explainer '{name}'. Available: {sorted(EXPLAINERS)}")
    return EXPLAINERS[name](model, target_layer, **kwargs)


__all__ = ["GradCAM", "GradCAMpp", "ScoreCAM", "LayerCAM",
           "IntegratedGradients", "GradientSHAP", "EXPLAINERS", "build_explainer"]
