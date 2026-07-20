from .gradcam import GradCAM
from .gradcampp import GradCAMpp
from .scorecam import ScoreCAM
from .layercam import LayerCAM

# CAM-family registry (name -> class). Attribution methods (Integrated Gradients, SHAP)
# are added in a later phase.
EXPLAINERS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMpp,
    "scorecam": ScoreCAM,
    "layercam": LayerCAM,
}


def build_explainer(name: str, model, target_layer, **kwargs):
    if name not in EXPLAINERS:
        raise KeyError(f"Unknown explainer '{name}'. Available: {sorted(EXPLAINERS)}")
    return EXPLAINERS[name](model, target_layer, **kwargs)


__all__ = ["GradCAM", "GradCAMpp", "ScoreCAM", "LayerCAM", "EXPLAINERS", "build_explainer"]
