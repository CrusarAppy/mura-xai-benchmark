"""SHAP for images via GradientSHAP / Expected-Gradients (Lundberg & Lee, NeurIPS 2017).

Gradient-based Shapley-value estimator: average input-gradients over random points
interpolated between a baseline and the input (with small Gaussian noise):

    phi ≈ (x - x') * E_{alpha~U[0,1], eps}[ dF_c/dx  at  x' + alpha(x - x') + eps ]

Default baseline x' = zeros. Per-pixel values are summed over channels and min-max
normalized to [0, 1]. A real data background can be supplied later for closer SHAP
fidelity; this estimator is self-contained (no external SHAP dependency). Interface
matches the CAM methods.
"""
from __future__ import annotations
from ._cam_utils import minmax_per_image, resolve_targets


class GradientSHAP:
    def __init__(self, model, target_layer=None, n_samples: int = 32,
                 stdev: float = 0.1, baseline: str = "blur"):
        self.model = model
        self.n_samples = int(n_samples)
        self.stdev = float(stdev)
        self.baseline = baseline                         # blur|mean|zero (see faithfulness.make_baseline)

    def __call__(self, x, target_class=None):
        import torch
        from ..evaluation.faithfulness import make_baseline
        self.model.eval()
        base = make_baseline(x, self.baseline)
        n = x.shape[0]
        with torch.no_grad():
            tc = resolve_targets(self.model(x), target_class)

        attr = torch.zeros_like(x)
        for _ in range(self.n_samples):
            alpha = torch.rand(n, 1, 1, 1, device=x.device)
            noise = self.stdev * torch.randn_like(x)
            xi = (base + alpha * (x - base) + noise).clone().detach().requires_grad_(True)
            score = self.model(xi).gather(1, tc.view(-1, 1)).sum()
            grad = torch.autograd.grad(score, xi)[0]
            attr += (x - base) * grad.detach()

        attr /= self.n_samples
        m = attr.sum(dim=1)
        return minmax_per_image(m)

    def remove(self):
        pass
