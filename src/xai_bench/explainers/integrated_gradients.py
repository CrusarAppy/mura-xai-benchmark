"""Integrated Gradients (Sundararajan et al., ICML 2017) — self-contained.

    IG_i(x) = (x_i - x'_i) * (1/m) * sum_{k=1..m} dF_c/dx  at  x' + (k/m)(x - x')

x' is the baseline (default: zeros = black image). Per-pixel attributions are summed
over channels to a 2-D map and min-max normalized to [0, 1]. Interface matches the CAM
methods: `IntegratedGradients(model, target_layer=None)`, `__call__(x, target_class)`.
"""
from __future__ import annotations
from ._cam_utils import minmax_per_image, resolve_targets


class IntegratedGradients:
    def __init__(self, model, target_layer=None, n_steps: int = 32, baseline: str = "zero"):
        self.model = model
        self.n_steps = int(n_steps)
        self.baseline = baseline

    def __call__(self, x, target_class=None):
        import torch
        self.model.eval()
        base = torch.zeros_like(x)                       # black baseline
        with torch.no_grad():
            tc = resolve_targets(self.model(x), target_class)

        total_grad = torch.zeros_like(x)
        for k in range(1, self.n_steps + 1):
            alpha = k / self.n_steps
            xi = (base + alpha * (x - base)).clone().detach().requires_grad_(True)
            score = self.model(xi).gather(1, tc.view(-1, 1)).sum()
            grad = torch.autograd.grad(score, xi)[0]
            total_grad += grad.detach()

        attr = (x - base) * (total_grad / self.n_steps)  # (N,3,H,W)
        m = attr.sum(dim=1)                               # (N,H,W)
        return minmax_per_image(m)

    def remove(self):
        pass
