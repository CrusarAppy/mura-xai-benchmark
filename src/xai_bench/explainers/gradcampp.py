"""Grad-CAM++ (Chattopadhyay et al., WACV 2018).

Pixel-wise weighting of gradients (higher-order terms) for sharper, multi-region maps:
    a_k^{ij} = g^2 / (2 g^2 + sum_{ab} A_k^{ab} g^3)          (g = dY_c/dA_k)
    w_k      = sum_{ij} a_k^{ij} * ReLU(g_k^{ij})
    L        = ReLU( sum_k w_k * A_k )
Gradients are taken w.r.t. the target-class logit (common library variant).
Returns numpy (N, H, W) in [0, 1].
"""
from __future__ import annotations
from ._cam_utils import ActivationGradientHook, upsample_to, minmax_per_image, resolve_targets


class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.hook = ActivationGradientHook(target_layer, capture_grads=True)

    def __call__(self, x, target_class=None):
        import torch
        import torch.nn.functional as F
        self.model.eval(); self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        tc = resolve_targets(logits, target_class)
        logits.gather(1, tc.view(-1, 1)).sum().backward()

        A = self.hook.acts                       # (N,K,h,w)
        g = self.hook.grads                       # (N,K,h,w)
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + (A * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / (denom + 1e-8)
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)   # (N,K,1,1)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))          # (N,1,h,w)
        cam = upsample_to(cam, x.shape[-2:]).squeeze(1)
        return minmax_per_image(cam)

    def remove(self):
        self.hook.remove()
