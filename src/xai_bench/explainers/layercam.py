"""LayerCAM (Jiang et al., IEEE TIP 2021).

Element-wise positive-gradient weighting of activations:
    w_k^{ij} = ReLU( dY_c/dA_k^{ij} )
    L        = ReLU( sum_k w_k^{ij} * A_k^{ij} )
Uses the backbone's final feature block (single-layer variant; multi-layer fusion
is a later extension). Returns numpy (N, H, W) in [0, 1].
"""
from __future__ import annotations
from ._cam_utils import ActivationGradientHook, upsample_to, minmax_per_image, resolve_targets


class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.hook = ActivationGradientHook(target_layer, capture_grads=True)

    def __call__(self, x, target_class=None):
        import torch.nn.functional as F
        self.model.eval(); self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        tc = resolve_targets(logits, target_class)
        logits.gather(1, tc.view(-1, 1)).sum().backward()

        A = self.hook.acts                          # (N,K,h,w)
        w = F.relu(self.hook.grads)                 # element-wise positive gradients
        cam = F.relu((w * A).sum(dim=1, keepdim=True))   # (N,1,h,w)
        cam = upsample_to(cam, x.shape[-2:]).squeeze(1)
        return minmax_per_image(cam)

    def remove(self):
        self.hook.remove()
