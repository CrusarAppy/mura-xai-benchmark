"""Grad-CAM (Selvaraju et al., ICCV 2017).

Standard formulation:
    w_k = GAP( dY_c / dA_k )              # gradient of class score wrt feature map k, pooled
    L   = ReLU( sum_k w_k * A_k )         # weighted combination of feature maps
    map = normalize( upsample(L, HxW) )   # to input resolution, min-max to [0, 1]

Gradients are captured with a tensor hook on the target layer output (see _cam_utils),
which is robust to in-place ops downstream (e.g. DenseNet). Returns numpy (N, H, W) in [0, 1].
"""
from __future__ import annotations
from ._cam_utils import ActivationGradientHook, upsample_to, minmax_per_image, resolve_targets


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.hook = ActivationGradientHook(target_layer, capture_grads=True)

    def __call__(self, x, target_class=None):
        import torch.nn.functional as F
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)                       # (N, C)
        tc = resolve_targets(logits, target_class)
        logits.gather(1, tc.view(-1, 1)).sum().backward()

        A = self.hook.acts                           # (N, K, h, w)
        g = self.hook.grads                          # (N, K, h, w)
        weights = g.mean(dim=(2, 3), keepdim=True)   # (N, K, 1, 1)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))     # (N,1,h,w)
        cam = upsample_to(cam, x.shape[-2:]).squeeze(1)          # (N, H, W)
        return minmax_per_image(cam)

    def remove(self):
        self.hook.remove()
