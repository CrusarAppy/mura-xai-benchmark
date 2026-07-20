"""Grad-CAM (Selvaraju et al., ICCV 2017).

Standard formulation:
    w_k = GAP( dY_c / dA_k )              # gradient of class score wrt feature map k, pooled
    L   = ReLU( sum_k w_k * A_k )         # weighted combination of feature maps
    map = normalize( upsample(L, HxW) )   # to input resolution, min-max to [0, 1]

Implemented with forward/backward hooks on the backbone's final feature block.
Returns a numpy array of shape (N, H, W) in [0, 1].
"""
from __future__ import annotations


class GradCAM:
    def __init__(self, model, target_layer):
        import torch  # noqa: F401
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        # full backward hook captures grad wrt the layer output
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self._acts = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, x, target_class=None):
        """x: (N,3,H,W) tensor. target_class: None=predicted, int, or (N,) tensor/list."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)                       # (N, C)
        n = logits.shape[0]

        if target_class is None:
            tc = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            tc = torch.full((n,), target_class, dtype=torch.long, device=logits.device)
        else:
            tc = torch.as_tensor(target_class, dtype=torch.long, device=logits.device)

        score = logits.gather(1, tc.view(-1, 1)).sum()
        score.backward()

        acts = self._acts                            # (N, K, h, w)
        grads = self._grads                          # (N, K, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (N, K, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (N,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                         # (N, H, W)

        # min-max normalize per image to [0, 1]
        flat = cam.view(n, -1)
        cmin = flat.min(dim=1, keepdim=True).values
        cmax = flat.max(dim=1, keepdim=True).values
        cam = (cam - cmin.view(n, 1, 1)) / (cmax - cmin + 1e-8).view(n, 1, 1)
        return cam.detach().cpu().numpy()

    def remove(self):
        self._fh.remove(); self._bh.remove()
