"""Shared helpers for CAM-family explainers: hooks, upsampling, normalization."""
from __future__ import annotations


class ActivationGradientHook:
    """Captures the forward activations and (optionally) backward gradients of a module."""

    def __init__(self, target_layer, capture_grads: bool = True):
        self.acts = None
        self.grads = None
        self._h = [target_layer.register_forward_hook(self._fwd)]
        if capture_grads:
            self._h.append(target_layer.register_full_backward_hook(self._bwd))

    def _fwd(self, module, inp, out):
        self.acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def remove(self):
        for h in self._h:
            h.remove()


def upsample_to(cam, size):
    """cam: (N,1,h,w) -> (N,1,H,W) bilinear."""
    import torch.nn.functional as F
    return F.interpolate(cam, size=size, mode="bilinear", align_corners=False)


def minmax_per_image(cam):
    """cam: (N,H,W) -> per-image min-max to [0,1], returned as numpy (N,H,W)."""
    n = cam.shape[0]
    flat = cam.view(n, -1)
    cmin = flat.min(dim=1, keepdim=True).values
    cmax = flat.max(dim=1, keepdim=True).values
    out = (cam - cmin.view(n, 1, 1)) / (cmax - cmin + 1e-8).view(n, 1, 1)
    return out.detach().cpu().numpy()


def resolve_targets(logits, target_class):
    """Return a (N,) long tensor of target class indices."""
    import torch
    n = logits.shape[0]
    if target_class is None:
        return logits.argmax(dim=1)
    if isinstance(target_class, int):
        return torch.full((n,), target_class, dtype=torch.long, device=logits.device)
    return torch.as_tensor(target_class, dtype=torch.long, device=logits.device)
