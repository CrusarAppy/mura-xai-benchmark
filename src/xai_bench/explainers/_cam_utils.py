"""Shared helpers for CAM-family explainers: hooks, upsampling, normalization."""
from __future__ import annotations


class ActivationGradientHook:
    """Captures forward activations and (optionally) their gradients.

    Uses a *tensor* gradient hook on the layer output rather than a module-level
    ``register_full_backward_hook``. Module backward hooks break when the layer's
    output is modified in place downstream (e.g. DenseNet's ``F.relu(..., inplace=True)``
    right after ``features``); tensor hooks avoid that. Activations are cloned so the
    stored values are the true pre-downstream-op feature maps.
    """

    def __init__(self, target_layer, capture_grads: bool = True):
        self.acts = None
        self.grads = None
        self.capture_grads = capture_grads
        self._h = [target_layer.register_forward_hook(self._fwd)]

    def _fwd(self, module, inp, out):
        self.acts = out.detach().clone()
        if self.capture_grads and out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad.detach()

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
