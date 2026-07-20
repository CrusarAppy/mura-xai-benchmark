"""Score-CAM (Wang et al., CVPR-W 2020) — gradient-free.

For each activation map A_k: upsample to input size, normalize to [0,1], mask the input,
forward-pass, and use the target-class softmax score as the weight w_k:
    L = ReLU( sum_k w_k * A_k )
No gradients are used. This costs K forward passes per image (K = #channels), so it is
run on a representative subset and channels are processed in chunks for memory.
Returns numpy (N, H, W) in [0, 1].
"""
from __future__ import annotations
from ._cam_utils import ActivationGradientHook, minmax_per_image, resolve_targets


class ScoreCAM:
    def __init__(self, model, target_layer, chunk: int = 64):
        self.model = model
        self.chunk = chunk
        self.hook = ActivationGradientHook(target_layer, capture_grads=False)

    def __call__(self, x, target_class=None):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)                     # populates activations via hook
        tc = resolve_targets(logits, target_class)
        acts = self.hook.acts                          # (N,K,h,w)
        N, K, h, w = acts.shape
        H, W = x.shape[-2:]
        cams = torch.zeros((N, H, W), device=x.device)

        for i in range(N):
            A = acts[i]                                # (K,h,w)
            up = F.interpolate(A.unsqueeze(1), size=(H, W), mode="bilinear",
                               align_corners=False).squeeze(1)   # (K,H,W)
            amin = up.view(K, -1).min(dim=1).values.view(K, 1, 1)
            amax = up.view(K, -1).max(dim=1).values.view(K, 1, 1)
            up_n = (up - amin) / (amax - amin + 1e-8)  # per-map [0,1]
            weights = torch.zeros(K, device=x.device)
            xi = x[i:i + 1]
            with torch.no_grad():
                for s in range(0, K, self.chunk):
                    masks = up_n[s:s + self.chunk].unsqueeze(1)   # (c,1,H,W)
                    masked = xi * masks                           # (c,3,H,W)
                    sc = F.softmax(self.model(masked), dim=1)[:, tc[i]]
                    weights[s:s + self.chunk] = sc
            cam = F.relu((weights.view(K, 1, 1) * up).sum(dim=0))  # (H,W)
            cams[i] = cam
        return minmax_per_image(cams)

    def remove(self):
        self.hook.remove()
