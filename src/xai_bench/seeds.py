"""Global seed control for reproducibility."""
from __future__ import annotations
import os
import random


def set_seed(seed: int, deterministic: bool = True, full_determinism: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    `full_determinism=True` additionally enforces deterministic CUDA kernels via
    torch.use_deterministic_algorithms and the cuBLAS workspace config (proposal 3.10.6).
    It can slow training and errors on ops without a deterministic implementation, so it is
    opt-in.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if full_determinism:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Worker init fn for DataLoader determinism."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
