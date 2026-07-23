"""Capture the software/hardware environment for reproducibility (proposal 3.10.6 / 4.2).

Every result row is stamped with library versions and the GPU, so a reader can tell which
environment produced a number. Returns a flat dict of primitives safe for a CSV.
"""
from __future__ import annotations
from typing import Dict


def capture_environment() -> Dict[str, str]:
    import platform
    env: Dict[str, str] = {
        "env_python": platform.python_version(),
        "env_platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            env[f"env_{mod}"] = __import__(mod).__version__
        except Exception:
            env[f"env_{mod}"] = "n/a"
    try:
        import torch
        env["env_torch"] = torch.__version__
        env["env_cuda"] = torch.version.cuda or "cpu"
        env["env_cudnn"] = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "n/a"
        env["env_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu/mps"
        env["env_deterministic"] = str(torch.backends.cudnn.deterministic)
    except Exception:
        env["env_torch"] = "n/a"
    return env
