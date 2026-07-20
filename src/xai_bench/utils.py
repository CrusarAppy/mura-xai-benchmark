"""Small utilities: config loading, device, timing, and a lightweight registry."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Callable, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    """Prefer CUDA, then Apple-Silicon MPS, then CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Registry:
    """Minimal name->factory registry so new components plug in without editing core."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(fn: Callable):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = fn
            return fn
        return deco

    def create(self, name: str, *args, **kwargs):
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](*args, **kwargs)

    def available(self):
        return sorted(self._items)


class Timer:
    """Context manager returning elapsed wall-clock seconds via `.seconds`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False
