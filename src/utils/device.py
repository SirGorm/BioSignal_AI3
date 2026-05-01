"""GPU detection helpers — cached probes used to auto-select device.

Policy (set by user 2026-04-27): always use GPU when available, fall back to CPU.

Two backends are probed independently because their build flags are independent:
- LightGBM: tries `device='gpu'` (OpenCL) on a tiny dummy fit. Some wheels also
  support `device='cuda'`; we prefer it when available since the LightGBM team
  reports it is faster than the OpenCL backend on NVIDIA hardware.
- PyTorch: `torch.cuda.is_available()`.

Probes run once and are cached in module-level state. They never raise.
"""
from __future__ import annotations

import os
from functools import lru_cache


def _force_cpu() -> bool:
    return os.environ.get("STRENGTH_RT_FORCE_CPU", "0") not in ("0", "", "false", "False")


@lru_cache(maxsize=1)
def lgbm_device() -> str:
    """Return 'cuda', 'gpu', or 'cpu' for use as LightGBM `device` param.

    Tries CUDA build first (faster on NVIDIA), then OpenCL `gpu`, then CPU.
    Cached after first call. Set STRENGTH_RT_FORCE_CPU=1 to override.
    """
    if _force_cpu():
        return "cpu"
    try:
        import lightgbm as lgb
        import numpy as np
    except Exception:
        return "cpu"

    X = np.random.rand(64, 4)
    y = np.random.randint(0, 2, 64)
    for dev in ("cuda", "gpu"):
        try:
            m = lgb.LGBMClassifier(device=dev, n_estimators=2, verbose=-1)
            m.fit(X, y)
            return dev
        except Exception:
            continue
    return "cpu"


@lru_cache(maxsize=1)
def torch_device() -> str:
    """Return 'cuda' or 'cpu' for PyTorch `.to(device)` calls."""
    if _force_cpu():
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def lgbm_params_with_device(params: dict | None = None) -> dict:
    """Merge auto-detected device into a LightGBM params dict (non-destructive)."""
    out = dict(params) if params else {}
    if "device" not in out and "device_type" not in out:
        dev = lgbm_device()
        if dev != "cpu":
            out["device"] = dev
            if dev == "gpu":
                # Best-effort: avoid OpenCL platform mismatch on multi-GPU systems.
                out.setdefault("gpu_platform_id", 0)
                out.setdefault("gpu_device_id", 0)
    return out


def report() -> str:
    """One-line human-readable summary of detected devices."""
    return f"lightgbm={lgbm_device()}, torch={torch_device()}"


if __name__ == "__main__":
    print(report())
