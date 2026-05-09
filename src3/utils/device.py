"""GPU detection helpers re-exported from src/ to avoid drift.

src3 reuses the original probes — there is one project, one CUDA toolkit,
one decision about which backend to use. Duplicating would invite skew.
"""

from __future__ import annotations

from src.utils.device import (
    lgbm_device,
    torch_device,
    lgbm_params_with_device,
    report,
)

__all__ = ["lgbm_device", "torch_device", "lgbm_params_with_device", "report"]
