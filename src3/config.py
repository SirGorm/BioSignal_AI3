"""OmegaConf-backed config loader.

Replaces ad-hoc `yaml.safe_load(open(...))` calls scattered across src/.
Returns a DictConfig that supports attribute access (cfg.training.lr) and
CLI dotted overrides (parsed by OmegaConf.from_dotlist).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "src3.yaml"
LEGACY_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Load a YAML config and apply OmegaConf dotted overrides.

    Parameters
    ----------
    path      : YAML config file. Defaults to configs/config.yaml.
    overrides : Iterable of "section.key=value" strings (e.g. parsed argv).
    """
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def resolve_paths(cfg: DictConfig) -> DictConfig:
    """Resolve every path in cfg.paths to an absolute Path under PROJECT_ROOT.

    Mutates and returns cfg for convenience.
    """
    if "paths" not in cfg:
        return cfg
    for k, v in cfg.paths.items():
        p = Path(v)
        if not p.is_absolute():
            cfg.paths[k] = str((PROJECT_ROOT / p).resolve())
    return cfg
