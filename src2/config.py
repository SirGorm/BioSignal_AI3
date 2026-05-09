"""OmegaConf wrapper around configs/config.yaml.

The legacy code reads the YAML file with `yaml.safe_load` and passes a dict
around. OmegaConf gives:
  - dotted attribute access      (CFG.training.lr)
  - CLI override syntax          (training.lr=2e-3)
  - merge with structured schemas (catches typos at load time)
  - resolvers for ${ref} interpolation (kept off here — config.yaml is flat).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    cli_overrides: List[str] | None = None,
) -> DictConfig:
    """Load configs/config.yaml with optional CLI dot-overrides.

    `cli_overrides` is a list like ['training.lr=2e-3', 'training.batch_size=128'].
    """
    cfg = OmegaConf.load(Path(path))
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
    return cfg


def resolve_paths(cfg: DictConfig, project_root: Path | None = None) -> DictConfig:
    """Convert relative paths in cfg.paths to absolute paths anchored at project_root.

    project_root defaults to the cwd. Mutates and returns cfg.
    """
    root = Path(project_root or Path.cwd()).resolve()
    for key, val in cfg.paths.items():
        p = Path(str(val))
        if not p.is_absolute():
            cfg.paths[key] = str(root / p)
    return cfg
