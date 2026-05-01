"""Project-wide seaborn plot styling.

All plotting code in this repo (offline QC scripts, training-curve generators,
inspection reports) goes through this helper so figures share a consistent
look: seaborn `white` theme + top/right spines removed via `sns.despine`.

Usage:

    from src.eval.plot_style import apply_style, despine
    apply_style()                       # once, at module import or in main()

    fig, ax = plt.subplots(...)
    ax.plot(...)
    despine(fig=fig)                    # right before fig.savefig(...)
    fig.savefig(out_path)

For scripts that already construct many figures, calling `apply_style()` once
at the top of the module is enough to set the theme; `despine` must still be
called per figure (matplotlib does not retroactively despine).
"""
from __future__ import annotations

from typing import Optional

import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

__all__ = ["apply_style", "despine"]

_DEFAULT_STYLE = "white"
_DEFAULT_CONTEXT = "notebook"


def apply_style(style: str = _DEFAULT_STYLE, context: str = _DEFAULT_CONTEXT) -> None:
    """Set the seaborn theme. Idempotent — safe to call repeatedly."""
    sns.set_theme(style=style, context=context)


def despine(
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    top: bool = True,
    right: bool = True,
    left: bool = False,
    bottom: bool = False,
    **kwargs,
) -> None:
    """Remove top/right spines on `fig` (all axes) or a single `ax`.

    Thin wrapper around `sns.despine` with the project default of stripping
    top + right only. Pass `left=True` / `bottom=True` to strip those too.
    """
    sns.despine(
        fig=fig, ax=ax,
        top=top, right=right, left=left, bottom=bottom,
        **kwargs,
    )
