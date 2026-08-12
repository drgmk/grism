"""Multi-panel figures: a grid of `PlotSpec` panels, optionally paginated.

This layer sits *above* the single-panel `render()` seam and reuses it via its
`ax=` hook — nothing in spec/core/render changes. It mirrors seaborn's split
between the single-Axes primitive (`barplot`) and the multi-panel wrapper
(`FacetGrid`/`catplot`).

Typical use: a wide sheet with a sample column (the x categories) and many value
columns. `figure_from_columns` turns one template + a list of value columns into
one panel each; `render_pages` lays them out N-per-page.
"""

from __future__ import annotations

import io
from contextlib import nullcontext
from math import ceil
from typing import Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from .core import _resolve_style
from .render import render
from .spec import SPEC_VERSION, DataBinding, PlotSpec, PlotTemplate


class FigureSpec(BaseModel):
    """A grid of single-panel plots, optionally split across pages."""

    spec_version: int = SPEC_VERSION
    name: str = "figure"
    title: Optional[str] = None  # per-page suptitle
    panels: List[PlotSpec]
    ncols: int = Field(2, ge=1)
    panels_per_page: Optional[int] = Field(None, ge=1)  # None = one page
    panel_size: Tuple[float, float] = (3.5, 4.0)  # (width, height) inches per panel

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "FigureSpec":
        return cls.model_validate_json(text)


def figure_from_columns(
    template: PlotTemplate,
    *,
    x: str,
    y_columns: Sequence[str],
    ncols: int = 2,
    panels_per_page: Optional[int] = None,
    panel_size: Tuple[float, float] = (3.5, 4.0),
    name: str = "figure",
) -> FigureSpec:
    """Build a FigureSpec from one template applied across many value columns.

    Each column in ``y_columns`` becomes a panel with the same look/stats, its
    x categories taken from ``x``. Panels are titled by their column unless the
    template already sets a title.
    """
    panels: List[PlotSpec] = []
    for col in y_columns:
        panel = template.bind(DataBinding(x=x, y=col))
        if panel.labels.title is None:
            panel.labels = panel.labels.model_copy(update={"title": col})
        panels.append(panel)
    return FigureSpec(
        name=name,
        panels=panels,
        ncols=ncols,
        panels_per_page=panels_per_page,
        panel_size=panel_size,
    )


def _chunk(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def render_pages(fig_spec: FigureSpec, df: pd.DataFrame) -> List[Figure]:
    """Render the figure to one Matplotlib Figure per page.

    Each panel is drawn by the single-panel `render()` into its own Axes; unused
    cells on the last page are hidden.
    """
    if not fig_spec.panels:
        return []

    per_page = fig_spec.panels_per_page or len(fig_spec.panels)
    ncols = fig_spec.ncols
    pw, ph = fig_spec.panel_size
    # All panels share a style; apply the first panel's so figure-level rcParams
    # (facecolor, etc.) take effect around subplot creation too.
    style = _resolve_style(fig_spec.panels[0].appearance.style)

    figs: List[Figure] = []
    for page in _chunk(fig_spec.panels, per_page):
        nrows = ceil(len(page) / ncols)
        ctx = plt.style.context(style) if style else nullcontext()
        with ctx:
            fig, axes = plt.subplots(
                nrows, ncols, squeeze=False, figsize=(ncols * pw, nrows * ph)
            )
            flat = axes.ravel()
            for panel, ax in zip(page, flat):
                render(panel, df, ax=ax)
            for ax in flat[len(page):]:  # hide empty cells on the last page
                ax.set_visible(False)
            if fig_spec.title:
                fig.suptitle(fig_spec.title)
            fig.tight_layout()
        figs.append(fig)
    return figs


def render_pdf_bytes(fig_spec: FigureSpec, df: pd.DataFrame) -> bytes:
    """Render to a single multi-page PDF (one page per grid page)."""
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in render_pages(fig_spec, df):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return buf.getvalue()


def render_page_images(fig_spec: FigureSpec, df: pd.DataFrame, fmt: str = "png") -> List[bytes]:
    """Render to one image (png/svg) per page."""
    out: List[bytes] = []
    for fig in render_pages(fig_spec, df):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")
        plt.close(fig)
        out.append(buf.getvalue())
    return out
