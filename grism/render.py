"""The render seam: turn a :class:`~grism.spec.PlotSpec` + DataFrame into a
figure. This is the one function the UI, CLI, and API all sit behind.

Everything reusable (data prep, the ``auto`` test pick, figure sizing) lives
here or in core/stats — never in an adapter — so the same spec renders
identically everywhere.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from . import core
from . import stats as stats_mod
from .spec import PlotSpec, StatTest


@dataclass
class RenderResult:
    figure: Figure
    resolved_test: str  # the test "auto" actually chose
    omnibus: Optional[stats_mod.StatResult]
    pairwise: List[stats_mod.PairwiseStatResult]
    normality: dict  # per-group Shapiro p-value, for display


def prepare_data(spec: PlotSpec, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Melt (wide form), filter rows, and resolve group order. Pure; shared by
    every adapter so they all see the same rows in the same order."""
    b = spec.data
    df = df.copy()

    if b.wide_form:
        id_cols = [c for c in b.wide_id_cols if c in df.columns]
        # wide -> long: melted columns are named to match the binding (x/y).
        df = df.melt(id_vars=id_cols, var_name=b.x, value_name=b.y)

    if b.row_filter:
        df = df.query(b.row_filter)

    present = list(pd.Series(df[b.x]).dropna().unique())
    order = [g for g in b.order if g in present] or present
    df_plot = df[df[b.x].isin(order)].copy()
    return df_plot, order


def _resolve_labels(spec: PlotSpec) -> Tuple[str, Optional[str], Optional[str]]:
    b, lab = spec.data, spec.labels
    title = lab.title if lab.title is not None else ""
    xlabel = lab.xlabel  # None -> core falls back to the group name
    ylabel = lab.ylabel  # None -> core falls back to the value name
    return title, xlabel, ylabel


def render(spec: PlotSpec, df: pd.DataFrame, *, ax=None) -> RenderResult:
    """Render a spec against a DataFrame. Does not touch the filesystem."""
    df_plot, order = prepare_data(spec, df)
    b, a, s = spec.data, spec.appearance, spec.stats

    normality = stats_mod.group_normality(df_plot, b.x, b.y, order)
    resolved_test = (
        stats_mod.pick_test_by_normality(normality)
        if s.test is StatTest.auto
        else s.test.value
    )

    elements = [e.value for e in a.elements]
    if "hist" in elements and len(elements) > 1:
        # Histogram is best shown alone (mirrors the old UI guard).
        elements = ["hist"]

    figsize = core.compute_figsize_for(order, a)
    title, xlabel, ylabel = _resolve_labels(spec)
    palette = a.palette if a.use_group_colors else None

    ax_out, omnibus, pairwise = core.plot_with_stats(
        df_plot,
        x=b.x,
        y=b.y,
        hue=b.hue,
        elements=elements,
        test=resolved_test,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        style=a.style,
        figsize=figsize,
        staple_scale=s.staple_scale,
        staple_threshold=s.staple_threshold,
        order=order,
        pairs=[tuple(p) for p in s.pairs] or None,
        whisker_mode=a.whisker_mode.value,
        estimator=a.estimator.value,
        bar_fill=a.bar_fill.value,
        palette=palette,
        rotate_xticks=a.rotate_xticks,
        y_zero=a.y_zero,
    )
    return RenderResult(ax_out.figure, resolved_test, omnibus, pairwise, normality)


def render_bytes(spec: PlotSpec, df: pd.DataFrame, fmt: str = "pdf") -> bytes:
    """Render a spec and return the figure encoded as ``fmt`` (pdf/png/svg)."""
    res = render(spec, df)
    buf = io.BytesIO()
    res.figure.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(res.figure)
    return buf.getvalue()
