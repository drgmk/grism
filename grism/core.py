"""Core plotting for grism.

Turns a DataFrame + drawing options into a matplotlib Axes. Statistics live in
:mod:`grism.stats`; the spec-driven entry point is :mod:`grism.render`.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .stats import (
    GrismError,
    PairwiseStatResult,
    StatResult,
    group_order,
    pairwise_stats,
    stats,
    validate_columns,
)

__all__ = [
    "GrismError",
    "PairwiseStatResult",
    "StatResult",
    "compute_figsize",
    "compute_figsize_for",
    "plot",
    "plot_with_stats",
]


def _default_style_path() -> Path:
    # Package lives at grism/core.py; data/ is alongside it.
    return Path(__file__).resolve().parent / "data" / "default.mplstyle"


def _resolve_style(style: Optional[str]) -> Optional[str]:
    if not style or style == "default":
        default_path = _default_style_path()
        return str(default_path) if default_path.exists() else None
    return style


def compute_figsize(n_groups: int, scale: float = 2.0) -> Tuple[float, float]:
    # 5h x 2w for one group, add 1w per extra group.
    width = 2.0 + max(n_groups - 1, 0) * 1.0
    height = 5.0
    return width * scale, height * scale


def compute_figsize_for(order: Sequence[str], appearance) -> Tuple[float, float]:
    """Figure size for a spec: base size scaled by the appearance scale knobs.

    ``appearance`` is a :class:`grism.spec.AppearanceSpec` (duck-typed to avoid
    importing spec here and creating a cycle).
    """
    n_groups = max(len(order), 1)
    base_w, base_h = compute_figsize(n_groups, scale=0.7)
    plot_scale = getattr(appearance, "plot_scale", 1.0)
    x_scale = getattr(appearance, "x_scale", 1.0)
    y_scale = getattr(appearance, "y_scale", 1.0)
    return base_w * plot_scale * x_scale, base_h * plot_scale * y_scale


def _resolve_palette(
    palette: Optional[str], groups: Sequence[str]
) -> Optional[List[Tuple[float, float, float]]]:
    if not palette or not groups:
        return None
    try:
        colors = sns.color_palette(palette, n_colors=len(groups))
    except Exception:
        return None
    return list(colors)


def _bar_estimator(mode: str):
    return np.median if mode == "median" else np.mean


def _whisker_stats(values: np.ndarray, mode: str) -> Tuple[float, float, float]:
    if mode == "quartiles":
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        return float(q1), float(med), float(q3)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return mean - std, mean, mean + std


def _format_pvalue(p: float) -> str:
    if p < 0.0001:
        return "p<0.0001"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"p={p:.3f}"


def _add_staple(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", fontsize=8)


def plot(
    df: pd.DataFrame,
    *,
    y: str,
    x: Optional[str] = None,
    hue: Optional[str] = None,
    elements: Iterable[str] = ("strip", "bar", "whisker"),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    seed: int = 0,
    style: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    order: Optional[Sequence[str]] = None,
    whisker_mode: str = "mean-std",
    estimator: str = "mean",
    bar_fill: str = "block",
    palette: Optional[str] = None,
    rotate_xticks: bool = False,
    y_zero: bool = False,
) -> plt.Axes:
    """Draw a plot with one or more elements.

    Uses seaborn's vocabulary: ``x`` (categorical/group), ``y`` (numeric),
    ``hue``, ``order``, ``palette``, ``estimator``. elements can include:
    strip, bar, whisker, hist.
    """
    # Internal aliases keep the body in grism's original terms.
    value, group, group_palette, bar_mode = y, x, palette, estimator
    validate_columns(df, value, group, hue)
    elements = list(elements)

    observed_groups = group_order(df, group) if group else []
    groups = list(order) if order else observed_groups
    n_groups = max(len(groups) if groups else len(observed_groups), 1)
    resolved_style = _resolve_style(style)
    fig_size = figsize or compute_figsize(n_groups)
    palette = _resolve_palette(group_palette, groups)

    style_ctx = plt.style.context(resolved_style) if resolved_style else nullcontext()
    with style_ctx:
        if ax is None:
            _, ax = plt.subplots(figsize=fig_size)

        # Jitter is handled internally by seaborn; keep seed for future use.
        _ = np.random.default_rng(seed)

        if "hist" in elements:
            hist_hue = hue or group
            sns.histplot(
                data=df,
                x=value,
                hue=hist_hue,
                multiple="layer",
                alpha=0.5,
                edgecolor="white",
                ax=ax,
            )
        else:
            if "bar" in elements:
                estimator = _bar_estimator(bar_mode)
                # With no real hue, colour by the x categories via hue=group +
                # legend=False (seaborn's replacement for a bare `palette`).
                color_by_x = hue is None and palette is not None and group is not None
                sns.barplot(
                    data=df,
                    x=group,
                    y=value,
                    hue=group if color_by_x else hue,
                    order=groups if group else None,
                    estimator=estimator,
                    errorbar=None,
                    ax=ax,
                    dodge=bool(hue),
                    zorder=1,
                    alpha=0.9,
                    palette=palette if color_by_x else None,
                    legend=not color_by_x,
                )
                if bar_fill == "none":
                    for patch in ax.patches:
                        patch.set_facecolor("none")
                        patch.set_alpha(1.0)
                        patch.set_edgecolor(patch.get_edgecolor() or "black")
                        patch.set_linewidth(1.2)
                elif bar_fill == "transparent":
                    for patch in ax.patches:
                        patch.set_alpha(0.5)

            if "whisker" in elements and group:
                cap = 0.06
                for idx, g in enumerate(groups):
                    vals = df.loc[df[group] == g, value].dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    low, mid, high = _whisker_stats(vals, whisker_mode)
                    color = palette[idx] if palette else "black"
                    ax.vlines(idx, low, high, color=color, lw=1.2, zorder=2)
                    ax.hlines(mid, idx - 0.12, idx + 0.12, color=color, lw=1.6, zorder=3)
                    ax.hlines(low, idx - cap, idx + cap, color=color, lw=1.2, zorder=2)
                    ax.hlines(high, idx - cap, idx + cap, color=color, lw=1.2, zorder=2)

            if "strip" in elements:
                strip_hue = hue if ("bar" not in elements) else None
                color_by_x = strip_hue is None and palette is not None and group is not None
                sns.stripplot(
                    data=df,
                    x=group,
                    y=value,
                    hue=group if color_by_x else strip_hue,
                    order=groups if group else None,
                    dodge=bool(hue),
                    jitter=0.2,
                    alpha=0.8,
                    ax=ax,
                    zorder=2,
                    palette=palette if color_by_x else None,
                    legend=not color_by_x,
                )

            # Avoid duplicate legends when overlaying elements.
            if hue and ("bar" in elements) and "strip" in elements:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    keep = len(dict.fromkeys(labels))
                    ax.legend(handles[:keep], labels[:keep], title=hue)

    ax.set_title(title or "")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif group:
        ax.set_xlabel(group)
    ax.set_ylabel(ylabel or value)
    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    if y_zero:
        _, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)
    return ax


def _annotate_pairwise(
    ax: plt.Axes,
    pairs: Sequence[PairwiseStatResult],
    groups: Sequence[str],
    staple_scale: float = 1.0,
    threshold: Optional[float] = None,
) -> None:
    if threshold is not None:
        pairs = [p for p in pairs if p.pvalue < threshold]
    if not pairs or not groups:
        return

    group_to_x = {g: i for i, g in enumerate(groups)}
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-9)

    scale = max(staple_scale, 0.2)
    base_y = y_max + y_span * 0.05 * scale
    h = y_span * 0.025 * scale
    step = y_span * 0.07 * scale

    # Draw shorter spans first to reduce overlaps.
    sorted_pairs = sorted(
        pairs,
        key=lambda r: abs(group_to_x[r.group_b] - group_to_x[r.group_a]),
    )

    top_y = base_y
    for i, res in enumerate(sorted_pairs):
        x1 = group_to_x[res.group_a]
        x2 = group_to_x[res.group_b]
        y = base_y + i * step
        _add_staple(ax, x1, x2, y, h, _format_pvalue(res.pvalue))
        top_y = max(top_y, y + h)

    # Leave headroom for the p-value text above the top staple.
    ax.set_ylim(y_min, top_y + step)


def plot_with_stats(
    df: pd.DataFrame,
    *,
    y: str,
    x: str,
    hue: Optional[str] = None,
    elements: Iterable[str] = ("strip", "bar", "whisker"),
    test: str = "t_test",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    style: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    staple_scale: float = 1.0,
    staple_threshold: Optional[float] = 0.05,
    order: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    whisker_mode: str = "mean-std",
    estimator: str = "mean",
    bar_fill: str = "block",
    palette: Optional[str] = None,
    rotate_xticks: bool = False,
    y_zero: bool = False,
) -> Tuple[plt.Axes, Optional[StatResult], List[PairwiseStatResult]]:
    """Draw plot + stats. ``staple_threshold`` draws only staples with
    ``pvalue < threshold`` (default 0.05 = significant only); pass ``None`` to
    draw every selected pair. The returned pairwise list is always complete."""
    value, group = y, x  # internal aliases (see plot())
    ax = plot(
        df,
        y=y,
        x=x,
        hue=hue,
        elements=elements,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        style=style,
        figsize=figsize,
        order=order,
        whisker_mode=whisker_mode,
        estimator=estimator,
        bar_fill=bar_fill,
        palette=palette,
        rotate_xticks=rotate_xticks,
        y_zero=y_zero,
    )

    groups = list(order) if order else group_order(df, group)
    if len(groups) < 2:
        return ax, None, []

    stats_out = stats(df, value=value, group=group, test=test)
    if isinstance(stats_out, list):
        omnibus = None
        pairs_out = stats_out
    else:
        omnibus = stats_out
        pairs_out = pairwise_stats(df, value=value, group=group, test=test)

    if pairs is not None:
        pair_map = {(p.group_a, p.group_b): p for p in pairs_out}
        ordered: List[PairwiseStatResult] = []
        for a, b in pairs:
            if (a, b) in pair_map:
                ordered.append(pair_map[(a, b)])
            elif (b, a) in pair_map:
                ordered.append(pair_map[(b, a)])
        pairs_out = ordered

    _annotate_pairwise(ax, pairs_out, groups, staple_scale=staple_scale, threshold=staple_threshold)
    return ax, omnibus, pairs_out
