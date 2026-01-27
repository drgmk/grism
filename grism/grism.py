"""Core plotting and stats helpers for grism."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sps


@dataclass
class StatResult:
    test: str
    groups: List[str]
    statistic: float
    pvalue: float


@dataclass
class PairwiseStatResult:
    test: str
    group_a: str
    group_b: str
    statistic: float
    pvalue: float


class GrismError(Exception):
    """Base error for grism."""


def _default_style_path() -> Path:
    # Package lives at grism/grism.py; data/ is one level above the package dir.
    return Path(__file__).resolve().parents[1] / "data" / "default.mplstyle"


def _resolve_style(style: Optional[str]) -> Optional[str]:
    if not style or style == "default":
        default_path = _default_style_path()
        return str(default_path) if default_path.exists() else None
    return style


def _compute_figsize(n_groups: int, scale: float = 2.0) -> Tuple[float, float]:
    # 3h x 2w for one group, add 1w per extra group.
    width = 2.0 + max(n_groups - 1, 0) * 1.0
    height = 5.0
    return width * scale, height * scale

def _validate_columns(df: pd.DataFrame, value: str, group: Optional[str], hue: Optional[str]) -> None:
    missing = [c for c in [value, group, hue] if c and c not in df.columns]
    if missing:
        raise GrismError(f"Missing columns: {', '.join(missing)}")


def _group_order(df: pd.DataFrame, group: Optional[str]) -> List[str]:
    if not group:
        return []
    # Preserve input order as much as possible.
    series = df[group]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [g for g in series.cat.categories if g in set(series.unique())]
    return list(series.unique())


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
    value: str,
    group: Optional[str] = None,
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
) -> plt.Axes:
    """Draw a plot with one or more elements.

    elements can include: strip, bar, whisker, hist
    """
    _validate_columns(df, value, group, hue)
    elements = list(elements)

    observed_groups = _group_order(df, group) if group else []
    groups = list(order) if order else observed_groups
    n_groups = max(len(groups) if groups else len(observed_groups), 1)
    resolved_style = _resolve_style(style)
    fig_size = figsize or _compute_figsize(n_groups)

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
            if "bar" in elements or "whisker" in elements:
                errorbar = "se" if "whisker" in elements else None
                sns.barplot(
                    data=df,
                    x=group,
                    y=value,
                    hue=hue,
                    order=groups if group else None,
                    errorbar=errorbar,
                    capsize=0.15 if "whisker" in elements else 0,
                    ax=ax,
                    dodge=bool(hue),
                    zorder=1,
                    alpha=0.9,
                )

            if "strip" in elements:
                sns.stripplot(
                    data=df,
                    x=group,
                    y=value,
                    hue=hue if ("bar" not in elements and "whisker" not in elements) else None,
                    order=groups if group else None,
                    dodge=bool(hue),
                    jitter=0.2,
                    alpha=0.8,
                    ax=ax,
                    zorder=2,
                )

            # Avoid duplicate legends when overlaying elements.
            if hue and ("bar" in elements or "whisker" in elements) and "strip" in elements:
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
    return ax


def _run_two_group_test(a: np.ndarray, b: np.ndarray, test: str) -> Tuple[str, float, float]:
    test_key = test.lower()
    if test_key in {"t_test", "ttest", "t-test", "anova", "oneway_anova"}:
        stat, p = sps.ttest_ind(a, b, equal_var=False)
        return "t_test", float(stat), float(p)
    if test_key in {"mann_whitney", "mannwhitney", "u_test", "kruskal", "kruskal_wallis", "kruskal-wallis"}:
        stat, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        return "mann_whitney", float(stat), float(p)
    raise GrismError(f"Unknown test: {test}")


def stats(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    test: str = "t_test",
) -> Union[StatResult, List[PairwiseStatResult]]:
    """Run a test across groups.

    For two-group tests with >2 groups, return all pairwise results.
    """
    _validate_columns(df, value, group, None)
    groups = _group_order(df, group)
    if len(groups) < 2:
        raise GrismError("Need at least two groups for a statistical test.")

    data_by_group = [df.loc[df[group] == g, value].dropna().values for g in groups]

    test_key = test.lower()
    if test_key in {"t_test", "ttest", "t-test"}:
        if len(groups) != 2:
            return pairwise_stats(df, value=value, group=group, test=test)
        stat, p = sps.ttest_ind(data_by_group[0], data_by_group[1], equal_var=False)
        return StatResult("t_test", groups, float(stat), float(p))

    if test_key in {"mann_whitney", "mannwhitney", "u_test"}:
        if len(groups) != 2:
            return pairwise_stats(df, value=value, group=group, test=test)
        stat, p = sps.mannwhitneyu(data_by_group[0], data_by_group[1], alternative="two-sided")
        return StatResult("mann_whitney", groups, float(stat), float(p))

    if test_key in {"anova", "oneway_anova"}:
        stat, p = sps.f_oneway(*data_by_group)
        return StatResult("anova", groups, float(stat), float(p))

    if test_key in {"kruskal", "kruskal_wallis", "kruskal-wallis"}:
        stat, p = sps.kruskal(*data_by_group)
        return StatResult("kruskal_wallis", groups, float(stat), float(p))

    raise GrismError(f"Unknown test: {test}")


def pairwise_stats(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    test: str = "t_test",
) -> List[PairwiseStatResult]:
    """Run pairwise tests for all group combinations."""
    _validate_columns(df, value, group, None)
    groups = _group_order(df, group)
    if len(groups) < 2:
        raise GrismError("Need at least two groups for a statistical test.")

    results: List[PairwiseStatResult] = []
    for group_a, group_b in combinations(groups, 2):
        a = df.loc[df[group] == group_a, value].dropna().values
        b = df.loc[df[group] == group_b, value].dropna().values
        test_name, stat, p = _run_two_group_test(a, b, test)
        results.append(PairwiseStatResult(test_name, group_a, group_b, stat, p))
    return results


def _annotate_pairwise(
    ax: plt.Axes,
    pairs: Sequence[PairwiseStatResult],
    groups: Sequence[str],
    staple_scale: float = 1.0,
) -> None:
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
    value: str,
    group: str,
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
    order: Optional[Sequence[str]] = None,
) -> Tuple[plt.Axes, Optional[StatResult], List[PairwiseStatResult]]:
    ax = plot(
        df,
        value=value,
        group=group,
        hue=hue,
        elements=elements,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        style=style,
        figsize=figsize,
        order=order,
    )

    groups = list(order) if order else _group_order(df, group)
    if len(groups) < 2:
        return ax, None, []

    stats_out = stats(df, value=value, group=group, test=test)
    if isinstance(stats_out, list):
        omnibus = None
        pairs = stats_out
    else:
        omnibus = stats_out
        pairs = pairwise_stats(df, value=value, group=group, test=test)

    _annotate_pairwise(ax, pairs, groups, staple_scale=staple_scale)
    return ax, omnibus, pairs
