"""Statistics for grism: group tests, pairwise tests, and normality-driven
test selection.

Pure data/stats only — no matplotlib. Both the plotting core and the render
seam depend on this module so the UI, CLI, and API pick the same test for the
same spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
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


def validate_columns(
    df: pd.DataFrame, value: str, group: Optional[str], hue: Optional[str]
) -> None:
    missing = [c for c in [value, group, hue] if c and c not in df.columns]
    if missing:
        raise GrismError(f"Missing columns: {', '.join(missing)}")


def group_order(df: pd.DataFrame, group: Optional[str]) -> List[str]:
    if not group:
        return []
    # Preserve input order as much as possible.
    series = df[group]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [g for g in series.cat.categories if g in set(series.unique())]
    return list(series.unique())


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
    validate_columns(df, value, group, None)
    groups = group_order(df, group)
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
    validate_columns(df, value, group, None)
    groups = group_order(df, group)
    if len(groups) < 2:
        raise GrismError("Need at least two groups for a statistical test.")

    results: List[PairwiseStatResult] = []
    for group_a, group_b in combinations(groups, 2):
        a = df.loc[df[group] == group_a, value].dropna().values
        b = df.loc[df[group] == group_b, value].dropna().values
        test_name, stat, p = _run_two_group_test(a, b, test)
        results.append(PairwiseStatResult(test_name, group_a, group_b, stat, p))
    return results


# --- normality-driven test selection (moved out of the Streamlit UI) --------


def group_normality(
    df: pd.DataFrame, group: str, value: str, order: Sequence[str]
) -> Dict[str, Optional[float]]:
    """Shapiro-Wilk p-value per group. <3 points -> assume normal (p=1)."""
    pvals: Dict[str, Optional[float]] = {}
    for g in order:
        vals = df.loc[df[group] == g, value].dropna().to_numpy()
        if vals.size < 3:
            pvals[g] = 1.0
            continue
        try:
            _stat, p = sps.shapiro(vals)
        except Exception:
            pvals[g] = None
            continue
        pvals[g] = float(p)
    return pvals


def normality_label(p: Optional[float]) -> str:
    if p is None:
        return "error"
    return "yes" if p >= 0.05 else "no"


def pick_test_by_normality(normality: Dict[str, Optional[float]]) -> str:
    """t-test if every group looks normal, else Mann-Whitney."""
    labels = [normality_label(p) for p in normality.values()]
    if labels and all(lbl == "yes" for lbl in labels):
        return "t_test"
    return "mann_whitney"
