"""grism: simple plotting + stats for biology-style figures."""

from .grism import (
    PairwiseStatResult,
    StatResult,
    compute_figsize,
    pairwise_stats,
    plot,
    plot_with_stats,
    stats,
)

__all__ = [
    "PairwiseStatResult",
    "StatResult",
    "compute_figsize",
    "pairwise_stats",
    "plot",
    "plot_with_stats",
    "stats",
]
