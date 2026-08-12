"""grism: simple plotting + stats for biology-style figures."""

from .core import (
    compute_figsize,
    compute_figsize_for,
    plot,
    plot_with_stats,
)
from .figure import (
    FigureSpec,
    figure_from_columns,
    render_page_images,
    render_pages,
    render_pdf_bytes,
)
from .library import list_templates, load_template
from .render import RenderResult, prepare_data, render, render_bytes
from .spec import (
    AppearanceSpec,
    BarFill,
    Estimator,
    DataBinding,
    Element,
    LabelSpec,
    PlotSpec,
    PlotTemplate,
    SPEC_VERSION,
    StatsSpec,
    StatTest,
    WhiskerMode,
    json_schema,
    migrate,
)
from .stats import (
    GrismError,
    PairwiseStatResult,
    StatResult,
    group_normality,
    pairwise_stats,
    pick_test_by_normality,
    stats,
)

__all__ = [
    # core plotting
    "compute_figsize",
    "compute_figsize_for",
    "plot",
    "plot_with_stats",
    # render seam
    "RenderResult",
    "prepare_data",
    "render",
    "render_bytes",
    # multi-panel figures
    "FigureSpec",
    "figure_from_columns",
    "render_pages",
    "render_pdf_bytes",
    "render_page_images",
    # template library
    "list_templates",
    "load_template",
    # spec
    "AppearanceSpec",
    "BarFill",
    "DataBinding",
    "Estimator",
    "Element",
    "LabelSpec",
    "PlotSpec",
    "PlotTemplate",
    "SPEC_VERSION",
    "StatsSpec",
    "StatTest",
    "WhiskerMode",
    "json_schema",
    "migrate",
    # stats
    "GrismError",
    "PairwiseStatResult",
    "StatResult",
    "group_normality",
    "pairwise_stats",
    "pick_test_by_normality",
    "stats",
]
