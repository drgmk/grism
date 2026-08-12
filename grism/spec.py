"""Declarative plot specification for grism.

A :class:`PlotSpec` fully describes a plot as data: what to draw, how to style
it, and which stats to run. It is the single source of truth shared by the UI,
the CLI, and the HTTP API. A :class:`PlotTemplate` is a spec minus its
dataset-dependent :class:`DataBinding`, which is how "plots we make the same
way" are reused across datasets.

Everything here is pure: no matplotlib, no filesystem, no Streamlit.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

SPEC_VERSION = 1


# --- enums (replace the magic strings used throughout the old config) -------


class Element(str, Enum):
    strip = "strip"
    bar = "bar"
    whisker = "whisker"
    hist = "hist"


class WhiskerMode(str, Enum):
    quartiles = "quartiles"
    mean_std = "mean-std"


class Estimator(str, Enum):
    median = "median"
    mean = "mean"


class BarFill(str, Enum):
    block = "block"
    transparent = "transparent"
    none = "none"


class StatTest(str, Enum):
    auto = "auto"  # normality-driven pick (the UI's current default)
    t_test = "t_test"
    mann_whitney = "mann_whitney"
    anova = "anova"
    kruskal = "kruskal_wallis"


# --- sub-specs --------------------------------------------------------------


class DataBinding(BaseModel):
    """The only dataset-dependent part of a spec. Swap it to reuse a template.

    Field names follow seaborn: ``x`` is the categorical/group column, ``y`` is
    the numeric value column, ``hue`` and ``order`` mean what they do in
    seaborn — so anything you'd look up in the seaborn docs applies here.
    """

    x: str  # categorical (group) column, as in seaborn
    y: str  # numeric (value) column, as in seaborn
    hue: Optional[str] = None
    order: List[str] = Field(default_factory=list)  # x-category display order
    wide_form: bool = False
    wide_id_cols: List[str] = Field(default_factory=list)
    # Row selection as a pandas .query() predicate, e.g. "dose > 0". Unlike the
    # old positional boolean mask, this survives being applied to new data.
    row_filter: Optional[str] = None


class StatsSpec(BaseModel):
    test: StatTest = StatTest.auto
    pairs: List[Tuple[str, str]] = Field(default_factory=list)  # which staples
    staple_scale: float = 1.0
    # Draw only staples with pvalue < threshold (default: significant only).
    # None draws every selected pair regardless of significance.
    staple_threshold: Optional[float] = 0.05


class LabelSpec(BaseModel):
    # None means "use the auto-generated default"; a string overrides it.
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None


class AppearanceSpec(BaseModel):
    # `palette` and `estimator` follow seaborn; `elements`, `whisker_mode`,
    # `bar_fill` and the scale knobs are grism's overlay/print extras.
    elements: List[Element] = Field(
        default_factory=lambda: [Element.strip, Element.bar, Element.whisker]
    )
    whisker_mode: WhiskerMode = WhiskerMode.quartiles
    estimator: Estimator = Estimator.median  # seaborn bar estimator
    bar_fill: BarFill = BarFill.block
    use_group_colors: bool = True
    palette: str = "tab10"  # seaborn palette name
    style: str = "default"
    rotate_xticks: bool = False
    y_zero: bool = True
    plot_scale: float = 1.0
    x_scale: float = 1.0
    y_scale: float = 1.0


# --- top-level spec + template ----------------------------------------------


class PlotSpec(BaseModel):
    spec_version: int = SPEC_VERSION
    name: str = "Plot 1"
    data: DataBinding
    appearance: AppearanceSpec = Field(default_factory=AppearanceSpec)
    stats: StatsSpec = Field(default_factory=StatsSpec)
    labels: LabelSpec = Field(default_factory=LabelSpec)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PlotSpec":
        return cls.model_validate(migrate(json.loads(text)))

    def to_template(self) -> "PlotTemplate":
        return PlotTemplate(
            spec_version=self.spec_version,
            name=self.name,
            appearance=self.appearance,
            stats=self.stats,
            labels=self.labels,
        )


class PlotTemplate(BaseModel):
    """A spec minus its :class:`DataBinding` — the reusable, dataset-free part."""

    spec_version: int = SPEC_VERSION
    name: str = "Template"
    appearance: AppearanceSpec = Field(default_factory=AppearanceSpec)
    stats: StatsSpec = Field(default_factory=StatsSpec)
    labels: LabelSpec = Field(default_factory=LabelSpec)

    def bind(self, data: DataBinding) -> PlotSpec:
        return PlotSpec(
            spec_version=self.spec_version,
            name=self.name,
            data=data,
            appearance=self.appearance,
            stats=self.stats,
            labels=self.labels,
        )

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PlotTemplate":
        return cls.model_validate(migrate(json.loads(text)))


# --- versioning -------------------------------------------------------------


def migrate(raw: dict) -> dict:
    """Bring a raw spec/template dict up to the current schema version.

    Add one ``_vN_to_vN+1`` step per schema change and chain it here so that
    specs saved by older grism versions keep loading.
    """
    raw = dict(raw)
    version = int(raw.get("spec_version", SPEC_VERSION))
    # (no migrations yet; v1 is the first version)
    # if version < 2:
    #     raw = _v1_to_v2(raw)
    #     version = 2
    raw["spec_version"] = SPEC_VERSION
    return raw


def json_schema() -> dict[str, Any]:
    """JSON Schema for :class:`PlotSpec`, for a future JS frontend to consume."""
    return PlotSpec.model_json_schema()
