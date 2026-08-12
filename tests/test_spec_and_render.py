"""First-pass tests for the spec-driven pipeline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

import grism
from grism.spec import (
    AppearanceSpec,
    DataBinding,
    Element,
    PlotSpec,
    PlotTemplate,
    SPEC_VERSION,
    StatTest,
    migrate,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("ctrl", 10.0), ("low", 12.0), ("high", 16.0)]:
        for _ in range(8):
            rows.append({"treatment": g, "tumour_volume": float(rng.normal(mu, 2.0)), "dose": mu})
    return pd.DataFrame(rows)


def _spec() -> PlotSpec:
    return PlotSpec(
        name="demo",
        data=DataBinding(x="treatment", y="tumour_volume",
                         order=["ctrl", "low", "high"]),
    )


def test_spec_json_roundtrip():
    spec = _spec()
    restored = PlotSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.data.x == "treatment"
    assert restored.appearance.elements[0] is Element.strip


def test_template_bind_roundtrip():
    tmpl = _spec().to_template()
    assert not hasattr(tmpl, "data")
    bound = tmpl.bind(DataBinding(x="treatment", y="tumour_volume"))
    assert bound.data.y == "tumour_volume"
    # template survives JSON too
    assert PlotTemplate.from_json(tmpl.to_json()) == tmpl


def test_migrate_stamps_current_version():
    raw = {"name": "old", "data": {"x": "g", "y": "v"}}  # no spec_version
    migrated = migrate(raw)
    assert migrated["spec_version"] == SPEC_VERSION
    spec = PlotSpec.from_json(PlotSpec.model_validate(migrated).to_json())
    assert spec.spec_version == SPEC_VERSION


def test_render_returns_figure_and_stats(df):
    res = grism.render(_spec(), df)
    assert res.figure is not None
    assert res.resolved_test in {"t_test", "mann_whitney"}
    assert len(res.pairwise) == 3  # 3 groups -> 3 pairwise comparisons
    assert set(res.normality) == {"ctrl", "low", "high"}


def test_render_bytes_formats(df):
    for fmt in ("pdf", "png", "svg"):
        data = grism.render_bytes(_spec(), df, fmt=fmt)
        assert isinstance(data, bytes) and len(data) > 0


def test_auto_test_selection_matches_manual(df):
    spec = _spec()
    normality = grism.group_normality(df, "treatment", "tumour_volume",
                                      ["ctrl", "low", "high"])
    expected = grism.pick_test_by_normality(normality)
    assert grism.render(spec, df).resolved_test == expected


def test_explicit_test_overrides_auto(df):
    spec = _spec()
    spec.stats.test = StatTest.mann_whitney
    assert grism.render(spec, df).resolved_test == "mann_whitney"


def test_row_filter_applied(df):
    spec = _spec()
    spec.data.row_filter = "dose > 10.5"  # drops ctrl (mu=10)
    _, order = grism.prepare_data(spec, df)
    assert "ctrl" not in order
    assert set(order) == {"low", "high"}


def test_staple_threshold_default_is_significant_only():
    from grism.spec import StatsSpec

    assert StatsSpec().staple_threshold == 0.05


def test_staple_threshold_filters_drawn_staples():
    # a,b nearly identical (non-significant); c far away (significant vs both)
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("a", 10.0), ("b", 10.1), ("c", 30.0)]:
        for _ in range(10):
            rows.append({"x": g, "y": float(rng.normal(mu, 1.0))})
    d = pd.DataFrame(rows)
    spec = PlotSpec(name="t", data=DataBinding(x="x", y="y", order=["a", "b", "c"]))

    # default threshold 0.05: only significant pairs get a staple (each adds one text)
    res = grism.render(spec, d)
    sig = [p for p in res.pairwise if p.pvalue < 0.05]
    assert len(sig) < 3  # a-b is not significant
    assert len(res.figure.axes[0].texts) == len(sig)
    assert len(res.pairwise) == 3  # returned stats are still complete

    # threshold None draws every pair
    spec.stats.staple_threshold = None
    res_all = grism.render(spec, d)
    assert len(res_all.figure.axes[0].texts) == 3


def test_wide_form_melt():
    wide = pd.DataFrame({"id": [1, 2, 3], "A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    spec = PlotSpec(
        name="w",
        data=DataBinding(x="group", y="value", wide_form=True, wide_id_cols=["id"]),
    )
    df_plot, order = grism.prepare_data(spec, wide)
    assert set(order) == {"A", "B"}
    assert {"group", "value"} <= set(df_plot.columns)


def test_template_reuse_across_datasets(df):
    """One template, two datasets -> both render."""
    tmpl = PlotTemplate(
        name="house-style",
        appearance=AppearanceSpec(elements=[Element.bar, Element.strip], palette="Set2"),
    )
    binding = DataBinding(x="treatment", y="tumour_volume")
    df2 = df.assign(tumour_volume=df["tumour_volume"] * 2)
    for data in (df, df2):
        assert len(grism.render_bytes(tmpl.bind(binding), data, fmt="png")) > 0
