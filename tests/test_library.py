"""Tests for the built-in template library."""

from __future__ import annotations

import pytest

import grism
from grism.spec import PlotTemplate


def test_list_templates_includes_defaults():
    names = grism.list_templates()
    assert "default" in names
    assert "bars" in names


def test_load_template_returns_template():
    tmpl = grism.load_template("default")
    assert isinstance(tmpl, PlotTemplate)
    # a template has no data binding
    assert not hasattr(tmpl, "data")


def test_load_unknown_template_raises():
    with pytest.raises(FileNotFoundError):
        grism.load_template("does-not-exist")


def test_builtin_template_binds_and_renders():
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    from grism.spec import DataBinding

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "g": np.repeat(["a", "b"], 6),
        "v": rng.normal(0, 1, 12),
    })
    spec = grism.load_template("bars").bind(DataBinding(x="g", y="v"))
    assert len(grism.render_bytes(spec, df, fmt="png")) > 0
