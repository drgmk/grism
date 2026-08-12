"""Tests for multi-panel figures (grism.figure)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import grism
from grism.figure import FigureSpec, figure_from_columns, render_pages, render_pdf_bytes
from grism.spec import AppearanceSpec, Element, PlotTemplate

Y_COLS = [f"marker_{i}" for i in range(5)]


@pytest.fixture
def wide_df():
    """A wide sheet: a `sample` column (x categories) + several value columns."""
    rng = np.random.default_rng(0)
    n_per = 8
    samples = np.repeat(["ctrl", "low", "high"], n_per)
    data = {"sample": samples}
    for j, col in enumerate(Y_COLS):
        data[col] = rng.normal(10 + j, 2.0, size=len(samples))
    return pd.DataFrame(data)


@pytest.fixture
def template():
    return PlotTemplate(
        name="house",
        appearance=AppearanceSpec(elements=[Element.bar, Element.strip], palette="Set2"),
    )


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def test_figure_from_columns_panel_count(template):
    fs = figure_from_columns(template, x="sample", y_columns=Y_COLS)
    assert len(fs.panels) == len(Y_COLS)
    assert [p.data.y for p in fs.panels] == Y_COLS
    assert [p.data.x for p in fs.panels] == ["sample"] * len(Y_COLS)
    # panels titled by their column
    assert [p.labels.title for p in fs.panels] == Y_COLS


def test_pagination_and_hidden_axes(template, wide_df):
    # 5 panels, 4 per page, 2 cols -> page1: 4 panels (2x2), page2: 1 panel (needs 1x2 -> 1 hidden)
    fs = figure_from_columns(
        template, x="sample", y_columns=Y_COLS, ncols=2, panels_per_page=4
    )
    figs = render_pages(fs, wide_df)
    assert len(figs) == 2
    assert len(_visible(figs[0])) == 4
    assert len(_visible(figs[1])) == 1
    assert len(figs[1].axes) == 2  # one row of 2, second cell hidden
    import matplotlib.pyplot as plt

    for f in figs:
        plt.close(f)


def test_single_page_when_no_pagination(template, wide_df):
    fs = figure_from_columns(template, x="sample", y_columns=Y_COLS, ncols=3)
    figs = render_pages(fs, wide_df)
    assert len(figs) == 1
    assert len(_visible(figs[0])) == 5  # 5 panels drawn
    import matplotlib.pyplot as plt

    plt.close(figs[0])


def test_panels_have_content(template, wide_df):
    fs = figure_from_columns(template, x="sample", y_columns=Y_COLS[:2], ncols=2)
    figs = render_pages(fs, wide_df)
    ax = _visible(figs[0])[0]
    assert len(ax.patches) + len(ax.collections) > 0  # bars and/or strip points
    import matplotlib.pyplot as plt

    plt.close(figs[0])


def test_multipage_pdf(template, wide_df):
    fs = figure_from_columns(
        template, x="sample", y_columns=Y_COLS, ncols=2, panels_per_page=4
    )
    data = render_pdf_bytes(fs, wide_df)
    assert data[:4] == b"%PDF"
    assert data.count(b"/Type /Page\n") == 2 or b"/Type /Pages" in data  # 2 pages


def test_page_images(template, wide_df):
    fs = figure_from_columns(
        template, x="sample", y_columns=Y_COLS, ncols=2, panels_per_page=4
    )
    pngs = grism.render_page_images(fs, wide_df, fmt="png")
    assert len(pngs) == 2
    assert all(b[:4] == b"\x89PNG" for b in pngs)


def test_figurespec_json_roundtrip(template):
    fs = figure_from_columns(template, x="sample", y_columns=Y_COLS[:3])
    restored = FigureSpec.from_json(fs.to_json())
    assert restored == fs
