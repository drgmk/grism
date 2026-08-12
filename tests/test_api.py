"""Tests for the HTTP API (in-process, via FastAPI TestClient)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from grism.api import app
from grism.figure import figure_from_columns
from grism.spec import AppearanceSpec, DataBinding, Element, PlotSpec, PlotTemplate

client = TestClient(app)


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("ctrl", 10.0), ("low", 12.0), ("high", 16.0)]:
        for _ in range(8):
            rows.append({"treatment": g, "tumour_volume": float(rng.normal(mu, 2.0))})
    return pd.DataFrame(rows).to_dict("records")


def _spec() -> PlotSpec:
    return PlotSpec(name="demo", data=DataBinding(x="treatment", y="tumour_volume"))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_schema_endpoint():
    props = client.get("/schema").json()["$defs"]["DataBinding"]["properties"]
    assert "x" in props and "y" in props


@pytest.mark.parametrize("fmt,media,magic", [
    ("png", "image/png", b"\x89PNG"),
    ("pdf", "application/pdf", b"%PDF"),
    ("svg", "image/svg+xml", b"<?xml"),
])
def test_render_formats(records, fmt, media, magic):
    r = client.post("/render", json={"spec": _spec().model_dump(), "data": records, "fmt": fmt})
    assert r.status_code == 200
    assert r.headers["content-type"] == media
    assert r.content[:5].startswith(magic)


def test_apply_endpoint(records):
    tmpl = PlotTemplate(name="house", appearance=AppearanceSpec(elements=[Element.bar]))
    r = client.post("/apply", json={
        "template": tmpl.model_dump(),
        "binding": DataBinding(x="treatment", y="tumour_volume").model_dump(),
        "data": records,
        "fmt": "png",
    })
    assert r.status_code == 200
    assert r.content[:4] == b"\x89PNG"


def test_empty_data_422():
    r = client.post("/render", json={"spec": _spec().model_dump(), "data": [], "fmt": "png"})
    assert r.status_code == 422


def test_bad_column_422(records):
    bad = PlotSpec(name="bad", data=DataBinding(x="nope", y="tumour_volume"))
    r = client.post("/render", json={"spec": bad.model_dump(), "data": records, "fmt": "png"})
    assert r.status_code == 422
    assert "Render failed" in r.json()["detail"]


def _wide_records():
    rng = np.random.default_rng(0)
    samples = np.repeat(["ctrl", "low", "high"], 8)
    df = pd.DataFrame({"sample": samples})
    for i in range(5):
        df[f"marker_{i}"] = rng.normal(10 + i, 2.0, size=len(samples))
    return df.to_dict("records")


def _fig_spec():
    tmpl = PlotTemplate(name="panel", appearance=AppearanceSpec(elements=[Element.bar]))
    return figure_from_columns(
        tmpl, x="sample", y_columns=[f"marker_{i}" for i in range(5)],
        ncols=2, panels_per_page=4,
    )


def test_render_figure_pdf():
    r = client.post("/render-figure", json={
        "figure": _fig_spec().model_dump(), "data": _wide_records(), "fmt": "pdf"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_render_figure_png_zip():
    import io
    import zipfile

    r = client.post("/render-figure", json={
        "figure": _fig_spec().model_dump(), "data": _wide_records(), "fmt": "png"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == ["page_0.png", "page_1.png"]  # 5 panels, 4/page -> 2 pages
