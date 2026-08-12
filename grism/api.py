"""HTTP API for grism: render a spec (or apply a template) to an image.

A thin adapter over :func:`grism.render.render_bytes` — the same seam the UI
and CLI use, so a spec renders identically here. Run it with::

    grism serve                      # or: uvicorn grism.api:app --reload

Then POST data + a spec to ``/render``. Interactive docs live at ``/docs``.
"""

from __future__ import annotations

# Requests are served on worker threads; force the headless Agg backend before
# pyplot is imported (via .render) so rendering never touches a GUI backend.
import matplotlib

matplotlib.use("Agg")

import io
import zipfile
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .figure import FigureSpec, render_page_images, render_pdf_bytes
from .render import render_bytes
from .spec import DataBinding, PlotSpec, PlotTemplate, json_schema

app = FastAPI(
    title="grism",
    description="Spec-driven plotting: POST a spec + data, get a print-quality figure.",
    version="0.0.1",
)

Fmt = Literal["pdf", "png", "svg"]

_MEDIA = {
    "pdf": "application/pdf",
    "png": "image/png",
    "svg": "image/svg+xml",
}


class RenderRequest(BaseModel):
    spec: PlotSpec
    data: List[dict] = Field(..., description="Rows as records, e.g. df.to_dict('records').")
    fmt: Fmt = "pdf"


class ApplyRequest(BaseModel):
    template: PlotTemplate
    binding: DataBinding
    data: List[dict] = Field(..., description="Rows as records.")
    fmt: Fmt = "pdf"


class FigureRequest(BaseModel):
    figure: FigureSpec
    data: List[dict] = Field(..., description="Rows as records.")
    fmt: Fmt = "pdf"


def _render_response(spec: PlotSpec, data: List[dict], fmt: str) -> Response:
    if not data:
        raise HTTPException(status_code=422, detail="`data` is empty.")
    df = pd.DataFrame(data)
    try:
        payload = render_bytes(spec, df, fmt=fmt)
    except Exception as exc:  # bad column names, empty groups, etc.
        raise HTTPException(status_code=422, detail=f"Render failed: {exc}") from exc
    return Response(content=payload, media_type=_MEDIA[fmt])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> JSONResponse:
    """JSON Schema for a PlotSpec (handy for a frontend or client validation)."""
    return JSONResponse(json_schema())


@app.post(
    "/render",
    responses={200: {"content": {m: {} for m in _MEDIA.values()}}},
)
def render_endpoint(req: RenderRequest) -> Response:
    """Render a full spec against posted data."""
    return _render_response(req.spec, req.data, req.fmt)


@app.post(
    "/apply",
    responses={200: {"content": {m: {} for m in _MEDIA.values()}}},
)
def apply_endpoint(req: ApplyRequest) -> Response:
    """Bind a template to a data binding, then render — 'same plot, new data'."""
    return _render_response(req.template.bind(req.binding), req.data, req.fmt)


@app.post(
    "/render-figure",
    responses={200: {"content": {"application/pdf": {}, "application/zip": {}}}},
)
def render_figure_endpoint(req: FigureRequest) -> Response:
    """Render a multi-panel figure: PDF (multi-page) for fmt=pdf, else a zip of
    one image per page."""
    if not req.data:
        raise HTTPException(status_code=422, detail="`data` is empty.")
    df = pd.DataFrame(req.data)
    try:
        if req.fmt == "pdf":
            return Response(render_pdf_bytes(req.figure, df), media_type="application/pdf")
        pages = render_page_images(req.figure, df, fmt=req.fmt)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Render failed: {exc}") from exc

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, data in enumerate(pages):
            zf.writestr(f"page_{i}.{req.fmt}", data)
    return Response(buf.getvalue(), media_type="application/zip")
