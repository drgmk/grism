"""grism command line: render a saved spec, or apply a template to new data.

    grism render spec.json data.csv -o figure.pdf
    grism apply template.json data.csv --group treatment --value tumour_volume
    grism schema > plotspec.schema.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from .figure import FigureSpec, figure_from_columns, render_page_images, render_pdf_bytes
from .library import list_templates, load_template
from .render import render_bytes
from .spec import DataBinding, PlotSpec, PlotTemplate, json_schema

app = typer.Typer(add_completion=False, help="Spec-driven plotting for grism.")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise typer.BadParameter(f"Unsupported data format: {suffix} (use .csv/.xlsx/.xls)")


def _write(out: Path, data: bytes) -> None:
    out.write_bytes(data)
    typer.echo(f"wrote {out} ({len(data)} bytes)")


def _resolve_template(ref: Path) -> PlotTemplate:
    """Load a template from a JSON file, or by built-in name (e.g. `default`)."""
    if ref.exists():
        return PlotTemplate.from_json(ref.read_text())
    if ref.stem in list_templates():
        return load_template(ref.stem)
    raise typer.BadParameter(
        f"Template {ref} not found (file or built-in). Built-ins: {', '.join(list_templates())}"
    )


@app.command()
def render(
    spec: Path = typer.Argument(..., help="PlotSpec JSON file."),
    data: Path = typer.Argument(..., help="CSV or Excel data file."),
    out: Path = typer.Option("figure.pdf", "-o", "--out", help="Output file (fmt from extension)."),
) -> None:
    """Render a saved PlotSpec against a data file."""
    plot_spec = PlotSpec.from_json(spec.read_text())
    df = _read_table(data)
    fmt = out.suffix.lstrip(".") or "pdf"
    _write(out, render_bytes(plot_spec, df, fmt=fmt))


@app.command()
def apply(
    template: Path = typer.Argument(..., help="PlotTemplate JSON file, or a built-in name (see `grism templates`)."),
    data: Path = typer.Argument(..., help="CSV or Excel data file."),
    x: str = typer.Option(..., "--x", "-x", help="Categorical/group column (seaborn x)."),
    y: str = typer.Option(..., "--y", "-y", help="Numeric value column (seaborn y)."),
    hue: Optional[str] = typer.Option(None, "--hue", help="Optional hue column."),
    order: Optional[List[str]] = typer.Option(None, "--order", help="x-category display order."),
    row_filter: Optional[str] = typer.Option(None, "--filter", help="pandas query() predicate."),
    out: Path = typer.Option("figure.pdf", "-o", "--out", help="Output file (fmt from extension)."),
) -> None:
    """Bind a template to a fresh dataset and render — the 'same plot, new data' path."""
    tmpl = _resolve_template(template)
    binding = DataBinding(
        x=x,
        y=y,
        hue=hue,
        order=list(order) if order else [],
        row_filter=row_filter,
    )
    df = _read_table(data)
    fmt = out.suffix.lstrip(".") or "pdf"
    _write(out, render_bytes(tmpl.bind(binding), df, fmt=fmt))


def _write_figure(fig_spec: FigureSpec, df: pd.DataFrame, out: Path) -> None:
    """Write a multi-panel figure: one PDF for .pdf, else one image per page."""
    fmt = out.suffix.lstrip(".") or "pdf"
    if fmt == "pdf":
        _write(out, render_pdf_bytes(fig_spec, df))
        return
    pages = render_page_images(fig_spec, df, fmt=fmt)
    stem = out.with_suffix("")
    for i, data in enumerate(pages):
        _write(Path(f"{stem}_p{i}.{fmt}"), data)


@app.command()
def figure(
    template: Path = typer.Argument(..., help="PlotTemplate JSON file, or a built-in name (see `grism templates`)."),
    data: Path = typer.Argument(..., help="CSV or Excel data file."),
    x: str = typer.Option(..., "--x", "-x", help="Categorical/sample column (x)."),
    y: Optional[List[str]] = typer.Option(
        None, "--y", "-y", help="Value column(s); repeat. Default: all numeric columns except x."
    ),
    ncols: int = typer.Option(2, "--ncols", help="Panels per row."),
    per_page: Optional[int] = typer.Option(
        None, "--per-page", help="Panels per page (default: all on one page)."
    ),
    out: Path = typer.Option("figure.pdf", "-o", "--out", help="Output (.pdf = multi-page; png/svg = one file per page)."),
) -> None:
    """One template across many value columns -> a paginated multi-panel figure."""
    tmpl = _resolve_template(template)
    df = _read_table(data)
    y_columns = list(y) if y else [
        c for c in df.columns if c != x and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not y_columns:
        raise typer.BadParameter("No value columns to plot (give --y, or check numeric columns).")
    fig_spec = figure_from_columns(
        tmpl, x=x, y_columns=y_columns, ncols=ncols, panels_per_page=per_page
    )
    _write_figure(fig_spec, df, out)


@app.command(name="render-figure")
def render_figure(
    figspec: Path = typer.Argument(..., help="FigureSpec JSON file."),
    data: Path = typer.Argument(..., help="CSV or Excel data file."),
    out: Path = typer.Option("figure.pdf", "-o", "--out", help="Output (.pdf = multi-page; png/svg = one file per page)."),
) -> None:
    """Render a saved FigureSpec (multi-panel) against a data file."""
    fig_spec = FigureSpec.from_json(figspec.read_text())
    df = _read_table(data)
    _write_figure(fig_spec, df, out)


@app.command()
def templates() -> None:
    """List the built-in plot templates (use with `grism apply`)."""
    for name in list_templates():
        typer.echo(name)


@app.command()
def schema() -> None:
    """Print the PlotSpec JSON Schema (for a frontend or validation)."""
    typer.echo(json.dumps(json_schema(), indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Run the HTTP API (needs the [api] extra: pip install grism[api])."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter("uvicorn not installed. Run: pip install grism[api]") from exc
    typer.echo(f"grism API on http://{host}:{port}  (docs at /docs)")
    uvicorn.run("grism.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
