# grism design: spec-driven plotting

Status: proposal. Captures the plan to turn grism into (a) a UI for customising
plots and (b) a reusable API for generating plots we make often in the same way.

Conda env: grism

## Ideas

- Could we make this look like seabborn or something that exists already?
  i.e. so that it's easy to look up what we should be doing, so that
  we don't have to figure out what to do, so that we are basically just
  doing some wrapping and plumbing?
  → DONE. grism *is* a thin wrapper over seaborn, so the spec + Python API
  now use seaborn's vocabulary: `x`, `y`, `hue`, `order`, `palette`,
  `estimator`. Users look these up in the seaborn docs. grism's genuine
  value-add stays as clearly-grism extras: `elements` (stack strip+bar+
  whisker in one call — seaborn's `catplot` won't), `whisker_mode`,
  `bar_fill`, significance staples, and print-scale knobs.

## Goal

One declarative **plot spec** as the single source of truth. The UI authors a
spec; the API renders one. Templating and versioning ride on top of the spec.

```
            +-------------+
   UI  ----->             |
            |  PlotSpec   |----> render() ----> PNG / SVG / PDF (matplotlib)
   API ----->  (JSON)     |
            +-------------+
```

Decisions taken:
- **Output is print-quality PDF/PNG/SVG only.** matplotlib stays the one
  renderer. No Plotly/Vega/web-rendering backend.
- **API consumers: both** Python/notebook (now) and remote HTTP (later). The
  spec + `render()` must not depend on Streamlit, FastAPI, or the filesystem.
- **UI: Streamlit now, open to a web app later.** A future JS frontend just
  produces the same spec JSON and POSTs it to `/render`.
- **Templating and versioning are in scope from day one.**

## Where we are today

grism already has the right split:
- `grism/grism.py` — pure core: `plot()`, `stats()`, `plot_with_stats()` take a
  DataFrame + kwargs, return a matplotlib `Axes`. This is the de-facto API.
- `grism/ui_streamlit.py` — interactive UI; builds a `plot_cfg` **dict that
  already fully describes a plot** (see `_default_plot_config` and the persisted
  dict) and saves it to `~/.grism`.

That `plot_cfg` dict *is* a plot spec — it just needs to be promoted to a
first-class, validated, versioned artifact, and the shared logic currently
living in the UI needs to move down into core.

## Architecture

```
grism/
  spec.py         # PlotSpec + PlotTemplate (pydantic) + JSON schema + migrate()
  render.py       # render(spec, df) -> RenderResult   <- pure, no I/O
  core.py         # existing plotting/figsize internals (renamed grism.py)
  stats.py        # stats + normality-driven test pick (moved out of the UI)
  cli.py          # `grism render` / `grism apply`     <- adapter
  api.py          # FastAPI POST /render               <- adapter (later)
  ui_streamlit.py # spec editor + live preview          <- adapter
```

The rule that keeps "later" cheap: **everything reusable lives below
`render()`.** Data prep, the `"auto"` test pick, and figure sizing must be in
core, not the UI — otherwise the CLI/API produce different plots from the same
spec, which defeats templating.

## The spec

Split so the dataset-dependent part (`DataBinding`) is separable from the
dataset-independent part (appearance / stats / labels). A **template** is a spec
minus its binding.

```python
SPEC_VERSION = 1

class DataBinding(BaseModel):        # the ONLY dataset-dependent part
    x: str                           # seaborn: categorical/group column
    y: str                           # seaborn: numeric value column
    hue: str | None = None
    order: list[str] = []            # x-category display order (seaborn)
    wide_form: bool = False
    wide_id_cols: list[str] = []
    row_filter: str | None = None    # predicate e.g. "dose > 0" (see note)

class StatsSpec(BaseModel):
    test: StatTest = StatTest.auto   # auto = normality-driven pick
    pairs: list[tuple[str, str]] = []
    staple_scale: float = 1.0

class LabelSpec(BaseModel):          # None == "use auto default"
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None

class AppearanceSpec(BaseModel):
    elements: list[Element] = [strip, bar, whisker]   # grism overlay extra
    whisker_mode: WhiskerMode = quartiles             # grism extra
    estimator: Estimator = median                     # seaborn bar estimator
    bar_fill: BarFill = block                         # grism extra
    use_group_colors: bool = True
    palette: str = "tab10"                            # seaborn palette
    style: str = "default"
    rotate_xticks: bool = False
    y_zero: bool = True
    plot_scale: float = 1.0
    x_scale: float = 1.0
    y_scale: float = 1.0

class PlotSpec(BaseModel):
    spec_version: int = SPEC_VERSION
    name: str = "Plot 1"
    data: DataBinding
    appearance: AppearanceSpec = AppearanceSpec()
    stats: StatsSpec = StatsSpec()
    labels: LabelSpec = LabelSpec()
```

Enums (`Element`, `WhiskerMode`, `Estimator`, `BarFill`, `StatTest`) replace
today's magic strings.

### Templating

```python
class PlotTemplate(BaseModel):       # spec minus DataBinding
    spec_version: int = SPEC_VERSION
    name: str
    appearance: AppearanceSpec
    stats: StatsSpec
    labels: LabelSpec

    def bind(self, data: DataBinding) -> PlotSpec: ...
```

"Plots we make the same way" = save one house-style template, then
`template.bind(DataBinding(group="treatment", value="tumor_volume")).render(df)`
against each new dataset. The UI's "Apply settings to all plots" button becomes
"apply template" — the same operation the CLI/API use.

### Versioning

`spec_version` at the top level; `from_json` runs raw dicts through a migration
chain before validating:

```python
def migrate(raw: dict) -> dict:
    v = raw.get("spec_version", 1)
    if v < 2: raw = _v1_to_v2(raw)
    # chain forward...
    raw["spec_version"] = SPEC_VERSION
    return raw
```

Each schema change adds one `_vN_to_vN+1` step; old saved specs/templates keep
opening. Pydantic then guarantees anything reaching `render()` is well-formed.

## The render seam

```python
@dataclass
class RenderResult:
    figure: Figure
    resolved_test: str          # what "auto" actually chose
    omnibus: object | None
    pairwise: list
    normality: dict

def prepare_data(spec, df) -> (df_plot, order):
    # melt (wide_form), apply row_filter, resolve group order vs. present groups
    ...

def render(spec: PlotSpec, df, *, ax=None) -> RenderResult:
    df_plot, order = prepare_data(spec, df)
    normality = core.group_normality(...)
    test = pick_test_by_normality(normality) if spec.stats.test is auto else spec.stats.test
    ax, omnibus, pairwise = core.plot_with_stats(df_plot, ...unpack spec...)
    return RenderResult(ax.figure, test, omnibus, pairwise, normality)

def render_bytes(spec, df, fmt="pdf") -> bytes:
    # savefig(bbox_inches="tight") into a buffer
    ...
```

`plot_with_stats` keeps its keyword signature and stays usable directly;
`render()` is just the adapter that unpacks a `PlotSpec` into those kwargs. The
batch-zip export loop in the UI collapses to `for spec in specs: render_bytes(spec, df)`.

Adapters are thin: `cli.py` reads a spec/template + CSV and writes bytes;
`api.py` is a FastAPI `POST /render {spec, data, fmt}` reusing `render_bytes`
verbatim; `ui_streamlit.py` builds a spec from widgets, previews with `render()`,
and saves templates via `PlotTemplate(**spec.dict(exclude={"data"}))`.

## Two deliberate changes from today's dict

1. **`row_include` (positional bool mask) -> `row_filter` (predicate).** The
   current per-row `[True, False, ...]` mask is tied to one file's row count and
   can't survive new data (there's already guard code fighting this). A
   predicate like `"dose > 0"` is templatable. Keep ad-hoc interactive row
   exclusion as Streamlit *session state*, not part of the saved spec.
2. **`*_enabled` + `*_text` pairs -> one `Optional[str]`.** `None` = auto-generate
   (`f"{value} by {group}"`), a string = use it. Removes the illegal
   "enabled but empty" state.

Everything else maps one-to-one from the current config.

## Migration checklist

- [x] Add `spec.py`: enums, sub-specs, `PlotSpec`, `PlotTemplate`, `SPEC_VERSION`,
      `migrate()`, `to_json`/`from_json`. Emit a JSON Schema for the frontend.
      (`json_schema()`; `grism schema` dumps it.)
- [x] Rename `grism.py` -> `core.py`; move `group_normality` and
      `pick_test_by_normality` out of the UI into `stats.py`. (Also moved
      `stats`/`pairwise_stats` + shared `validate_columns`/`group_order`.)
- [x] Add `render.py`: `prepare_data`, `render`, `render_bytes`, `RenderResult`.
      Move wide-form melt, order resolution, and figsize math into core
      (`compute_figsize_for`).
- [x] Point `ui_streamlit.py` at `PlotSpec`/`render()`; add "save as template".
      Live preview + batch-zip both go through `render()`/`render_bytes()` via
      `_canonical_spec()`; "Save spec / template (JSON)" download buttons added.
      Interactive row-include stays session-only (not in the exported spec).
- [x] Add `cli.py` (`grism render`, `grism apply`, `grism figure`,
      `grism render-figure`, `grism templates`, `grism schema`, `grism serve`)
      + `grism` entry point in pyproject.
- [x] Template library (minimal): `grism/templates/*.json` (`default`, `bars`)
      + `library.py` (`list_templates`/`load_template`), shipped via
      package-data. `grism apply`/`figure` accept a built-in name or a file.
- [x] Add `api.py` (FastAPI): `POST /render`, `POST /apply`, `GET /schema`,
      `GET /health`; `grism serve` runs it. Forces the Agg backend (requests
      run on worker threads). Tested in-process (TestClient) and over HTTP.
- [x] Add `figure.py` (multi-panel): `FigureSpec` (list of `PlotSpec` panels +
      grid/pagination), `figure_from_columns` (one template across many value
      columns), `render_pages` -> `list[Figure]`, `render_pdf_bytes` (one
      multi-page PDF), `render_page_images`. Sits ABOVE `render()` and reuses
      its `ax=` hook — spec/core/render unchanged. seaborn primitive vs
      FacetGrid split.
- [x] Tests: round-trip `to_json`/`from_json`, migration, render from spec,
      template reuse, row-filter, wide-form, API (TestClient), multi-panel
      pagination + multi-page PDF (`tests/`, 25 passing).
      Golden-image cross-adapter test: not yet.

## Multi-panel figures

`PlotSpec` stays the single-panel atom (one Axes, one set of stats — the 1:1
seaborn mapping). Multiple panels are a layer above, not a change to the spec:

- `FigureSpec` = `list[PlotSpec]` + layout (`ncols`, `panels_per_page`,
  `panel_size`). A future `FigureTemplate` could pair a `PlotTemplate` with
  layout, but `figure_from_columns(template, x, y_columns, ...)` covers the
  common wide-sheet case now.
- `render_pages` makes a grid of axes per page and calls `render(panel, df,
  ax=ax)` for each — the single-panel seam does all the drawing/stats; the
  figure layer only does layout + pagination + hiding empty cells.
- Not folded into `y=[list]` deliberately: stats/labels/limits are per-panel,
  and a flat `y` list would force every `RenderResult` field to become plural
  and break the seaborn-shaped single call.
- Adapters: CLI `grism figure` (template + columns) and `grism render-figure`
  (saved FigureSpec), and API `POST /render-figure` (PDF, or a zip of per-page
  images) are wired and tested over HTTP. `.pdf` = one multi-page file; png/svg
  = one file/entry per page. UI page-grouping is still not wired.
