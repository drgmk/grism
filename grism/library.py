"""A small built-in library of reusable plot templates.

Templates ship as JSON under ``grism/templates/``. This is deliberately
minimal — a couple of starters plus a loader; grow it by dropping more
``<name>.json`` files (a saved `PlotTemplate`) into that directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .spec import PlotTemplate

_DIR = Path(__file__).resolve().parent / "templates"


def list_templates() -> List[str]:
    """Names of the built-in templates (sorted)."""
    return sorted(p.stem for p in _DIR.glob("*.json"))


def load_template(name: str) -> PlotTemplate:
    """Load a built-in template by name (e.g. "default")."""
    path = _DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_templates()) or "(none)"
        raise FileNotFoundError(f"No template {name!r}. Available: {available}")
    return PlotTemplate.from_json(path.read_text())
