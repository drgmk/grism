"""Streamlit UI for grism."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy import stats as sps

import grism as core


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _style_options() -> Dict[str, str]:
    options: Dict[str, str] = {}
    data_dir = _data_dir()
    default_path = data_dir / "default.mplstyle"
    if default_path.exists():
        options["grism default"] = str(default_path)

    if data_dir.exists():
        for style_path in sorted(data_dir.glob("*.mplstyle")):
            if style_path == default_path:
                continue
            options[f"data/{style_path.stem}"] = str(style_path)

    for name in sorted(plt.style.available):
        options[f"mpl/{name}"] = name

    if not options:
        options["matplotlib default"] = "default"
    return options



def _jsonify(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return str(obj)



def _guess_wide_form(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 2:
        return False
    nrows = len(df)
    # Count columns with repeated values (potential group/id cols).
    repeated_cols = [c for c in df.columns if df[c].nunique(dropna=True) < nrows]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Heuristic: if no repeated columns and multiple numeric columns, likely wide.
    if not repeated_cols and len(numeric_cols) >= 2:
        return True
    # If there is exactly one repeated column and many numeric columns, maybe wide with an ID column.
    if len(repeated_cols) == 1 and len(numeric_cols) >= 2:
        return True
    return False



def _config_path() -> Path:
    return Path.home() / ".grism"


def _load_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    path = _config_path()
    path.write_text(json.dumps(_jsonify(cfg), indent=2, sort_keys=True))


def _ensure_file_config(cfg: dict, filename: str) -> dict:
    file_cfg = cfg.get(filename)
    if not file_cfg:
        file_cfg = {"plots": {}, "order": [], "current": None}
        cfg[filename] = file_cfg
    return file_cfg


def _new_plot_name(existing: list[str]) -> str:
    idx = 1
    while f"Plot {idx}" in existing:
        idx += 1
    return f"Plot {idx}"


def _default_plot_config() -> dict:
    return {
        "group": None,
        "value": None,
        "hue": "(none)",
        "elements": ["strip", "bar", "whisker"],
        "whisker_mode": "quartiles",
        "bar_mode": "median",
        "bar_fill": "block",
        "use_group_colors": True,
        "color_cycle": "tab10",
        "plot_scale": 1.0,
        "x_scale": 1.0,
        "y_scale": 1.0,
        "order": [],
        "pairs": [],
        "title_enabled": False,
        "title_text": "",
        "xlabel_enabled": False,
        "xlabel_text": "",
        "ylabel_enabled": False,
        "ylabel_text": "",
        "staple_scale": 1.0,
        "style_label": None,
    }


def _widget_key(filename: str, plot_name: str, field: str) -> str:
    return f"{filename}:{plot_name}:{field}"



def _default_group_column(df: pd.DataFrame, columns: list[str]) -> str:
    uniques = {c: df[c].nunique(dropna=True) for c in columns}
    return min(columns, key=lambda c: (uniques[c], columns.index(c)))


def _is_uniform_step_numeric(series: pd.Series) -> bool:
    vals = pd.Series(series).dropna().to_numpy()
    if vals.size < 3:
        return False
    try:
        diffs = vals.astype(float)
    except (TypeError, ValueError):
        return False
    diffs = sorted(diffs)
    if len(diffs) < 3:
        return False
    steps = [b - a for a, b in zip(diffs[:-1], diffs[1:])]
    steps = [s for s in steps if pd.notna(s)]
    if not steps:
        return False
    first = steps[0]
    return all(abs(s - first) <= 1e-9 for s in steps)


def _default_value_column(df: pd.DataFrame, columns: list[str], group: str) -> str:
    def _ranked(cands: list[str]) -> list[str]:
        non_index = [c for c in cands if not _is_uniform_step_numeric(df[c])]
        return non_index or cands

    float_cols = [c for c in columns if c != group and pd.api.types.is_float_dtype(df[c])]
    ranked_float = _ranked(float_cols)
    if ranked_float:
        return ranked_float[0]

    numeric_cols = [c for c in columns if c != group and pd.api.types.is_numeric_dtype(df[c])]
    ranked_numeric = _ranked(numeric_cols)
    if ranked_numeric:
        return ranked_numeric[0]

    for c in columns:
        if c != group:
            return c
    return columns[0]


def _auto_plot_columns(df: pd.DataFrame) -> Tuple[Optional[str], list[str]]:
    columns = list(df.columns)
    if not columns:
        return None, []
    group_col = _default_group_column(df, columns)
    value_cols = [c for c in columns if c != group_col] or [group_col]
    return group_col, value_cols


def _sync_file_plots_with_columns(file_cfg: dict, df: pd.DataFrame) -> bool:
    group_col, value_cols = _auto_plot_columns(df)
    if not value_cols:
        return False

    changed = False
    for value_col in value_cols:
        if value_col in file_cfg["plots"]:
            continue
        cfg = _default_plot_config()
        cfg["group"] = group_col
        cfg["value"] = value_col
        file_cfg["plots"][value_col] = cfg
        file_cfg["order"].append(value_col)
        changed = True

    if not file_cfg["order"]:
        file_cfg["order"] = [value_cols[0]]
        changed = True

    current = file_cfg.get("current")
    if current not in file_cfg["plots"]:
        file_cfg["current"] = file_cfg["order"][0]
        changed = True

    return changed






def _set_if_missing(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _init_widget(key: str, default) -> None:
    if key not in st.session_state:
        st.session_state[key] = default



def _slider_state(label: str, key: str, min_value: float, max_value: float, step: float, default: float) -> float:
    if key in st.session_state:
        return st.slider(label, min_value=min_value, max_value=max_value, step=step, key=key)
    # First render: seed with default.
    return st.slider(label, min_value=min_value, max_value=max_value, value=default, step=step, key=key)



def _ordered_selection(label: str, options: list[str], default: list[str], key: str) -> list[str]:
    if key in st.session_state:
        selected = st.multiselect(label, options, key=key)
    else:
        selected = st.multiselect(label, options, default=default, key=key)
    order_key = f"{key}__order"
    prior = st.session_state.get(order_key, [])
    # Remove items no longer selected.
    ordered = [item for item in prior if item in selected]
    # Append newly selected items in the order they appear in `selected`.
    for item in selected:
        if item not in ordered:
            ordered.append(item)
    st.session_state[order_key] = ordered
    return ordered



def _cycle_select(label: str, options: list[str], key: str, default: str) -> str:
    idx_key = f"{key}__idx"
    select_key = f"{key}__select"
    flag_key = f"{key}__from_button"

    default_value = default if default in options else options[0]

    if idx_key not in st.session_state:
        st.session_state[idx_key] = options.index(default_value)
    if select_key not in st.session_state:
        st.session_state[select_key] = default_value
    if flag_key not in st.session_state:
        st.session_state[flag_key] = False

    def _prev() -> None:
        st.session_state[idx_key] = (st.session_state[idx_key] - 1) % len(options)
        st.session_state[flag_key] = True

    def _next() -> None:
        st.session_state[idx_key] = (st.session_state[idx_key] + 1) % len(options)
        st.session_state[flag_key] = True

    # Only override the selectbox value when a button was used.
    if st.session_state[flag_key]:
        st.session_state[select_key] = options[st.session_state[idx_key]]
        st.session_state[flag_key] = False

    cols = st.columns([0.12, 1.0, 0.12], gap="small")
    cols[0].caption(" ")
    cols[0].button("◀", key=f"{key}__prev", on_click=_prev)
    selection = cols[1].selectbox(
        label,
        options,
        key=select_key,
    )
    cols[2].caption(" ")
    cols[2].button("▶", key=f"{key}__next", on_click=_next)

    # Sync index to selected value in case of manual select.
    st.session_state[idx_key] = options.index(selection)
    return selection



def _group_normality(df: pd.DataFrame, group: str, value: str, order: list[str]) -> Dict[str, Optional[float]]:
    pvals: Dict[str, Optional[float]] = {}
    for g in order:
        vals = df.loc[df[group] == g, value].dropna().to_numpy()
        if vals.size < 3:
            # Too few points to test: assume normal (p=1).
            pvals[g] = 1.0
            continue
        try:
            _stat, p = sps.shapiro(vals)
        except Exception:
            pvals[g] = None
            continue
        pvals[g] = float(p)
    return pvals


def _normality_label(p: Optional[float]) -> str:
    if p is None:
        return "error"
    return "yes" if p >= 0.05 else "no"


def _default_pairwise_test(normality: Dict[str, Optional[float]]) -> str:
    labels = [_normality_label(p) for p in normality.values()]
    if labels and all(lbl == "yes" for lbl in labels):
        return "t_test"
    return "mann_whitney"


@st.cache_data
def _load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Unsupported file format. Use CSV or Excel.")


def main() -> None:
    st.set_page_config(page_title="grism", layout="wide")
    st.markdown("""
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 1rem; }
            section[data-testid="stSidebarHeader"] > div { height: 2rem; }
            .st-emotion-cache-10p9htt { margin-bottom: 1rem !important; height: 2rem !important; }
        </style>
    """, unsafe_allow_html=True)
    st.title("grism")
    # st.caption("Simple biology-style plots with stats and annotations")

    st.sidebar.header("Basic setup")
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"],
                                        label_visibility="collapsed")

    if not uploaded:
        st.info("Upload a file to begin.")
        st.stop()

    filename = uploaded.name
    df = _load_dataframe(uploaded.getvalue(), uploaded.name)
    raw_df = df.copy()
    cfg = _load_config()
    file_cfg = _ensure_file_config(cfg, filename)

    if _sync_file_plots_with_columns(file_cfg, df):
        _save_config(cfg)

    plot_names = file_cfg["order"]
    current_plot = file_cfg.get("current") or plot_names[0]

    plot_row = st.sidebar.columns([0.7, 0.3], gap="small")
    plot_key = f"{filename}__plot_select"
    if plot_key not in st.session_state:
        st.session_state[plot_key] = current_plot
    pending_key = f"{plot_key}__pending"
    if pending_key in st.session_state:
        st.session_state[plot_key] = st.session_state.pop(pending_key)
    selected_plot = plot_row[0].selectbox("Plot selection and rename", plot_names, key=plot_key)

    # Rename current plot
    rename_key = f"{filename}:{selected_plot}:rename"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = selected_plot
    new_name = plot_row[0].text_input("Rename", key=rename_key, label_visibility="collapsed")
    if new_name and new_name != selected_plot and new_name not in plot_names:
        file_cfg["plots"][new_name] = file_cfg["plots"].pop(selected_plot)
        file_cfg["order"] = [new_name if p == selected_plot else p for p in file_cfg["order"]]
        file_cfg["current"] = new_name
        st.session_state[f"{plot_key}__pending"] = new_name
        _save_config(cfg)
        st.rerun()

    # New/Delete buttons aligned with select/rename block
    plot_row[1].caption(" ")
    plot_row[1].caption(" ")
    if plot_row[1].button("New", use_container_width=True):
        fresh = _new_plot_name(plot_names)
        file_cfg["order"].append(fresh)
        file_cfg["plots"][fresh] = _default_plot_config()
        file_cfg["current"] = fresh
        st.session_state[f"{plot_key}__pending"] = fresh
        _save_config(cfg)
        st.rerun()

    if plot_row[1].button("Delete", use_container_width=True):
        if selected_plot in file_cfg["plots"]:
            file_cfg["plots"].pop(selected_plot)
        file_cfg["order"] = [p for p in file_cfg["order"] if p != selected_plot]
        if not file_cfg["order"]:
            group_col, value_cols = _auto_plot_columns(df)
            fresh = value_cols[0] if value_cols else _new_plot_name([])
            fresh_cfg = _default_plot_config()
            fresh_cfg["group"] = group_col
            fresh_cfg["value"] = fresh
            file_cfg["order"] = [fresh]
            file_cfg["plots"][fresh] = fresh_cfg
        file_cfg["current"] = file_cfg["order"][0]
        st.session_state[f"{plot_key}__pending"] = file_cfg["current"]
        _save_config(cfg)
        st.rerun()

    apply_all_key = f"{filename}:{selected_plot}:apply_all"
    apply_to_all = st.sidebar.button("Apply settings to all plots", key=apply_all_key)

    file_cfg["current"] = st.session_state[plot_key]
    selected_plot = st.session_state[plot_key]

    plot_cfg = file_cfg["plots"].setdefault(selected_plot, _default_plot_config())
    _save_config(cfg)

    wide_key = _widget_key(filename, selected_plot, "wide_form")
    wide_default = plot_cfg.get("wide_form", _guess_wide_form(df))
    _init_widget(wide_key, wide_default)
    wide_form = st.sidebar.checkbox("Table is wide form", key=wide_key)

    if wide_form:
        id_cols = list(df.columns)
        id_key = _widget_key(filename, selected_plot, "wide_id_cols")
        if id_key not in st.session_state:
            # First time: assume first column is an ID to keep.
            st.session_state[id_key] = [id_cols[0]] if id_cols else []
        id_keep = st.sidebar.multiselect(
            "Columns not to melt",
            id_cols,
            key=id_key,
            help="These columns stay as identifiers; all others become groups.",
        )
        value_name = "value"
        var_name = "group"
        df = df.melt(id_vars=id_keep, var_name=var_name, value_name=value_name)

    columns = list(df.columns)

    # Row inclusion filter (post-melt if wide form is enabled).
    include_key = _widget_key(filename, selected_plot, "row_include")
    if include_key not in st.session_state:
        saved_include = plot_cfg.get("row_include")
        if isinstance(saved_include, list) and len(saved_include) == len(df):
            st.session_state[include_key] = saved_include
        else:
            st.session_state[include_key] = [True] * len(df)
    include_series = st.session_state[include_key]
    if len(include_series) != len(df):
        include_series = [True] * len(df)
        st.session_state[include_key] = include_series
    df = df.assign(Include=include_series)

    if wide_form:
        plot_cfg["group"] = "group"
        plot_cfg["value"] = "value"

    plot_id = f"{filename}:{selected_plot}:{wide_form}"
    plot_changed = st.session_state.get("__plot_id") != plot_id
    st.session_state["__plot_id"] = plot_id
    if not columns:
        st.warning("No columns found in the uploaded file.")
        st.stop()

    if plot_changed:
        st.session_state.pop(_widget_key(filename, selected_plot, "group"), None)
        st.session_state.pop(_widget_key(filename, selected_plot, "value"), None)
        st.session_state.pop(_widget_key(filename, selected_plot, "row_include"), None)

    if wide_form and "group" in columns:
        default_group = "group"
    else:
        default_group = plot_cfg.get("group") if plot_cfg.get("group") in columns else _default_group_column(df, columns)
    group_key = _widget_key(filename, selected_plot, "group")
    _set_if_missing(group_key, default_group)
    group = st.sidebar.selectbox(
        "Group column",
        columns,
        key=group_key,
    )

    if wide_form and "value" in columns and "value" != group:
        default_value = "value"
    else:
        default_value = plot_cfg.get("value") if plot_cfg.get("value") in columns and plot_cfg.get("value") != group else _default_value_column(df, columns, group)
    value_key = _widget_key(filename, selected_plot, "value")
    _set_if_missing(value_key, default_value)
    value = st.sidebar.selectbox(
        "Value column",
        columns,
        key=value_key,
    )

    if group == value:
        st.sidebar.warning("Group and value columns should differ.")

    hue_options = ["(none)"] + [c for c in columns if c not in {value, group}]
    hue_default = plot_cfg.get("hue") if plot_cfg.get("hue") in hue_options else "(none)"
    hue_key = _widget_key(filename, selected_plot, "hue")
    _set_if_missing(hue_key, hue_default)
    hue_choice = st.sidebar.selectbox("Hue (optional)", hue_options, key=hue_key)
    hue: Optional[str] = None if hue_choice == "(none)" else hue_choice

    top_plot_col, top_style_col, top_empty_col = st.columns([2, 1, 1], gap="large")

    style_map = _style_options()
    with top_style_col:
        st.subheader("Plot setup")

        element_options = ["strip", "bar", "whisker", "hist"]
        elements_default = [e for e in plot_cfg.get("elements", []) if e in element_options] or ["strip", "bar", "whisker"]
        elements_key = _widget_key(filename, selected_plot, "elements")
        _set_if_missing(elements_key, elements_default)
        elements = st.multiselect(
            "Elements",
            element_options,
            key=elements_key,
        )

        mode_cols = st.columns(2, gap="small")
        whisker_mode_key = _widget_key(filename, selected_plot, "whisker_mode")
        _set_if_missing(whisker_mode_key, plot_cfg.get("whisker_mode", "quartiles"))
        whisker_mode = mode_cols[0].radio(
            "Whisker style",
            ["quartiles", "mean-std"],
            horizontal=True,
            key=whisker_mode_key,
        )
        bar_mode_key = _widget_key(filename, selected_plot, "bar_mode")
        _set_if_missing(bar_mode_key, plot_cfg.get("bar_mode", "median"))
        bar_mode = mode_cols[1].radio(
            "Bar value",
            ["median", "mean"],
            horizontal=True,
            key=bar_mode_key,
        )

        group_values = list(pd.Series(df[group]).dropna().unique())
        group_default = [g for g in plot_cfg.get("order", []) if g in group_values] or group_values
        group_order_key = _widget_key(filename, selected_plot, "group_order")
        if plot_changed:
            st.session_state[group_order_key] = group_default
            st.session_state[f"{group_order_key}__order"] = list(group_default)
        order = _ordered_selection(
            "Groups to plot",
            group_values,
            default=group_default,
            key=group_order_key,
        )

        # st.caption("Only selected groups will be plotted. Order follows your selection.")

        pair_options = [
            (a, b)
            for i, a in enumerate(order)
            for b in order[i + 1 :]
        ]
        pair_labels = [f"{a}-{b}" for a, b in pair_options]
        stored_pairs = plot_cfg.get("pairs", [])
        stored_labels = [
            f"{a}-{b}" for a, b in stored_pairs if f"{a}-{b}" in pair_labels
        ]
        pair_default = stored_labels or pair_labels
        pair_order_key = _widget_key(filename, selected_plot, "pair_order")
        if plot_changed:
            st.session_state[pair_order_key] = pair_default
            st.session_state[f"{pair_order_key}__order"] = list(pair_default)
        pair_selection = _ordered_selection(
            "Staples to show",
            pair_labels,
            default=pair_default,
            key=pair_order_key,
        )
        # st.caption("Only selected pairwise staples will be shown. Order follows your selection.")
        pairs = [pair_options[pair_labels.index(label)] for label in pair_selection]

        title_row = st.columns([1.1, 2.4], gap="small")
        title_en_key = _widget_key(filename, selected_plot, "title_enabled")
        _set_if_missing(title_en_key, plot_cfg.get("title_enabled", False))
        use_custom_title = title_row[0].checkbox("Title", key=title_en_key)
        title_text_key = _widget_key(filename, selected_plot, "title_text")
        _set_if_missing(title_text_key, plot_cfg.get("title_text", f"{value} by {group}"))
        custom_title = title_row[1].text_input(
            "Title text",
            value=st.session_state[title_text_key],
            disabled=not use_custom_title,
            label_visibility="collapsed",
            placeholder="Title",
            key=title_text_key,
        )

        xlabel_row = st.columns([1.1, 2.4], gap="small")
        xlabel_en_key = _widget_key(filename, selected_plot, "xlabel_enabled")
        _set_if_missing(xlabel_en_key, plot_cfg.get("xlabel_enabled", False))
        use_custom_xlabel = xlabel_row[0].checkbox("Custom x label", key=xlabel_en_key)
        xlabel_text_key = _widget_key(filename, selected_plot, "xlabel_text")
        _set_if_missing(xlabel_text_key, plot_cfg.get("xlabel_text", ""))
        custom_xlabel = xlabel_row[1].text_input(
            "X label text",
            value=st.session_state[xlabel_text_key],
            disabled=not use_custom_xlabel,
            label_visibility="collapsed",
            placeholder="X label",
            key=xlabel_text_key,
        )

        ylabel_row = st.columns([1.1, 2.4], gap="small")
        ylabel_en_key = _widget_key(filename, selected_plot, "ylabel_enabled")
        _set_if_missing(ylabel_en_key, plot_cfg.get("ylabel_enabled", False))
        use_custom_ylabel = ylabel_row[0].checkbox("Custom y label", key=ylabel_en_key)
        ylabel_text_key = _widget_key(filename, selected_plot, "ylabel_text")
        _set_if_missing(ylabel_text_key, plot_cfg.get("ylabel_text", ""))
        custom_ylabel = ylabel_row[1].text_input(
            "Y label text",
            value=st.session_state[ylabel_text_key],
            disabled=not use_custom_ylabel,
            label_visibility="collapsed",
            placeholder="Y label",
            key=ylabel_text_key,
        )

    with top_empty_col:
        st.subheader("Plot options")
        # st.caption("If we need them...")

        button_cols = st.columns(2, gap="small")
        rotate_key = _widget_key(filename, selected_plot, "rotate_xticks")
        _init_widget(rotate_key, plot_cfg.get("rotate_xticks", False))
        rotate_xticks = button_cols[0].checkbox("Rotate x labels", key=rotate_key)

        yzero_key = _widget_key(filename, selected_plot, "y_zero")
        _init_widget(yzero_key, plot_cfg.get("y_zero", True))
        y_zero = button_cols[1].checkbox("Y-axis starts at 0", key=yzero_key)

        bar_fill_key = _widget_key(filename, selected_plot, "bar_fill")
        _set_if_missing(bar_fill_key, plot_cfg.get("bar_fill", "transparent"))
        bar_fill = st.selectbox(
            "Bar fill",
            ["block", "transparent", "none"],
            key=bar_fill_key,
        )
        
        style_labels = list(style_map.keys())
        style_key = _widget_key(filename, selected_plot, "style_label")
        _set_if_missing(style_key, plot_cfg.get("style_label", style_labels[0]))
        style_label = _cycle_select("Matplotlib style", style_labels, key=style_key, default=plot_cfg.get("style_label", style_labels[0]))
        style_choice = style_map[style_label]

        colors_key = _widget_key(filename, selected_plot, "use_group_colors")
        _set_if_missing(colors_key, plot_cfg.get("use_group_colors", True))
        use_group_colors = st.checkbox("Color groups", key=colors_key)
        color_cycles = ["tab10", "Set2", "Set3", "colorblind", "deep", "muted", "pastel", "dark", "bright"]
        cycle_key = _widget_key(filename, selected_plot, "color_cycle")
        _set_if_missing(cycle_key, plot_cfg.get("color_cycle", color_cycles[0]))
        if use_group_colors:
            color_cycle = _cycle_select("Group color cycle", color_cycles, key=cycle_key, default=plot_cfg.get("color_cycle", color_cycles[0]))
        else:
            color_cycle = color_cycles[0]

        scale_cols = st.columns(4, gap="small")
        plot_scale_key = _widget_key(filename, selected_plot, "plot_scale")
        plot_scale = scale_cols[0].slider("Plot scale", min_value=0.6, max_value=1.4, value=plot_cfg.get("plot_scale", 1.0), step=0.1, key=plot_scale_key)
        x_scale_key = _widget_key(filename, selected_plot, "x_scale")
        x_scale = scale_cols[1].slider("X scale", min_value=0.6, max_value=1.4, value=plot_cfg.get("x_scale", 1.0), step=0.1, key=x_scale_key)
        y_scale_key = _widget_key(filename, selected_plot, "y_scale")
        y_scale = scale_cols[2].slider("Y scale", min_value=0.6, max_value=1.4, value=plot_cfg.get("y_scale", 1.0), step=0.1, key=y_scale_key)
        staple_key = _widget_key(filename, selected_plot, "staple_scale")
        staple_scale = scale_cols[3].slider("Staple spacing", min_value=0.6, max_value=1.8, value=plot_cfg.get("staple_scale", 1.0), step=0.1, key=staple_key)

    # Persist current plot config on every run.
    plot_cfg = {
        "group": group,
        "value": value,
        "hue": hue_choice,
        "elements": elements,
        "whisker_mode": whisker_mode,
        "bar_mode": bar_mode,
        "bar_fill": bar_fill,
        "use_group_colors": use_group_colors,
        "color_cycle": color_cycle,
        "plot_scale": plot_scale,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "order": order,
        "pairs": pairs,
        "title_enabled": use_custom_title,
        "title_text": custom_title,
        "xlabel_enabled": use_custom_xlabel,
        "xlabel_text": custom_xlabel,
        "ylabel_enabled": use_custom_ylabel,
        "ylabel_text": custom_ylabel,
        "staple_scale": staple_scale,
        "style_label": style_label,
        "style_choice": style_choice,
        "wide_form": wide_form,
        "wide_id_cols": id_keep if wide_form else [],
        "row_include": st.session_state.get(include_key, [True] * len(df)),
        "rotate_xticks": rotate_xticks,
        "y_zero": y_zero,
    }
    file_cfg["plots"][selected_plot] = plot_cfg
    file_cfg["current"] = selected_plot
    _save_config(cfg)

    if apply_to_all:
        source_cfg = dict(plot_cfg)
        for plot_name in file_cfg["order"]:
            target_cfg = file_cfg["plots"].setdefault(plot_name, _default_plot_config())
            target_value = target_cfg.get("value", plot_name)
            merged_cfg = dict(source_cfg)
            merged_cfg["value"] = target_value
            file_cfg["plots"][plot_name] = merged_cfg
        _save_config(cfg)
        st.rerun()

    try:
        bottom_left, bottom_right = st.columns(2, gap="large")
        if not order:
            st.warning("Select at least one group to plot.")
            st.stop()

        with bottom_right:
            st.subheader("Data preview")
            edited = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", default=True, width="small"),
                },
                column_order=["Include"] + columns,
            )
        include_vals = edited["Include"].tolist() if "Include" in edited.columns else [True] * len(df)
        st.session_state[include_key] = include_vals
        df = edited

        if include_vals != include_series:
            st.rerun()

        df_plot = df[df[group].isin(order)].copy()

        if "Include" in df_plot.columns:
            df_plot = df_plot[df_plot["Include"]].drop(columns=["Include"])

        if "hist" in elements and len(elements) > 1:
            st.sidebar.warning("Histogram is best used alone. Showing histogram only.")
            elements = ["hist"]

        n_groups = max(int(df_plot[group].nunique(dropna=True)), 1)
        base_w, base_h = core.compute_figsize(n_groups, scale=0.7)
        figsize = (base_w * plot_scale * x_scale, base_h * plot_scale * y_scale)

        title_text = custom_title if use_custom_title else ""
        xlabel_text = custom_xlabel if use_custom_xlabel else None
        ylabel_text = custom_ylabel if use_custom_ylabel else None

        normality_p = _group_normality(df_plot, group, value, order)
        stat_test = _default_pairwise_test(normality_p)
        group_palette = color_cycle if use_group_colors else None

        ax, omnibus, pairs = core.plot_with_stats(
            df_plot,
            value=value,
            group=group,
            hue=hue,
            elements=elements,
            test=stat_test,
            title=title_text,
            xlabel=xlabel_text,
            ylabel=ylabel_text,
            style=style_choice,
            figsize=figsize,
            staple_scale=staple_scale,
            order=order,
            pairs=pairs,
            whisker_mode=whisker_mode,
            bar_mode=bar_mode,
            bar_fill=bar_fill,
            group_palette=group_palette,
            rotate_xticks=rotate_xticks,
            y_zero=y_zero,
        )

        with top_plot_col:
            # st.subheader("Plot")
            st.pyplot(ax.figure, clear_figure=False, use_container_width=False)

        with top_empty_col:
            def _fig_bytes(fmt: str) -> bytes:
                buf = io.BytesIO()
                ax.figure.savefig(buf, format=fmt, bbox_inches="tight")
                buf.seek(0)
                return buf.read()

            def _safe_file_stem(name: str) -> str:
                return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name).strip("._") or "plot"

            def _plot_df_for_cfg(cfg_item: dict) -> pd.DataFrame:
                df_item = raw_df.copy()
                if cfg_item.get("wide_form"):
                    id_cols = [c for c in cfg_item.get("wide_id_cols", []) if c in df_item.columns]
                    df_item = df_item.melt(id_vars=id_cols, var_name="group", value_name="value")
                return df_item

            def _all_plots_zip_bytes(fmt: str) -> bytes:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for plot_name in file_cfg["order"]:
                        cfg_item = file_cfg["plots"].get(plot_name, {})
                        group_i = cfg_item.get("group")
                        value_i = cfg_item.get("value")
                        if not group_i or not value_i:
                            continue
                        try:
                            df_item = _plot_df_for_cfg(cfg_item)
                            if group_i not in df_item.columns or value_i not in df_item.columns:
                                continue

                            include_i = cfg_item.get("row_include")
                            if isinstance(include_i, list) and len(include_i) == len(df_item):
                                df_item = df_item.assign(Include=include_i)
                                df_item = df_item[df_item["Include"]].drop(columns=["Include"])

                            order_i = [g for g in cfg_item.get("order", []) if g in set(df_item[group_i].dropna().unique())]
                            if not order_i:
                                order_i = list(pd.Series(df_item[group_i]).dropna().unique())
                            if not order_i:
                                continue

                            df_plot_i = df_item[df_item[group_i].isin(order_i)].copy()
                            if df_plot_i.empty:
                                continue

                            elements_i = cfg_item.get("elements", ["strip", "bar", "whisker"])
                            if "hist" in elements_i and len(elements_i) > 1:
                                elements_i = ["hist"]

                            n_groups_i = max(int(df_plot_i[group_i].nunique(dropna=True)), 1)
                            base_w_i, base_h_i = core.compute_figsize(n_groups_i, scale=0.7)
                            plot_scale_i = cfg_item.get("plot_scale", 1.0)
                            x_scale_i = cfg_item.get("x_scale", 1.0)
                            y_scale_i = cfg_item.get("y_scale", 1.0)
                            figsize_i = (base_w_i * plot_scale_i * x_scale_i, base_h_i * plot_scale_i * y_scale_i)

                            hue_choice_i = cfg_item.get("hue", "(none)")
                            hue_i = None if hue_choice_i == "(none)" else hue_choice_i
                            if hue_i and hue_i not in df_plot_i.columns:
                                hue_i = None

                            title_i = cfg_item.get("title_text", "") if cfg_item.get("title_enabled") else ""
                            xlabel_i = cfg_item.get("xlabel_text", "") if cfg_item.get("xlabel_enabled") else None
                            ylabel_i = cfg_item.get("ylabel_text", "") if cfg_item.get("ylabel_enabled") else None

                            normality_i = _group_normality(df_plot_i, group_i, value_i, order_i)
                            stat_test_i = _default_pairwise_test(normality_i)

                            pairs_i = []
                            for pair in cfg_item.get("pairs", []):
                                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    a, b = pair
                                    if a in order_i and b in order_i:
                                        pairs_i.append((a, b))

                            group_palette_i = cfg_item.get("color_cycle") if cfg_item.get("use_group_colors", True) else None

                            ax_i, _, _ = core.plot_with_stats(
                                df_plot_i,
                                value=value_i,
                                group=group_i,
                                hue=hue_i,
                                elements=elements_i,
                                test=stat_test_i,
                                title=title_i,
                                xlabel=xlabel_i,
                                ylabel=ylabel_i,
                                style=cfg_item.get("style_choice"),
                                figsize=figsize_i,
                                staple_scale=cfg_item.get("staple_scale", 1.0),
                                order=order_i,
                                pairs=pairs_i,
                                whisker_mode=cfg_item.get("whisker_mode", "quartiles"),
                                bar_mode=cfg_item.get("bar_mode", "median"),
                                bar_fill=cfg_item.get("bar_fill", "block"),
                                group_palette=group_palette_i,
                                rotate_xticks=cfg_item.get("rotate_xticks", False),
                                y_zero=cfg_item.get("y_zero", False),
                            )

                            fig_i = ax_i.figure
                            fig_buf = io.BytesIO()
                            fig_i.savefig(fig_buf, format=fmt, bbox_inches="tight")
                            plt.close(fig_i)
                            fig_buf.seek(0)
                            zf.writestr(f"{_safe_file_stem(plot_name)}.{fmt}", fig_buf.read())
                        except Exception:
                            continue

                zip_buf.seek(0)
                return zip_buf.read()

            def _zip_state_key(fmt: str) -> str:
                return f"{filename}:zip_bytes:{fmt}"

            dl_sections = st.columns(2, gap="small")
            with dl_sections[0]:
                st.caption("Download current plot")
                dcols = st.columns(3, gap="small")
                dcols[0].download_button(
                    "PNG",
                    data=_fig_bytes("png"),
                    file_name="grism_plot.png",
                    mime="image/png",
                )
                dcols[1].download_button(
                    "SVG",
                    data=_fig_bytes("svg"),
                    file_name="grism_plot.svg",
                    mime="image/svg+xml",
                )
                dcols[2].download_button(
                    "PDF",
                    data=_fig_bytes("pdf"),
                    file_name="grism_plot.pdf",
                    mime="application/pdf",
                )
            with dl_sections[1]:
                st.caption("Generate zip of all plots")
                zcols = st.columns(3, gap="small")
                if zcols[0].button("PNG", key=f"{filename}:download_zip:png"):
                    with st.spinner("Building PNG zip..."):
                        st.session_state[_zip_state_key("png")] = _all_plots_zip_bytes("png")
                if zcols[1].button("SVG", key=f"{filename}:download_zip:svg"):
                    with st.spinner("Building SVG zip..."):
                        st.session_state[_zip_state_key("svg")] = _all_plots_zip_bytes("svg")
                if zcols[2].button("PDF", key=f"{filename}:download_zip:pdf"):
                    with st.spinner("Building PDF zip..."):
                        st.session_state[_zip_state_key("pdf")] = _all_plots_zip_bytes("pdf")

                dzcols = st.columns(3, gap="small")
                png_zip = st.session_state.get(_zip_state_key("png"))
                svg_zip = st.session_state.get(_zip_state_key("svg"))
                pdf_zip = st.session_state.get(_zip_state_key("pdf"))
                dzcols[0].download_button(
                    "Get",
                    data=png_zip or b"",
                    file_name="grism_all_plots_png.zip",
                    mime="application/zip",
                    disabled=png_zip is None,
                )
                dzcols[1].download_button(
                    "Get",
                    data=svg_zip or b"",
                    file_name="grism_all_plots_svg.zip",
                    mime="application/zip",
                    disabled=svg_zip is None,
                )
                dzcols[2].download_button(
                    "Get",
                    data=pdf_zip or b"",
                    file_name="grism_all_plots_pdf.zip",
                    mime="application/zip",
                    disabled=pdf_zip is None,
                )

        with bottom_left:
            st.subheader("Stats")
            st.caption(f"Normality tests -> pairwise test: {stat_test}")

            error_groups = [g for g in order if normality_p.get(g) is None]
            if error_groups:
                st.warning(f"Normality test error for: {', '.join(error_groups)}")

            normality_table = pd.DataFrame(
                [
                    ["Normal (Shapiro p>=0.05)"] + [_normality_label(normality_p[g]) for g in order],
                    ["Shapiro p-value"] + [
                        "error" if normality_p[g] is None else f"{normality_p[g]:.3g}" for g in order
                    ],
                ],
                columns=["metric"] + order,
            ).set_index("metric")
            st.table(normality_table)

            if pairs:
                st.caption("Pairwise comparisons")
                pair_rows = [
                    {
                        "group_a": r.group_a,
                        "group_b": r.group_b,
                        "test": r.test,
                        "statistic": r.statistic,
                        "pvalue": r.pvalue,
                    }
                    for r in pairs
                ]
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
