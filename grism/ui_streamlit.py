"""Streamlit UI for grism."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy import stats as sps

import grism as core


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


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



def _group_normality(df: pd.DataFrame, group: str, value: str, order: list[str]) -> Dict[str, Optional[float]]:
    pvals: Dict[str, Optional[float]] = {}
    for g in order:
        vals = df.loc[df[group] == g, value].dropna().to_numpy()
        if vals.size < 3:
            pvals[g] = None
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
        return "n/a"
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
    st.title("grism")
    # st.caption("Simple biology-style plots with stats and annotations")

    st.sidebar.header("Basic setup")
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if not uploaded:
        st.info("Upload a file to begin.")
        st.stop()

    df = _load_dataframe(uploaded.getvalue(), uploaded.name)
    columns = list(df.columns)

    if not columns:
        st.warning("No columns found in the uploaded file.")
        st.stop()

    default_group = _default_group_column(df, columns)
    group = st.sidebar.selectbox(
        "Group column",
        columns,
        index=columns.index(default_group),
    )

    default_value = _default_value_column(df, columns, group)
    value = st.sidebar.selectbox(
        "Value column",
        columns,
        index=columns.index(default_value),
    )

    if group == value:
        st.sidebar.warning("Group and value columns should differ.")

    hue_options = ["(none)"] + [c for c in columns if c not in {value, group}]
    hue_choice = st.sidebar.selectbox("Hue (optional)", hue_options)
    hue: Optional[str] = None if hue_choice == "(none)" else hue_choice

    elements = st.sidebar.multiselect(
        "Elements",
        ["strip", "bar", "whisker", "hist"],
        default=["strip", "bar", "whisker"],
    )

    plot_scale = st.sidebar.slider("Plot scale", min_value=0.6, max_value=1.4, value=1.0, step=0.1)
    x_scale = st.sidebar.slider("X scale", min_value=0.6, max_value=1.4, value=1.0, step=0.1)
    y_scale = st.sidebar.slider("Y scale", min_value=0.6, max_value=1.4, value=1.0, step=0.1)

    top_plot_col, top_style_col, top_empty_col = st.columns([2, 1, 1], gap="large")

    style_map = _style_options()
    with top_style_col:
        st.subheader("Plot setup")

        group_values = list(pd.Series(df[group]).dropna().unique())
        order = st.multiselect(
            "Groups to plot",
            group_values,
            default=group_values,
            help="Only selected groups will be plotted.",
        )

        title_row = st.columns([1.1, 2.4], gap="small")
        use_custom_title = title_row[0].checkbox("Title", value=False)
        custom_title = title_row[1].text_input(
            "Title text",
            value=f"{value} by {group}",
            disabled=not use_custom_title,
            label_visibility="collapsed",
            placeholder="Title",
        )

        xlabel_row = st.columns([1.1, 2.4], gap="small")
        use_custom_xlabel = xlabel_row[0].checkbox("Custom x label", value=False)
        custom_xlabel = xlabel_row[1].text_input(
            "X label text",
            value="",
            disabled=not use_custom_xlabel,
            label_visibility="collapsed",
            placeholder="X label",
        )

        ylabel_row = st.columns([1.1, 2.4], gap="small")
        use_custom_ylabel = ylabel_row[0].checkbox("Custom y label", value=False)
        custom_ylabel = ylabel_row[1].text_input(
            "Y label text",
            value="",
            disabled=not use_custom_ylabel,
            label_visibility="collapsed",
            placeholder="Y label",
        )

    with top_empty_col:
        st.subheader("More options")
        # st.caption("If we need them...")

        staple_scale = st.slider(
            "Staple spacing scale",
            min_value=0.6,
            max_value=1.8,
            value=1.0,
            step=0.1,
        )
        style_label = st.selectbox("Matplotlib style", list(style_map.keys()), index=0)
        style_choice = style_map[style_label]

    try:
        if not order:
            st.warning("Select at least one group to plot.")
            st.stop()

        df_plot = df[df[group].isin(order)].copy()

        if "hist" in elements and len(elements) > 1:
            st.sidebar.warning("Histogram is best used alone. Showing histogram only.")
            elements = ["hist"]

        n_groups = max(int(df_plot[group].nunique(dropna=True)), 1)
        base_w, base_h = core._compute_figsize(n_groups, scale=0.7)
        figsize = (base_w * plot_scale * x_scale, base_h * plot_scale * y_scale)

        title_text = custom_title if use_custom_title else ""
        xlabel_text = custom_xlabel if use_custom_xlabel else None
        ylabel_text = custom_ylabel if use_custom_ylabel else None

        normality_p = _group_normality(df_plot, group, value, order)
        stat_test = _default_pairwise_test(normality_p)

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
        )

        with top_plot_col:
            # st.subheader("Plot")
            st.pyplot(ax.figure, clear_figure=True, use_container_width=False)
            st.caption("Use the plot toolbar to download PNG/SVG/PDF.")

        bottom_left, bottom_right = st.columns(2, gap="large")

        with bottom_left:
            st.subheader("Stats")
            st.caption(f"Pairwise test: {stat_test}")

            normality_table = pd.DataFrame(
                [
                    ["Normal (Shapiro p>=0.05)"] + [_normality_label(normality_p[g]) for g in order],
                    ["Shapiro p-value"] + [
                        "n/a" if normality_p[g] is None else f"{normality_p[g]:.3g}" for g in order
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

        with bottom_right:
            st.subheader("Data preview")
            st.dataframe(df.head(20), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
