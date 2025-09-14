# src/visualize.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import plotly.express as px


def _melt_long(
    df_annot: pd.DataFrame,
    features: Iterable[str],
    *,
    id_cols: Optional[List[str]] = None,
    index_name: str = "idx",
) -> pd.DataFrame:
    """Return a tidy/long dataframe with columns [idx, Variable, Value, ...id_cols].

    Parameters
    ----------
    df_annot : pd.DataFrame
        Annotated dataframe that already includes anomaly columns (anomaly_flag, anomaly_score).
    features : Iterable[str]
        Feature names to melt (e.g., ["NPHI", "RHOB"]).
    id_cols : list[str] | None
        Extra columns to keep alongside the melted values (defaults to ["anomaly_flag", "anomaly_score"]).
    index_name : str
        Name given to the reset index column.
    """
    if id_cols is None:
        id_cols = ["anomaly_flag", "anomaly_score"]

    # Preserve index as an explicit column for x-axis if needed
    if df_annot.index.name != index_name:
        long = df_annot.reset_index(drop=False).rename(columns={"index": index_name})
    else:
        long = df_annot.reset_index(drop=False)

    long = (
        long.melt(
            id_vars=[index_name] + id_cols,
            value_vars=list(features),
            var_name="Variable",
            value_name="Value",
        )
        .dropna(subset=["Value"])  # drop missing values so plots are clean
        .reset_index(drop=True)
    )
    return long


def plot_scatter_melted(
    df_annot: pd.DataFrame,
    features: Iterable[str],
    *,
    index_name: str = "idx",
    symbol_by_flag: bool = True,
    hover_extra: Optional[List[str]] = None,
    facet: bool = False,
    title: str = "Features (melted) with IsolationForest anomalies",
    show: bool = True,
    save_html: Optional[str | Path] = None,
):
    """Scatter of melted features over index, colored by Variable, symbol by anomaly flag.

    Returns the Plotly figure. Optionally shows and/or saves as an HTML file.
    """
    long = _melt_long(df_annot, features, index_name=index_name)
    hover = ["anomaly_score"] + (hover_extra or [])

    fig = px.scatter(
        long,
        x=index_name,
        y="Value",
        color="Variable",
        symbol="anomaly_flag" if symbol_by_flag else None,
        hover_data=hover,
        facet_col="Variable" if facet else None,
        title=title,
    )

    if save_html is not None:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
    if show:
        fig.show()
    return fig


def plot_score_hist(
    df_annot: pd.DataFrame,
    *,
    nbins: int = 50,
    by_flag: bool = True,
    title: str = "Anomaly score distribution",
    show: bool = True,
    save_html: Optional[str | Path] = None,
):
    """Histogram of anomaly scores, optionally colored by anomaly_flag."""
    fig = px.histogram(
        df_annot,
        x="anomaly_score",
        color="anomaly_flag" if by_flag else None,
        nbins=nbins,
        title=title,
    )
    if save_html is not None:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
    if show:
        fig.show()
    return fig


def plot_pair_scatter(
    df_annot: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str = "anomaly_flag",
    hover_extra: Optional[List[str]] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_html: Optional[str | Path] = None,
):
    """Direct feature-vs-feature scatter (e.g., RHOB vs NPHI) with anomaly overlay."""
    hover = ["anomaly_score"] + (hover_extra or [])
    fig = px.scatter(
        df_annot,
        x=x,
        y=y,
        color=color,
        hover_data=hover,
        title=title or f"{y} vs {x} (IsolationForest anomalies)",
    )
    if save_html is not None:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
    if show:
        fig.show()
    return fig


def plot_feature_box(
    df_annot: pd.DataFrame,
    features: Iterable[str],
    *,
    points: str | bool = "all",
    title: str = "Box plots of features",
    show: bool = True,
    save_html: Optional[str | Path] = None,
):
    """Box plots for the given features (melted), optionally showing points."""
    long = _melt_long(df_annot, features)
    fig = px.box(long, x="Variable", y="Value", color="Variable", points=points, title=title)
    if save_html is not None:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
    if show:
        fig.show()
    return fig
