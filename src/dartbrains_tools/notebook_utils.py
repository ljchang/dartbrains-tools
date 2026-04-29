"""
Notebook UI helpers for marimo tutorials.

``marimo`` and ``plotly`` are imported lazily inside each helper so the
module loads cleanly in environments that don't have the optional
``notebook`` extra installed.
"""

from __future__ import annotations

from typing import Any, Sequence


def youtube(video_id: str):
    """Embed a YouTube video by video ID.

    Usage:
        from dartbrains_tools.notebook_utils import youtube
        youtube("dQw4w9WgXcQ")
    """
    import marimo as mo

    return mo.Html(
        f'<iframe width="560" height="315" '
        f'src="https://www.youtube.com/embed/{video_id}" '
        f'frameborder="0" allowfullscreen></iframe>'
    )


def plot_timeseries(
    data: Any,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    linewidth: float = 2,
    xaxis_title: str = "Time (TR)",
    yaxis_title: str = "Intensity",
    height: int = 350,
):
    """Plot a 1D or 2D timeseries as an interactive plotly figure.

    Args:
        data: 1D array (a single trace) or 2D array shaped ``(n_timepoints,
            n_signals)`` — each column becomes a trace.
        labels: per-trace labels; length must match the number of columns
            when ``data`` is 2D. ``None`` falls back to "Signal 1", "Signal 2".
        title: optional figure title (also adjusts the top margin so the
            title isn't clipped).
        linewidth: trace width in pixels.
        xaxis_title / yaxis_title: axis labels. Defaults match the GLM /
            Group-Analysis tutorial convention ("Time (TR)" / "Intensity").
        height: figure height in pixels.

    Returns:
        A ``plotly.graph_objects.Figure``. Pass it as a cell's last
        expression to render in marimo, or call ``.show()`` to display
        in Jupyter.

    Raises:
        ValueError: when ``labels`` is provided and its length doesn't
            match the number of signal columns.

    Imports plotly + numpy lazily so importing this module from a
    minimal env (e.g. for the ``youtube`` helper alone) doesn't pull in
    the plotting stack.
    """
    import numpy as np
    import plotly.graph_objects as go

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_series = arr.shape[1]
    if labels is not None and len(labels) != n_series:
        raise ValueError(
            f"Need to have the same number of labels as columns in data "
            f"(got {len(labels)} labels for {n_series} signals)."
        )
    x = np.arange(arr.shape[0])
    fig = go.Figure()
    for i in range(n_series):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=arr[:, i],
                mode="lines",
                name=labels[i] if labels is not None else f"Signal {i + 1}",
                line=dict(width=linewidth),
                hovertemplate="t=%{x}<br>y=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 20, b=50),
        showlegend=labels is not None or n_series > 1,
    )
    return fig
