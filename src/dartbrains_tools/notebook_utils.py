"""
Notebook UI helpers for marimo tutorials.

``marimo`` and ``plotly`` are imported lazily inside each helper so the
module loads cleanly in environments that don't have the optional
``notebook`` extra installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def youtube(video_id: str):
    """Embed a YouTube video by video ID.

    Usage:
        from dartbrains_tools.notebook_utils import youtube
        youtube("dQw4w9WgXcQ")

    The explicit ``referrerpolicy`` is required, not cosmetic. marimo's server
    responds with ``Referrer-Policy: same-origin``, which sends no ``Referer``
    at all on cross-origin requests. YouTube's player uses that header to
    identify the embedding site and, without it, refuses to play with
    "Error 153: Video player configuration error". An iframe-level
    ``referrerpolicy`` overrides the document policy for that one request, so
    the origin — and only the origin — reaches YouTube.
    """
    import marimo as mo

    return mo.Html(
        f'<iframe width="560" height="315" '
        f'src="https://www.youtube.com/embed/{video_id}" '
        f'frameborder="0" allowfullscreen '
        f'referrerpolicy="strict-origin-when-cross-origin"></iframe>'
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


def assignment_card(
    slug: str,
    *,
    title: str | None = None,
    server: str | None = None,
    course: str | None = None,
    term: str | None = None,
):
    """A card linking to this chapter's assignment on the grader.

    The assignment lives on the grader, not in the chapter: one link opens
    it in molab (the grader hands molab the version it currently publishes),
    the other downloads the same notebook for a laptop. Where it lives comes
    from the chapter's ``[tool.grader]`` block (written by marimo-book's
    ``sync-deps``), so a new term never touches the chapter. On the static
    site the page's own assignment drawer already does this; the card is for
    molab and local runs.
    """
    import os

    import marimo as mo

    from .storage import _auth, _notebook

    if os.environ.get("GRADER_RENDER"):
        # The static site: the page's own assignment drawer already carries
        # these links, so the card would be a duplicate there.
        return mo.Html("")

    server = (server or _notebook.resolve_server(None) or _auth.DEFAULT_SERVER).rstrip("/")
    course = course or _notebook.resolve_course()
    term = term or _notebook.resolve_term()
    if not course or not term:
        return mo.md(
            f"*Assignment `{slug}`: this notebook does not say which course and term it belongs "
            "to (no `[tool.grader]` in its script block), so the assignment link cannot be built.*"
        )
    base = f"{server}/a/{course}/{term}/{slug}"
    name = title or slug.replace("-", " ").replace("_", " ").title()
    return mo.Html(
        '<div style="border:1px solid var(--border-color,#ddd);border-radius:8px;'
        'padding:0.9rem 1.1rem;margin:0.5rem 0">'
        f'<div style="font-weight:600;margin-bottom:0.35rem">Assignment: {name}</div>'
        '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;align-items:center">'
        f'<a href="{base}/molab" target="_blank" rel="noopener" '
        'style="display:inline-block;padding:0.35rem 0.8rem;border-radius:6px;'
        'background:#00693E;color:#fff;text-decoration:none">Open in molab</a>'
        f'<a href="{base}/student.py" target="_blank" rel="noopener">Download the notebook</a>'
        '<span style="opacity:0.7">Sign in with Dartmouth inside it to submit.</span>'
        "</div></div>"
    )
