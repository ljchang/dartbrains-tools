"""
Notebook UI helpers for marimo tutorials.

``marimo`` and ``plotly`` are imported lazily inside each helper so the
module loads cleanly in environments that don't have the optional
``notebook`` extra installed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
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


def assignment_card(slug: str, **kwargs):
    """A card linking to this chapter's assignment on the grader.

    The client's :func:`marimo_grader_client.assignment_card`, with DartBrains'
    grader as the default server.
    """
    from marimo_grader_client.assignments import assignment_card as _card

    from . import storage  # noqa: F401 - sets the default server

    return _card(slug, **kwargs)


# The notebook molab runs comes from the repository's master branch, so its
# figures come from the same place: a figure added to a chapter is reachable
# the moment the merge lands, not after the next site build. The published
# site (https://dartbrains.org/images) also serves them; DARTBRAINS_IMAGES_URL
# switches the fallback without a release.
IMAGES_URL = "https://raw.githubusercontent.com/ljchang/dartbrains/master/images"


def image(rel: str, **kwargs):
    """A figure from the book's ``images/`` directory, wherever the notebook runs.

    In a checkout of the book -- ``marimo edit``, the site build -- the file is
    at ``<repo>/images/<rel>`` and its bytes are embedded (the static site
    then re-encodes them). Anywhere else -- molab, a downloaded notebook --
    only the notebook file exists, so the same image is fetched from the
    repository on GitHub instead. ``DARTBRAINS_IMAGES`` overrides the local root,
    ``DARTBRAINS_IMAGES_URL`` the fallback.

    Usage::

        image("single_subject/MultipleRegression.png")
    """
    import marimo as mo

    rel = rel.strip("/")
    for root in _image_roots():
        p = root / rel
        if p.is_file():
            return mo.image(p, **kwargs)
    base = os.environ.get("DARTBRAINS_IMAGES_URL", IMAGES_URL).rstrip("/")
    return mo.image(f"{base}/{rel}", **kwargs)


def _image_roots():
    from marimo_grader_client.notebook import notebook_path

    override = os.environ.get("DARTBRAINS_IMAGES")
    if override:
        yield Path(override)
    nb = notebook_path()
    if nb is not None:
        # content/<chapter>.py -> <repo>/images ; also a notebook beside images/
        for up in (nb.resolve().parent.parent, nb.resolve().parent):
            yield up / "images"
