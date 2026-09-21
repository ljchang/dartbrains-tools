"""Tests for ``dartbrains_tools.notebook_utils`` helpers."""

import importlib.util

import numpy as np
import pytest

from dartbrains_tools.notebook_utils import plot_timeseries, youtube

requires_marimo = pytest.mark.skipif(
    importlib.util.find_spec("marimo") is None,
    reason="marimo is an optional `notebook` extra; install with `pip install dartbrains-tools[notebook]`",
)


@requires_marimo
def test_youtube_embeds_iframe_with_video_id():
    out = youtube("dQw4w9WgXcQ")
    html = out.text if hasattr(out, "text") else str(out)
    assert "dQw4w9WgXcQ" in html
    assert "<iframe" in html
    assert "youtube.com/embed" in html


@requires_marimo
def test_youtube_sets_referrerpolicy():
    """The embed must carry its own ``referrerpolicy``.

    marimo's server responds with ``Referrer-Policy: same-origin``, which sends
    no ``Referer`` at all on cross-origin requests. YouTube's player uses that
    header to identify the embedding site, and without it refuses to play with
    "Error 153: Video player configuration error". An iframe-level
    ``referrerpolicy`` overrides the document policy for that request, so the
    origin — and only the origin — reaches YouTube.

    Verified in a browser against a page serving ``Referrer-Policy:
    same-origin``: without the attribute the player shows Error 153; with it the
    video plays.
    """
    out = youtube("dQw4w9WgXcQ")
    html = out.text if hasattr(out, "text") else str(out)
    assert 'referrerpolicy="strict-origin-when-cross-origin"' in html


def test_plot_timeseries_1d_returns_single_trace_figure():
    fig = plot_timeseries(np.arange(20))
    # plotly Figure with a single Scatter trace
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "Signal 1"
    # legend hidden for single unlabeled trace
    assert fig.layout.showlegend is False


def test_plot_timeseries_2d_treats_columns_as_separate_signals():
    data = np.column_stack([np.arange(20), np.arange(20) * 2, np.arange(20) * 3])
    fig = plot_timeseries(data, labels=["a", "b", "c"], title="Three signals")
    assert len(fig.data) == 3
    assert [t.name for t in fig.data] == ["a", "b", "c"]
    assert fig.layout.title.text == "Three signals"
    assert fig.layout.showlegend is True


def test_plot_timeseries_label_count_must_match_signal_count():
    with pytest.raises(ValueError, match="same number of labels as columns"):
        plot_timeseries(
            np.column_stack([np.arange(5), np.arange(5)]),
            labels=["only one"],
        )


def test_plot_timeseries_axis_titles_are_overridable():
    fig = plot_timeseries(np.arange(5), xaxis_title="frame", yaxis_title="amp")
    assert fig.layout.xaxis.title.text == "frame"
    assert fig.layout.yaxis.title.text == "amp"


def test_image_uses_the_repo_file_when_present_else_the_site(tmp_path, monkeypatch):
    pytest.importorskip("marimo")
    from dartbrains_tools.notebook_utils import image

    monkeypatch.delenv("DARTBRAINS_IMAGES", raising=False)
    monkeypatch.delenv("DARTBRAINS_IMAGES_URL", raising=False)
    # A checkout: content/<nb>.py beside images/<section>/<file>
    (tmp_path / "content").mkdir()
    nb = tmp_path / "content" / "ch.py"
    nb.write_text("import marimo\n")
    (tmp_path / "images" / "glm").mkdir(parents=True)
    (tmp_path / "images" / "glm" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("DARTBRAINS_NOTEBOOK", str(nb))
    assert "data:image/png;base64" in image("glm/fig.png").text
    # molab / a downloaded notebook: only the notebook exists
    # a molab sandbox: nothing two levels up but the sandbox itself
    (tmp_path / "sandbox" / "session" / "nb").mkdir(parents=True)
    alone = tmp_path / "sandbox" / "session" / "nb" / "notebook.py"
    alone.write_text("import marimo\n")
    monkeypatch.setenv("DARTBRAINS_NOTEBOOK", str(alone))
    assert (
        "https://raw.githubusercontent.com/ljchang/dartbrains/master/images/glm/fig.png"
        in image("glm/fig.png").text
    )
    monkeypatch.setenv("DARTBRAINS_IMAGES_URL", "https://mirror.example/img/")
    assert "https://mirror.example/img/glm/fig.png" in image("/glm/fig.png").text
