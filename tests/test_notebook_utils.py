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
