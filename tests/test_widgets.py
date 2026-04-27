"""Verify each widget instantiates and its bundled JS file ships with the package."""

from pathlib import Path

import pytest

from dartbrains_tools.mr_widgets import (
    CompassWidget,
    ConvolutionWidget,
    CostFunctionWidget,
    EncodingWidget,
    KSpaceWidget,
    NetMagnetizationWidget,
    PrecessionWidget,
    SmoothingWidget,
    SpinEnsembleWidget,
    TransformCubeWidget,
)

ALL_WIDGETS = [
    CompassWidget,
    ConvolutionWidget,
    CostFunctionWidget,
    EncodingWidget,
    KSpaceWidget,
    NetMagnetizationWidget,
    PrecessionWidget,
    SmoothingWidget,
    SpinEnsembleWidget,
    TransformCubeWidget,
]


@pytest.mark.parametrize("widget_cls", ALL_WIDGETS, ids=lambda c: c.__name__)
def test_widget_instantiates(widget_cls):
    w = widget_cls()
    assert w is not None


@pytest.mark.parametrize("widget_cls", ALL_WIDGETS, ids=lambda c: c.__name__)
def test_widget_esm_loads_from_package(widget_cls):
    """Each widget's _esm wraps a Path inside the installed package and reads JS content.

    anywidget wraps Path-typed _esm attributes in a FileContents watcher; the
    underlying Path lives on `_path`. We confirm it points inside the installed
    dartbrains_tools package and that str(_esm) (which reads the file) is non-trivial.
    """
    esm = widget_cls._esm
    underlying_path = getattr(esm, "_path", None)
    assert isinstance(underlying_path, Path), (
        f"{widget_cls.__name__}._esm has no Path: {esm!r}"
    )
    assert underlying_path.exists(), (
        f"{widget_cls.__name__}._esm missing: {underlying_path}"
    )
    assert underlying_path.suffix == ".js"
    assert "dartbrains_tools" in str(underlying_path), (
        f"{widget_cls.__name__}._esm is outside the package: {underlying_path}"
    )
    contents = str(esm)
    assert len(contents) > 100, (
        f"{widget_cls.__name__}._esm content suspiciously short: {len(contents)} chars"
    )


def test_all_ten_js_files_present():
    """All 10 expected JS widget files ship with the package."""
    import dartbrains_tools

    js_dir = Path(dartbrains_tools.__file__).parent / "js"
    assert js_dir.is_dir()
    js_files = sorted(p.name for p in js_dir.glob("*.js"))
    expected = sorted(
        [
            "compass_widget.js",
            "convolution_widget.js",
            "cost_function_widget.js",
            "encoding_widget.js",
            "kspace_widget.js",
            "net_magnetization_widget.js",
            "precession_widget.js",
            "smoothing_widget.js",
            "spin_ensemble_widget.js",
            "transform_cube_widget.js",
        ]
    )
    assert js_files == expected
