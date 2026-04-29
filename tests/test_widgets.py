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


# Expected Python trait defaults — the JS-side `model.get(...) ?? <default>` reads
# in each widget's animate() loop must mirror these so the rAF loop renders sane
# values during the brief window before the first Python→JS trait sync.
EXPECTED_DEFAULTS = {
    PrecessionWidget: {
        "b0": 3.0,
        "flip_angle": 90.0,
        "t1": 0.0,
        "t2": 0.0,
        "show_relaxation": False,
        "paused": False,
    },
    SpinEnsembleWidget: {
        "sequence_type": "spin_echo",
        "speed": 1.0,
        "paused": False,
    },
    CompassWidget: {"b0": 3.0},
    NetMagnetizationWidget: {"n_protons": 100, "b0_on": False},
    KSpaceWidget: {
        "mask_type": "progressive",
        "radius_fraction": 0.2,
        "speed": 1.0,
    },
    ConvolutionWidget: {"pattern": "single", "speed": 1.0},
    EncodingWidget: {"speed": 1.0},
    TransformCubeWidget: {
        "trans_x": 0.0,
        "trans_y": 0.0,
        "trans_z": 0.0,
        "rot_x": 0.0,
        "rot_y": 0.0,
        "rot_z": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "scale_z": 1.0,
    },
    CostFunctionWidget: {"trans_x": 0.0, "trans_y": 0.0},
    SmoothingWidget: {"fwhm": 0.0},
}


@pytest.mark.parametrize(
    "widget_cls,expected", list(EXPECTED_DEFAULTS.items()), ids=lambda x: getattr(x, "__name__", "")
)
def test_widget_python_defaults(widget_cls, expected):
    """Python trait defaults must match the JS-side `?? <default>` fallbacks.

    The widget JS reads `model.get("trait") ?? <fallback>` in its animate() loop
    to avoid rAF-before-sync TypeErrors. If a Python default is changed without
    updating the matching JS literal, the widget will briefly render a stale
    value on first paint until the first user interaction. This test pins the
    contract.
    """
    w = widget_cls()
    for trait, value in expected.items():
        assert getattr(w, trait) == value, (
            f"{widget_cls.__name__}.{trait}: expected {value!r}, got {getattr(w, trait)!r}"
        )


@pytest.mark.parametrize("widget_cls", ALL_WIDGETS, ids=lambda c: c.__name__)
def test_widget_js_has_animate_guard(widget_cls):
    """Each widget's animate() body is wrapped in a log-once try/catch.

    Defense-in-depth: if a future trait is misnamed or removed, the rAF loop
    catches the error and logs once via console.warn, instead of spamming 60
    errors/sec to the browser console for every frame until interaction.
    """
    js = str(widget_cls._esm)
    assert "_animateErrLogged" in js, (
        f"{widget_cls.__name__}: missing log-once guard flag in {widget_cls._esm._path.name}"
    )
    assert "console.warn(" in js, (
        f"{widget_cls.__name__}: missing console.warn() in animate() catch block"
    )
