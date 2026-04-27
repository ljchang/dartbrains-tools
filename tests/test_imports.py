"""Smoke tests: every public symbol imports cleanly."""


def test_top_level_package():
    import dartbrains_tools

    assert dartbrains_tools.__version__


def test_data_module():
    from dartbrains_tools.data import (
        CONDITIONS,
        REPO_ID,
        TR,
        get_file,
        get_subjects,
        get_tr,
        load_confounds,
        load_events,
    )

    assert REPO_ID == "dartbrains/localizer"
    assert isinstance(CONDITIONS, (list, tuple))
    assert TR > 0
    assert callable(get_file)
    assert callable(get_subjects)
    assert callable(get_tr)
    assert callable(load_confounds)
    assert callable(load_events)


def test_mr_simulations_module():
    from dartbrains_tools.mr_simulations import (
        GAMMA,
        GAMMA_H,
        TISSUE_PROPERTIES,
        apply_relaxation,
        apply_rf_pulse,
        compute_spectrum,
        fid_signal,
        gradient_echo_signal,
        hrf,
        image_to_kspace,
        kspace_to_image,
        mask_kspace,
        plot_contrast_bars,
        plot_kspace_and_image,
        plot_magnetization_3d,
        plot_pulse_sequence,
        plot_signal_timeline,
        rotation_x,
        rotation_y,
        rotation_z,
        simulate_bloch,
        spin_echo_signal,
        t1_recovery,
        t2_decay,
    )

    assert GAMMA_H > 0
    assert isinstance(GAMMA, dict)
    assert "1H" in GAMMA
    assert isinstance(TISSUE_PROPERTIES, dict)
    for fn in (
        apply_relaxation,
        apply_rf_pulse,
        compute_spectrum,
        fid_signal,
        gradient_echo_signal,
        hrf,
        image_to_kspace,
        kspace_to_image,
        mask_kspace,
        plot_contrast_bars,
        plot_kspace_and_image,
        plot_magnetization_3d,
        plot_pulse_sequence,
        plot_signal_timeline,
        rotation_x,
        rotation_y,
        rotation_z,
        simulate_bloch,
        spin_echo_signal,
        t1_recovery,
        t2_decay,
    ):
        assert callable(fn)


def test_mr_widgets_module():
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

    classes = [
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
    assert len(classes) == 10


def test_notebook_utils_module():
    # marimo is an optional dep; only import if available.
    try:
        from dartbrains_tools.notebook_utils import youtube
    except ImportError:
        import pytest

        pytest.skip("marimo not installed (notebook extra)")
    assert callable(youtube)
