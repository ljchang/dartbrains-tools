"""Shape tests for the paranoia dataset module -- does not hit the network."""

import pytest

from dartbrains_tools.data import paranoia


def test_constants():
    assert paranoia.REPO_ID == "dartbrains/paranoia"
    assert paranoia.TR == 1.0
    assert paranoia.RUNS == (1, 2, 3)
    assert len(paranoia.SUBJECTS) == 22
    assert all(s.startswith("sub-tb") for s in paranoia.SUBJECTS)


def test_get_subjects():
    s = paranoia.get_subjects()
    assert isinstance(s, list)
    assert len(s) == 22


def test_get_runs():
    assert paranoia.get_runs() == [1, 2, 3]


def test_get_file_bold(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=1, suffix="bold")
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/"
        "sub-tb2994_task-story_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_bold_denoised_smoothed(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=2, suffix="bold", denoised=True, smoothed=True)
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/"
        "sub-tb2994_denoise_smooth6mm_task-story_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_confounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=3, suffix="confounds")
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/sub-tb2994_task-story_run-3_desc-confounds_regressors.tsv"
    )


def test_get_file_unknown_run_raises():
    with pytest.raises(ValueError, match="run"):
        paranoia.get_file("sub-tb2994", run=4, suffix="bold")


def test_get_file_unknown_suffix_raises():
    with pytest.raises(ValueError, match="Unknown suffix"):
        paranoia.get_file("sub-tb2994", run=1, suffix="bogus")


def test_load_participants(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f) or "/dev/null")
    monkeypatch.setattr(
        "dartbrains_tools.data.paranoia.pd.read_csv",
        lambda *a, **kw: "sentinel",
    )
    paranoia.load_participants()
    assert captured["f"] == "participants.tsv"


def test_load_transcript(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f) or "/dev/null")
    # Stub Path so we don't actually read /dev/null
    monkeypatch.setattr(
        "dartbrains_tools.data.paranoia.Path",
        lambda p: type("P", (), {"read_text": lambda self: "sentinel"})(),
    )
    out = paranoia.load_transcript(2)
    assert captured["f"] == "stimuli/paranoia_story2_transcript.txt"
    assert out == "sentinel"


def test_get_stimulus_audio(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_stimulus_audio(1)
    assert captured["f"] == "stimuli/stimuli_story1_audio.wav"
