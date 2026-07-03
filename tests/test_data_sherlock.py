"""Shape tests for the sherlock dataset module -- does not hit the network."""

import pytest

from dartbrains_tools.data import sherlock


def test_constants():
    assert sherlock.REPO_ID == "dartbrains/sherlock"
    assert sherlock.TR == 1.5
    assert set(sherlock.TASKS) == {"sherlockPart1", "sherlockPart2", "freerecall"}
    assert len(sherlock.SUBJECTS) == 16
    assert sherlock.SUBJECTS[0] == "sub-01"
    assert sherlock.SUBJECTS[-1] == "sub-16"


def test_get_subjects():
    s = sherlock.get_subjects()
    assert isinstance(s, list)
    assert len(s) == 16


def test_get_file_bold_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
    assert captured["f"] == (
        "derivatives/fmriprep/sub-01/func/"
        "sub-01_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_bold_denoised_cropped_smoothed(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file(
        "sub-01",
        task="sherlockPart1",
        suffix="bold",
        denoised=True,
        cropped=True,
        smoothed=True,
    )
    assert captured["f"] == (
        "derivatives/fmriprep/sub-01/func/"
        "sub-01_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_confounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="confounds")
    assert captured["f"] == (
        "derivatives/fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-confounds_regressors.tsv"
    )


def test_get_file_mask(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="mask")
    assert captured["f"] == (
        "derivatives/fmriprep/sub-01/func/"
        "sub-01_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )


def test_get_file_t1w_ignores_task(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="T1w")
    assert captured["f"] == (
        "derivatives/fmriprep/sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )


def test_get_file_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        sherlock.get_file("sub-01", task="bogus", suffix="bold")


def test_get_file_unknown_suffix_raises():
    with pytest.raises(ValueError, match="Unknown suffix"):
        sherlock.get_file("sub-01", task="sherlockPart1", suffix="bogus")


def test_get_stimulus_video(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_stimulus_video()
    assert captured["f"] == "stimuli/stimuli_Sherlock.m4v"


def test_load_onsets_kind(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f) or "/dev/null")
    # Use a dummy read_csv that returns a sentinel so we don't actually parse /dev/null
    monkeypatch.setattr(
        "dartbrains_tools.data.sherlock.pd.read_csv",
        lambda *a, **kw: "sentinel",
    )
    sherlock.load_onsets("watch")
    assert captured["f"] == "onsets/Sherlock_Watch_Scene_N50_Onsets.csv"
