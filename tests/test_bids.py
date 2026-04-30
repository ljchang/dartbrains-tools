"""Path-construction tests for the bids module.

Builds a fake BIDS tree in tmp_path; no network, no real data.
"""

import json

import pandas as pd
import pytest

from dartbrains_tools import bids


# ---------------------------------------------------------------------------
# Fixtures: build a minimal BIDS tree on disk
# ---------------------------------------------------------------------------
def _touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def fake_bids(tmp_path):
    """A BIDS tree with two subjects, one task, two runs (zero-padded)."""
    root = tmp_path / "ds"

    # Top-level inheritance sidecar
    _touch(root / "task-stopsignal_bold.json",
           json.dumps({"RepetitionTime": 2.0, "TaskName": "stopsignal"}))

    for sub_label in ("S01", "S02"):
        sub = f"sub-{sub_label}"
        # raw
        _touch(root / sub / "anat" / f"{sub}_T1w.nii.gz")
        for run in (1, 2):
            stem = f"{sub}_task-stopsignal_run-{run:02d}"
            _touch(root / sub / "func" / f"{stem}_bold.nii.gz")
            _touch(root / sub / "func" / f"{stem}_events.tsv",
                   "onset\tduration\ttrial_type\n0.0\t1.0\tgo\n")
        # derivatives (newer fmriprep: confounds_timeseries)
        deriv = root / "derivatives" / "fmriprep" / sub
        _touch(deriv / "anat"
               / f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
        for run in (1, 2):
            stem = f"{sub}_task-stopsignal_run-{run:02d}"
            _touch(deriv / "func"
                   / f"{stem}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
            _touch(deriv / "func"
                   / f"{stem}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
            _touch(deriv / "func" / f"{stem}_desc-confounds_timeseries.tsv",
                   "trans_x\trot_x\n0.0\t0.0\n")

    return root


@pytest.fixture
def fake_bids_legacy_confounds(tmp_path):
    """A BIDS tree using the older fmriprep confounds filename."""
    root = tmp_path / "ds_legacy"
    sub = "sub-S01"
    deriv = root / "derivatives" / "fmriprep" / sub / "func"
    _touch(deriv / f"{sub}_task-stopsignal_run-01_desc-confounds_regressors.tsv",
           "trans_x\n0.0\n")
    return root


@pytest.fixture
def fake_bids_unpadded_runs(tmp_path):
    """A BIDS tree where runs are NOT zero-padded (run-1 not run-01)."""
    root = tmp_path / "ds_unpadded"
    sub = "sub-S01"
    base = root / sub / "func"
    _touch(base / f"{sub}_task-stopsignal_run-1_bold.nii.gz")
    return root


@pytest.fixture
def fake_bids_betas(tmp_path):
    root = tmp_path / "ds_betas"
    beta = root / "derivatives" / "betas"
    _touch(beta / "S01_betas.nii.gz")
    _touch(beta / "S01_beta_audio_computation.nii.gz")
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSubjectsAndTr:
    def test_get_subjects(self, fake_bids):
        assert bids.get_subjects(fake_bids) == ["S01", "S02"]

    def test_get_subjects_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bids.get_subjects(tmp_path / "nope")

    def test_get_tr_top_level_sidecar(self, fake_bids):
        assert bids.get_tr(fake_bids, "stopsignal") == 2.0

    def test_get_tr_missing_task(self, fake_bids):
        with pytest.raises(FileNotFoundError):
            bids.get_tr(fake_bids, "nonexistent_task")


class TestSubjectNormalization:
    @pytest.mark.parametrize("sub_input", ["S01", "sub-S01"])
    def test_accepts_with_or_without_prefix(self, fake_bids, sub_input):
        path = bids.get_file(fake_bids, sub_input, "raw", "bold",
                             task="stopsignal", run=1)
        assert path.endswith("sub-S01_task-stopsignal_run-01_bold.nii.gz")


class TestRawScope:
    def test_bold(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "raw", "bold",
                             task="stopsignal", run=1)
        assert path.endswith("sub-S01/func/sub-S01_task-stopsignal_run-01_bold.nii.gz")

    def test_events(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "raw", "events",
                             task="stopsignal", run=1)
        assert path.endswith("sub-S01_task-stopsignal_run-01_events.tsv")

    def test_events_extension_is_forced_to_tsv(self, fake_bids):
        # Even if a user passes extension=".nii.gz", events should still be .tsv
        path = bids.get_file(fake_bids, "S01", "raw", "events",
                             task="stopsignal", run=1, extension=".nii.gz")
        assert path.endswith(".tsv")

    def test_t1w(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "raw", "T1w")
        assert path.endswith("sub-S01/anat/sub-S01_T1w.nii.gz")

    def test_unknown_suffix(self, fake_bids):
        with pytest.raises(ValueError, match="Unknown raw suffix"):
            bids.get_file(fake_bids, "S01", "raw", "diffusion",
                          task="stopsignal")


class TestDerivativesScope:
    def test_bold(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "derivatives", "bold",
                             task="stopsignal", run=1)
        assert path.endswith(
            "fmriprep/sub-S01/func/"
            "sub-S01_task-stopsignal_run-01_space-MNI152NLin2009cAsym"
            "_desc-preproc_bold.nii.gz"
        )

    def test_mask(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "derivatives", "mask",
                             task="stopsignal", run=1)
        assert path.endswith("_desc-brain_mask.nii.gz")

    def test_t1w(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "derivatives", "T1w")
        assert path.endswith(
            "fmriprep/sub-S01/anat/"
            "sub-S01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
        )

    def test_confounds_modern_fmriprep(self, fake_bids):
        path = bids.get_file(fake_bids, "S01", "derivatives", "confounds",
                             task="stopsignal", run=1)
        assert path.endswith("_desc-confounds_timeseries.tsv")

    def test_confounds_legacy_fmriprep(self, fake_bids_legacy_confounds):
        # Falls back to the older filename when the new one isn't present.
        path = bids.get_file(fake_bids_legacy_confounds, "S01",
                             "derivatives", "confounds",
                             task="stopsignal", run=1)
        assert path.endswith("_desc-confounds_regressors.tsv")

    def test_custom_pipeline(self, tmp_path):
        root = tmp_path / "ds"
        _touch(root / "derivatives" / "myown" / "sub-S01" / "func"
               / "sub-S01_task-foo_run-01_space-MNI152NLin2009cAsym"
                 "_desc-preproc_bold.nii.gz")
        path = bids.get_file(root, "S01", "derivatives", "bold",
                             task="foo", run=1, pipeline="myown")
        assert "/derivatives/myown/" in path


class TestRunPadding:
    def test_padded_int(self, fake_bids):
        # run-01 exists; int=1 should resolve to it
        assert bids.get_file(fake_bids, "S01", "raw", "bold",
                             task="stopsignal", run=1).endswith("run-01_bold.nii.gz")

    def test_unpadded_fallback(self, fake_bids_unpadded_runs):
        # Only run-1 (unpadded) exists; int=1 should still find it.
        path = bids.get_file(fake_bids_unpadded_runs, "S01", "raw", "bold",
                             task="stopsignal", run=1)
        assert path.endswith("run-1_bold.nii.gz")

    def test_explicit_string(self, fake_bids):
        # Strings are passed verbatim.
        path = bids.get_file(fake_bids, "S01", "raw", "bold",
                             task="stopsignal", run="01")
        assert path.endswith("run-01_bold.nii.gz")


class TestBetasScope:
    def test_all(self, fake_bids_betas):
        path = bids.get_file(fake_bids_betas, "S01", "betas", "all")
        assert path.endswith("derivatives/betas/S01_betas.nii.gz")

    def test_condition(self, fake_bids_betas):
        path = bids.get_file(fake_bids_betas, "S01", "betas",
                             "audio_computation")
        assert path.endswith("S01_beta_audio_computation.nii.gz")


class TestErrors:
    def test_unknown_scope(self, fake_bids):
        with pytest.raises(ValueError, match="Unknown scope"):
            bids.get_file(fake_bids, "S01", "weird", "bold")

    def test_missing_file_lists_candidates(self, fake_bids):
        with pytest.raises(FileNotFoundError) as excinfo:
            bids.get_file(fake_bids, "S99", "raw", "bold",
                          task="stopsignal", run=1)
        msg = str(excinfo.value)
        assert "sub-S99" in msg
        assert "Tried:" in msg
        # both padded and unpadded run candidates should appear
        assert "run-01" in msg
        assert "run-1" in msg


class TestLoaders:
    def test_load_events_returns_dataframe(self, fake_bids):
        df = bids.load_events(fake_bids, "S01", task="stopsignal", run=1)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["onset", "duration", "trial_type"]

    def test_load_confounds_returns_dataframe(self, fake_bids):
        df = bids.load_confounds(fake_bids, "S01", task="stopsignal", run=1)
        assert isinstance(df, pd.DataFrame)
        assert "trans_x" in df.columns
