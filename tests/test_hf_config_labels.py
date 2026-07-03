"""Pure filename-parsing tests for the config generator -- no network."""

from hf_configs.labels import (
    extract_beta_labels,
    extract_onset_kind,
    parse_bids_entities,
)


def test_parse_bids_entities_bold():
    p = ("derivatives/fmriprep/sub-S01/func/"
         "sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    assert parse_bids_entities(p) == {
        "subject": "S01",
        "task": "localizer",
        "space": "MNI152NLin2009cAsym",
        "desc": "preproc",
    }


def test_parse_bids_entities_with_run():
    p = "derivatives/fmriprep/sub-tb2994/func/sub-tb2994_task-story_run-2_desc-confounds_regressors.tsv"
    ent = parse_bids_entities(p)
    assert ent["subject"] == "tb2994"
    assert ent["task"] == "story"
    assert ent["run"] == "2"


def test_parse_bids_entities_sherlock_numeric_subject():
    p = "derivatives/fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-brain_mask.nii.gz"
    assert parse_bids_entities(p)["subject"] == "01"


def test_extract_beta_labels_individual():
    assert extract_beta_labels("derivatives/betas/S01_beta_audio_computation.nii.gz") == {
        "subject": "S01",
        "condition": "audio_computation",
        "type": "individual",
    }


def test_extract_beta_labels_stacked():
    assert extract_beta_labels("derivatives/betas/S07_betas.nii.gz") == {
        "subject": "S07",
        "type": "stacked",
    }


def test_extract_onset_kind():
    assert extract_onset_kind("onsets/Sherlock_Watch_Scene_N50_Onsets.csv") == {"kind": "watch"}
    assert extract_onset_kind("onsets/Sherlock_Recall_Scene_n50_Onsets.csv") == {"kind": "recall"}
    assert extract_onset_kind("onsets/Sherlock_Crop_Onsets.csv") == {"kind": "crop"}
