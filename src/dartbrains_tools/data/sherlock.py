"""
DartBrains Sherlock Dataset Access
===================================

Helper functions to download and access the Sherlock naturalistic-fMRI dataset
from HuggingFace Hub (dartbrains/sherlock).

Originally distributed via DataLad at https://gin.g-node.org/ljchang/Sherlock.
Source paper: Chen et al., 2017, Nat Neurosci (doi:10.1038/nn.4450).

Files are downloaded on first access and cached locally by huggingface_hub.
"""

from typing import Literal

import pandas as pd

from ._hub import download

REPO_ID = "dartbrains/sherlock"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 17)]  # sub-01 .. sub-16

TR = 1.5  # seconds

TASKS = ("sherlockPart1", "sherlockPart2", "freerecall")

_SPACE = "MNI152NLin2009cAsym"

Suffix = Literal["bold", "confounds", "mask", "boldref", "T1w"]
OnsetKind = Literal["watch", "recall", "crop"]

_ONSET_FILES = {
    "watch": "onsets/Sherlock_Watch_Scene_N50_Onsets.csv",
    "recall": "onsets/Sherlock_Recall_Scene_n50_Onsets.csv",
    "crop": "onsets/Sherlock_Crop_Onsets.csv",
}


def _download(filename: str) -> str:
    return download(REPO_ID, filename)


def get_subjects() -> list[str]:
    """Return the list of Sherlock subject IDs (sub-01 .. sub-16)."""
    return list(SUBJECTS)


def get_tasks() -> list[str]:
    """Return the list of task names."""
    return list(TASKS)


def get_tr() -> float:
    """Return the repetition time in seconds."""
    return TR


def get_file(
    subject: str,
    task: str,
    suffix: Suffix,
    *,
    denoised: bool = False,
    smoothed: bool = False,
    cropped: bool = False,
    extension: str = ".nii.gz",
) -> str:
    """Download and return the local path to a Sherlock fmriprep file.

    Args:
        subject: Subject ID, e.g. "sub-01".
        task: One of "sherlockPart1", "sherlockPart2", "freerecall".
        suffix: BIDS suffix -- "bold", "confounds", "mask", "boldref", "T1w".
            "T1w" ignores ``task`` and returns the anatomical file.
        denoised: If True, prepend "denoise_" to the bold filename
            (the denoise+smooth+crop variant from the original course).
        smoothed: If True, include "smooth6mm" (6 mm FWHM) in the bold filename.
        cropped: If True, include "crop" in the bold filename.
        extension: File extension including dot. Defaults to ".nii.gz".

    Returns:
        Local filesystem path to the cached file.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}. Use one of {TASKS}.")

    sub = subject
    func = f"derivatives/fmriprep/{sub}/func"
    anat = f"derivatives/fmriprep/{sub}/anat"

    if suffix == "T1w":
        filename = f"{anat}/{sub}_space-{_SPACE}_desc-preproc_T1w{extension}"
    elif suffix == "bold":
        mods = []
        if denoised:
            mods.append("denoise")
        if cropped:
            mods.append("crop")
        if smoothed:
            mods.append("smooth6mm")
        mod_str = "_".join(mods) + "_" if mods else ""
        filename = f"{func}/{sub}_{mod_str}task-{task}_space-{_SPACE}_desc-preproc_bold{extension}"
    elif suffix == "confounds":
        filename = f"{func}/{sub}_task-{task}_desc-confounds_regressors.tsv"
    elif suffix == "mask":
        filename = f"{func}/{sub}_task-{task}_space-{_SPACE}_desc-brain_mask{extension}"
    elif suffix == "boldref":
        filename = f"{func}/{sub}_task-{task}_space-{_SPACE}_boldref{extension}"
    else:
        raise ValueError(f"Unknown suffix: {suffix!r}.")

    return _download(filename)


def load_confounds(subject: str, task: str) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a (subject, task)."""
    path = get_file(subject, task=task, suffix="confounds")
    return pd.read_csv(path, sep="\t")


def load_onsets(kind: OnsetKind) -> pd.DataFrame:
    """Download and load one of the scene-onset CSVs.

    Args:
        kind: "watch" (50 scenes during viewing), "recall" (50 scenes during
            free recall), or "crop" (per-subject crop onsets).
    """
    if kind not in _ONSET_FILES:
        raise ValueError(f"Unknown onsets kind: {kind!r}. Use one of {list(_ONSET_FILES)}.")
    path = _download(_ONSET_FILES[kind])
    return pd.read_csv(path)


def load_roi_timeseries(subject: str, part: int) -> pd.DataFrame:
    """Download and load the 50-ROI average timeseries CSV for (subject, part)."""
    if part not in (1, 2):
        raise ValueError("part must be 1 or 2 (Sherlock has two viewing runs).")
    filename = f"derivatives/fmriprep/{subject}/func/{subject}_Part{part}_Average_ROI_n50.csv"
    return pd.read_csv(_download(filename))


def get_stimulus_video() -> str:
    """Download and return the local path to the Sherlock Part 1 video (.m4v)."""
    return _download("stimuli/stimuli_Sherlock.m4v")
