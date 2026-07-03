"""
DartBrains Paranoia Dataset Access
===================================

Helper functions to download and access the Paranoia naturalistic-fMRI dataset
from HuggingFace Hub (dartbrains/paranoia).

Originally distributed via DataLad at https://gin.g-node.org/ljchang/Paranoia.
Source paper: Finn et al., 2018, Nat Commun (doi:10.1038/s41467-018-04387-2).

Files are downloaded on first access and cached locally by huggingface_hub.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from ._hub import download

REPO_ID = "dartbrains/paranoia"

SUBJECTS = [
    "sub-tb2994", "sub-tb3132", "sub-tb3240", "sub-tb3279", "sub-tb3512",
    "sub-tb3592", "sub-tb3602", "sub-tb3626", "sub-tb3646", "sub-tb3744",
    "sub-tb3757", "sub-tb3784", "sub-tb3810", "sub-tb3846", "sub-tb3858",
    "sub-tb3920", "sub-tb3929", "sub-tb3964", "sub-tb3977", "sub-tb4450",
    "sub-tb4547", "sub-tb4572",
]  # fmt: skip

TR = 1.0  # seconds

RUNS = (1, 2, 3)

_SPACE = "MNI152NLin2009cAsym"
_TASK = "story"

Suffix = Literal["bold", "confounds", "mask", "boldref", "T1w"]


def _download(filename: str) -> str:
    return download(REPO_ID, filename)


def get_subjects() -> list[str]:
    """Return the list of Paranoia subject IDs (sub-tbXXXX)."""
    return list(SUBJECTS)


def get_runs() -> list[int]:
    """Return the list of run numbers (3 story runs)."""
    return list(RUNS)


def get_tr() -> float:
    """Return the repetition time in seconds."""
    return TR


def get_file(
    subject: str,
    run: int,
    suffix: Suffix,
    *,
    denoised: bool = False,
    smoothed: bool = False,
    extension: str = ".nii.gz",
) -> str:
    """Download and return the local path to a Paranoia fmriprep file.

    Args:
        subject: Subject ID, e.g. "sub-tb2994".
        run: 1, 2, or 3 (three story runs).
        suffix: BIDS suffix -- "bold", "confounds", "mask", "boldref", "T1w".
            "T1w" ignores ``run`` and returns the anatomical file.
        denoised: If True, prepend "denoise_" to the bold filename
            (the denoise+smooth variant from the original course).
        smoothed: If True, include "smooth6mm" (6 mm FWHM) in the bold filename.
        extension: File extension including dot. Defaults to ".nii.gz".

    Returns:
        Local filesystem path to the cached file.
    """
    if run not in RUNS:
        raise ValueError(f"Unknown run: {run!r}. Use one of {list(RUNS)}.")

    sub = subject
    func = f"derivatives/fmriprep/{sub}/func"
    anat = f"derivatives/fmriprep/{sub}/anat"

    if suffix == "T1w":
        filename = f"{anat}/{sub}_space-{_SPACE}_desc-preproc_T1w{extension}"
    elif suffix == "bold":
        mods = []
        if denoised:
            mods.append("denoise")
        if smoothed:
            mods.append("smooth6mm")
        mod_str = "_".join(mods) + "_" if mods else ""
        filename = (
            f"{func}/{sub}_{mod_str}task-{_TASK}_run-{run}_"
            f"space-{_SPACE}_desc-preproc_bold{extension}"
        )
    elif suffix == "confounds":
        filename = f"{func}/{sub}_task-{_TASK}_run-{run}_desc-confounds_regressors.tsv"
    elif suffix == "mask":
        filename = f"{func}/{sub}_task-{_TASK}_run-{run}_space-{_SPACE}_desc-brain_mask{extension}"
    elif suffix == "boldref":
        filename = f"{func}/{sub}_task-{_TASK}_run-{run}_space-{_SPACE}_boldref{extension}"
    else:
        raise ValueError(f"Unknown suffix: {suffix!r}.")

    return _download(filename)


def load_confounds(subject: str, run: int) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a (subject, run)."""
    path = get_file(subject, run=run, suffix="confounds")
    return pd.read_csv(path, sep="\t")


def load_participants() -> pd.DataFrame:
    """Download and load participants.tsv (age, sex, GPTSA score)."""
    path = _download("participants.tsv")
    return pd.read_csv(path, sep="\t")


def load_transcript(story: int) -> str:
    """Download and return the text content of a story transcript (1, 2, or 3)."""
    if story not in RUNS:
        raise ValueError(f"Unknown story: {story!r}. Use 1, 2, or 3.")
    path = _download(f"stimuli/paranoia_story{story}_transcript.txt")
    return Path(path).read_text()


def load_roi_timeseries(subject: str, run: int) -> pd.DataFrame:
    """Download and load the node-timeseries CSV for (subject, run)."""
    filename = f"derivatives/fmriprep/{subject}/func/{subject}_run-{run}_nodeTimeSeries.csv"
    return pd.read_csv(_download(filename))


def get_stimulus_audio(story: int) -> str:
    """Download and return the local path to a story audio file (1, 2, or 3)."""
    if story not in RUNS:
        raise ValueError(f"Unknown story: {story!r}. Use 1, 2, or 3.")
    return _download(f"stimuli/stimuli_story{story}_audio.wav")
