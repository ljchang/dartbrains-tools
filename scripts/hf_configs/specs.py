"""Per-dataset config specs. The only place dataset-specific knowledge lives.

After layout normalization all three repos use ``derivatives/fmriprep/``, so
the bold/confounds/mask globs are identical everywhere.
"""

from __future__ import annotations

from .labels import extract_beta_labels, extract_onset_kind

_BOLD = "derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz"
_CONFOUNDS = "derivatives/fmriprep/**/*_desc-confounds_*.tsv"
_MASK = "derivatives/fmriprep/**/*_desc-brain_mask.nii.gz"

_FMRIPREP = {
    "bold": {"glob": _BOLD},
    "confounds": {"glob": _CONFOUNDS},
    "mask": {"glob": _MASK},
}

DATASETS: dict[str, dict] = {
    "dartbrains/localizer": {
        "repo": "dartbrains/localizer",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "events": {"glob": "sub-*/func/*_events.tsv"},
            "betas": {"glob": "derivatives/betas/*.nii.gz", "labels": extract_beta_labels},
            "participants": {"content": "participants.tsv", "content_out": "participants.csv"},
        },
    },
    "dartbrains/sherlock": {
        "repo": "dartbrains/sherlock",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "onsets": {"glob": "onsets/*.csv", "labels": extract_onset_kind},
        },
    },
    "dartbrains/paranoia": {
        "repo": "dartbrains/paranoia",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "participants": {"content": "participants.tsv", "content_out": "participants.csv"},
        },
    },
}
