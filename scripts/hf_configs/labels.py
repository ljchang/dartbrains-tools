"""Extract label columns from HF dataset file paths.

The shared BIDS parser handles the fmriprep-style names; two small custom
extractors handle the non-BIDS beta and onset filenames.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

# BIDS `key-value` tokens we care about, mapped to output column names.
_BIDS_KEYS = {"sub": "subject", "task": "task", "run": "run",
              "space": "space", "desc": "desc"}


def parse_bids_entities(path: str) -> dict[str, str]:
    """Return BIDS entity values parsed from the basename of *path*.

    Only the keys in ``_BIDS_KEYS`` are returned, and only when present.
    ``subject`` is the raw value after ``sub-`` (e.g. ``S01``, ``01``, ``tb2994``).
    """
    stem = PurePosixPath(path).name
    out: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, _, value = token.partition("-")
        col = _BIDS_KEYS.get(key)
        if col is not None:
            out[col] = value
    return out


_BETA_INDIVIDUAL = re.compile(r"(?P<subject>S\d+)_beta_(?P<condition>.+)\.nii\.gz$")
_BETA_STACKED = re.compile(r"(?P<subject>S\d+)_betas\.nii\.gz$")


def extract_beta_labels(path: str) -> dict[str, str]:
    """Labels for localizer beta maps (filenames are not BIDS-encoded)."""
    name = PurePosixPath(path).name
    m = _BETA_INDIVIDUAL.match(name)
    if m:
        return {"subject": m["subject"], "condition": m["condition"], "type": "individual"}
    m = _BETA_STACKED.match(name)
    if m:
        return {"subject": m["subject"], "type": "stacked"}
    raise ValueError(f"Unrecognized beta filename: {name!r}")


_ONSET_KINDS = {"watch": "watch", "recall": "recall", "crop": "crop"}


def extract_onset_kind(path: str) -> dict[str, str]:
    """Label for sherlock onset CSVs -- matched by keyword in the filename."""
    name = PurePosixPath(path).name.lower()
    for needle, kind in _ONSET_KINDS.items():
        if needle in name:
            return {"kind": kind}
    raise ValueError(f"Unrecognized onset filename: {name!r}")
