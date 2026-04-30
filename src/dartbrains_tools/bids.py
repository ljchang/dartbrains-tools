"""
DartBrains BIDS Dataset Access
==============================

Helpers for loading data from a local BIDS-formatted folder. Mirrors the API
of ``dartbrains_tools.data`` (which loads from HuggingFace), so the same
downstream code can switch between a local BIDS tree and the HF cache by
swapping the import:

    from dartbrains_tools import data    # HuggingFace
    from dartbrains_tools import bids    # local BIDS folder

Differences from the HF helper:

* All public functions take ``bids_root`` as the first argument.
* ``task`` and ``run`` are exposed because real BIDS datasets are commonly
  multi-task / multi-run (the HF localizer is single-task / single-run).
* Returned paths are checked for existence; ``FileNotFoundError`` is raised
  with the candidate paths that were tried, which is a much friendlier error
  than getting a missing-file failure deep inside ``nibabel``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_PIPELINE = "fmriprep"
DEFAULT_SPACE = "MNI152NLin2009cAsym"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _norm_sub(subject: str | int) -> str:
    """Return the subject label without a ``sub-`` prefix (e.g. ``"S01"``)."""
    s = str(subject)
    return s[4:] if s.startswith("sub-") else s


def _format_task(task: str | None) -> str:
    return f"_task-{task}" if task else ""


def _format_run_candidates(run: str | int | None) -> list[str]:
    """Yield BIDS run-entity strings to try, in priority order.

    Datasets vary on zero-padding (``run-1`` vs ``run-01``), so when an int
    is passed we try the zero-padded form first and fall back to unpadded.
    """
    if run is None:
        return [""]
    if isinstance(run, int):
        return [f"_run-{run:02d}", f"_run-{run}"]
    return [f"_run-{run}"]


def _resolve(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def _raise_not_found(scope: str, suffix: str, sub: str, task, run, root, candidates):
    tried = "\n  ".join(str(c) for c in candidates)
    extras = []
    if task:
        extras.append(f"task={task}")
    if run is not None:
        extras.append(f"run={run}")
    detail = " ".join(extras)
    raise FileNotFoundError(
        f"Could not find {scope}/{suffix} for {sub}"
        + (f" ({detail})" if detail else "")
        + f" under {root}.\nTried:\n  {tried}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_subjects(bids_root: str | Path) -> list[str]:
    """Return subject labels (without ``sub-`` prefix) found in ``bids_root``."""
    root = Path(bids_root)
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")
    return sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())


def get_tr(bids_root: str | Path, task: str) -> float:
    """Read ``RepetitionTime`` for ``task`` from a JSON sidecar.

    Looks first at ``<bids_root>/task-<task>_bold.json`` (the inheritance-style
    location), then falls back to the first matching subject-level sidecar.
    """
    root = Path(bids_root)
    top = root / f"task-{task}_bold.json"
    if top.exists():
        sidecar = top
    else:
        sidecar = next(root.glob(f"sub-*/func/sub-*_task-{task}_*bold.json"), None)
    if sidecar is None:
        raise FileNotFoundError(
            f"No JSON sidecar for task '{task}' under {root}. "
            f"Expected {top} or sub-*/func/*_task-{task}_*bold.json"
        )
    with open(sidecar) as f:
        meta = json.load(f)
    return float(meta["RepetitionTime"])


def get_file(
    bids_root: str | Path,
    subject: str | int,
    scope: str,
    suffix: str,
    task: str | None = None,
    run: str | int | None = None,
    *,
    pipeline: str = DEFAULT_PIPELINE,
    space: str = DEFAULT_SPACE,
    extension: str = ".nii.gz",
) -> str:
    """Locate a file in a local BIDS tree and return its absolute path.

    Args:
        bids_root: Path to the BIDS dataset root.
        subject: Subject label, with or without the ``sub-`` prefix
            (``"S01"``, ``"sub-S01"``, and ``"01"`` are all accepted).
        scope: One of ``"raw"``, ``"derivatives"``, or ``"betas"``.
        suffix: BIDS suffix -- ``"bold"``, ``"T1w"``, ``"events"``,
            ``"confounds"``, ``"mask"``, or a condition / ``"all"`` for
            ``scope="betas"``.
        task: BIDS task label (required for ``"bold"``, ``"events"``,
            ``"confounds"``, ``"mask"``).
        run: BIDS run index. ``int`` is tried with and without zero-padding;
            ``str`` is used verbatim; ``None`` means no ``run-`` entity.
        pipeline: Derivatives pipeline directory under ``derivatives/``
            (default ``"fmriprep"``).
        space: ``space-`` entity for fmriprep outputs (default
            ``"MNI152NLin2009cAsym"``).
        extension: File extension including dot (default ``".nii.gz"``).
            Ignored for suffixes whose extension is fixed (``events``,
            ``confounds``).

    Returns:
        Absolute filesystem path as a string.

    Raises:
        FileNotFoundError: with the list of candidate paths tried.
        ValueError: on an unknown ``scope`` or ``suffix``.
    """
    root = Path(bids_root)
    s = _norm_sub(subject)
    sub = f"sub-{s}"
    task_ent = _format_task(task)
    run_candidates = _format_run_candidates(run)

    candidates: list[Path] = []

    if scope == "raw":
        if suffix in ("bold", "events"):
            base = root / sub / "func"
            ext = ".tsv" if suffix == "events" else extension
            for run_ent in run_candidates:
                candidates.append(base / f"{sub}{task_ent}{run_ent}_{suffix}{ext}")
        elif suffix == "T1w":
            candidates.append(root / sub / "anat" / f"{sub}_T1w{extension}")
        else:
            raise ValueError(
                f"Unknown raw suffix: {suffix!r}. Use 'bold', 'events', or 'T1w'."
            )

    elif scope == "derivatives":
        deriv = root / "derivatives" / pipeline / sub
        space_ent = f"_space-{space}"
        if suffix == "bold":
            for run_ent in run_candidates:
                candidates.append(
                    deriv / "func"
                    / f"{sub}{task_ent}{run_ent}{space_ent}_desc-preproc_bold{extension}"
                )
        elif suffix == "mask":
            for run_ent in run_candidates:
                candidates.append(
                    deriv / "func"
                    / f"{sub}{task_ent}{run_ent}{space_ent}_desc-brain_mask{extension}"
                )
        elif suffix == "T1w":
            candidates.append(
                deriv / "anat" / f"{sub}{space_ent}_desc-preproc_T1w{extension}"
            )
        elif suffix == "confounds":
            # fmriprep renamed this file in 21.x: try the new name first.
            for run_ent in run_candidates:
                candidates.append(
                    deriv / "func"
                    / f"{sub}{task_ent}{run_ent}_desc-confounds_timeseries.tsv"
                )
                candidates.append(
                    deriv / "func"
                    / f"{sub}{task_ent}{run_ent}_desc-confounds_regressors.tsv"
                )
        else:
            raise ValueError(
                f"Unknown derivatives suffix: {suffix!r}. "
                f"Use 'bold', 'mask', 'T1w', or 'confounds'."
            )

    elif scope == "betas":
        beta_root = root / "derivatives" / "betas"
        if suffix == "all":
            candidates.append(beta_root / f"{s}_betas{extension}")
        else:
            candidates.append(beta_root / f"{s}_beta_{suffix}{extension}")

    else:
        raise ValueError(
            f"Unknown scope: {scope!r}. Use 'raw', 'derivatives', or 'betas'."
        )

    found = _resolve(candidates)
    if found is None:
        _raise_not_found(scope, suffix, sub, task, run, root, candidates)
    return str(found)


def load_events(
    bids_root: str | Path,
    subject: str | int,
    task: str,
    run: str | int | None = None,
) -> pd.DataFrame:
    """Load a raw events TSV as a DataFrame."""
    path = get_file(bids_root, subject, "raw", "events", task=task, run=run)
    return pd.read_csv(path, sep="\t")


def load_confounds(
    bids_root: str | Path,
    subject: str | int,
    task: str,
    run: str | int | None = None,
    *,
    pipeline: str = DEFAULT_PIPELINE,
) -> pd.DataFrame:
    """Load the fmriprep confounds TSV as a DataFrame."""
    path = get_file(
        bids_root, subject, "derivatives", "confounds",
        task=task, run=run, pipeline=pipeline,
    )
    return pd.read_csv(path, sep="\t")
