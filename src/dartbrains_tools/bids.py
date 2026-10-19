"""
DartBrains BIDS Dataset Access
==============================

Helpers for loading data from a BIDS-formatted folder. Mirrors the API
of ``dartbrains_tools.data`` (which loads from HuggingFace), so the same
downstream code can switch between a BIDS tree and the HF cache by
swapping the import:

    from dartbrains_tools import data    # HuggingFace
    from dartbrains_tools import bids    # BIDS folder

The folder can be on disk or in course storage. Pass a storage mount as
``bids_root`` and each file is downloaded the first time it is asked for
(then reused from the local cache), so a notebook that uses two subjects
never fetches the rest:

    data = storage.assignment("<slug>")
    bids.get_subjects(data)                              # one listing, no downloads
    bids.get_file(data, "01", "derivatives", "bold", task="<task>", run=1)

Differences from the HF helper:

* All public functions take ``bids_root`` (a path or a storage mount) as the
  first argument.
* ``task`` and ``run`` are exposed because real BIDS datasets are commonly
  multi-task / multi-run (the HF localizer is single-task / single-run).
* Returned paths are checked for existence; ``FileNotFoundError`` is raised
  with the candidate paths that were tried, which is a much friendlier error
  than getting a missing-file failure deep inside ``nibabel``.
"""

from __future__ import annotations

import fnmatch
import json
import os
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


class _LocalTree:
    """A BIDS root on disk."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def __str__(self) -> str:
        return str(self.root)

    def check(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"BIDS root does not exist: {self.root}")

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def dirs(self, pattern: str) -> list[str]:
        return sorted(p.name for p in self.root.glob(pattern) if p.is_dir())

    def glob(self, pattern: str) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.glob(pattern))

    def path(self, rel: str) -> str:
        return str(self.root / rel)

    def where(self, rel: str) -> str:
        return str(self.root / rel)


class _MountTree:
    """A BIDS root in course storage (``storage.assignment(...)``, ``storage.course()``,
    ...). Lookups are HEAD requests and one listing; only :meth:`path` downloads."""

    def __init__(self, mount) -> None:
        self.mount = mount
        self._keys: list[str] | None = None

    def __str__(self) -> str:
        return self.mount.logical

    def check(self) -> None:
        pass  # the mount exists or storage would not have handed it out

    def exists(self, rel: str) -> bool:
        return self.mount.exists(rel)

    def _listing(self) -> list[str]:
        if self._keys is None:
            self._keys = self.mount.ls()
        return self._keys

    def dirs(self, pattern: str) -> list[str]:
        # Storage has no directories, only keys: a top-level "directory" is the
        # first segment of a key that has more after it.
        tops = {k.split("/", 1)[0] for k in self._listing() if "/" in k}
        return sorted(d for d in tops if fnmatch.fnmatchcase(d, pattern))

    def glob(self, pattern: str) -> list[str]:
        # Match segment by segment, as Path.glob does: fnmatch's ``*`` alone would
        # cross ``/`` and also find sub-*/func files in a nested copy.
        want = pattern.split("/")
        return sorted(
            k for k in self._listing()
            if len(parts := k.split("/")) == len(want)
            and all(fnmatch.fnmatchcase(p, w) for p, w in zip(parts, want))
        )

    def path(self, rel: str) -> str:
        return self.mount.local_path(rel)

    def where(self, rel: str) -> str:
        return f"{self.mount.logical.rstrip('/')}/{rel}"


def _tree(bids_root) -> _LocalTree | _MountTree:
    if isinstance(bids_root, (str, os.PathLike)):
        return _LocalTree(bids_root)
    if hasattr(bids_root, "local_path") and hasattr(bids_root, "ls"):
        return _MountTree(bids_root)
    raise TypeError(
        f"bids_root must be a path or a storage mount, not {type(bids_root).__name__}"
    )


def _resolve(tree, candidates: list[str]) -> str | None:
    for c in candidates:
        if tree.exists(c):
            return c
    return None


def _raise_not_found(scope: str, suffix: str, sub: str, task, run, tree, candidates):
    tried = "\n  ".join(tree.where(c) for c in candidates)
    extras = []
    if task:
        extras.append(f"task={task}")
    if run is not None:
        extras.append(f"run={run}")
    detail = " ".join(extras)
    raise FileNotFoundError(
        f"Could not find {scope}/{suffix} for {sub}"
        + (f" ({detail})" if detail else "")
        + f" under {tree}.\nTried:\n  {tried}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_subjects(bids_root) -> list[str]:
    """Return subject labels (without ``sub-`` prefix) found in ``bids_root``."""
    tree = _tree(bids_root)
    tree.check()
    return [d[4:] for d in tree.dirs("sub-*")]


def get_tr(bids_root, task: str) -> float:
    """Read ``RepetitionTime`` for ``task`` from a JSON sidecar.

    Looks first at ``<bids_root>/task-<task>_bold.json`` (the inheritance-style
    location), then falls back to the first matching subject-level sidecar.
    """
    tree = _tree(bids_root)
    top = f"task-{task}_bold.json"
    if tree.exists(top):
        sidecar = top
    else:
        sidecar = next(iter(tree.glob(f"sub-*/func/sub-*_task-{task}_*bold.json")), None)
    if sidecar is None:
        raise FileNotFoundError(
            f"No JSON sidecar for task '{task}' under {tree}. "
            f"Expected {tree.where(top)} or sub-*/func/*_task-{task}_*bold.json"
        )
    with open(tree.path(sidecar)) as f:
        meta = json.load(f)
    return float(meta["RepetitionTime"])


def get_file(
    bids_root,
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
    """Locate a file in a BIDS tree and return its absolute path.

    Args:
        bids_root: Path to the BIDS dataset root, or a storage mount whose
            root is one (``storage.assignment("<slug>")``). From a mount, the
            file is downloaded on first use and the cached copy's path is
            returned.
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
        Absolute filesystem path as a string (a local copy for a mount).

    Raises:
        FileNotFoundError: with the list of candidate paths tried.
        ValueError: on an unknown ``scope`` or ``suffix``.
    """
    tree = _tree(bids_root)
    s = _norm_sub(subject)
    sub = f"sub-{s}"
    task_ent = _format_task(task)
    run_candidates = _format_run_candidates(run)

    # Candidates are paths relative to the root, "/"-separated.
    candidates: list[str] = []

    if scope == "raw":
        if suffix in ("bold", "events"):
            base = f"{sub}/func"
            ext = ".tsv" if suffix == "events" else extension
            for run_ent in run_candidates:
                candidates.append(f"{base}/{sub}{task_ent}{run_ent}_{suffix}{ext}")
        elif suffix == "T1w":
            candidates.append(f"{sub}/anat/{sub}_T1w{extension}")
        else:
            raise ValueError(
                f"Unknown raw suffix: {suffix!r}. Use 'bold', 'events', or 'T1w'."
            )

    elif scope == "derivatives":
        deriv = f"derivatives/{pipeline}/{sub}"
        space_ent = f"_space-{space}"
        if suffix == "bold":
            for run_ent in run_candidates:
                candidates.append(
                    f"{deriv}/func/{sub}{task_ent}{run_ent}{space_ent}_desc-preproc_bold{extension}"
                )
        elif suffix == "mask":
            for run_ent in run_candidates:
                candidates.append(
                    f"{deriv}/func/{sub}{task_ent}{run_ent}{space_ent}_desc-brain_mask{extension}"
                )
        elif suffix == "T1w":
            candidates.append(f"{deriv}/anat/{sub}{space_ent}_desc-preproc_T1w{extension}")
        elif suffix == "confounds":
            # fmriprep renamed this file in 21.x: try the new name first.
            for run_ent in run_candidates:
                candidates.append(
                    f"{deriv}/func/{sub}{task_ent}{run_ent}_desc-confounds_timeseries.tsv"
                )
                candidates.append(
                    f"{deriv}/func/{sub}{task_ent}{run_ent}_desc-confounds_regressors.tsv"
                )
        else:
            raise ValueError(
                f"Unknown derivatives suffix: {suffix!r}. "
                f"Use 'bold', 'mask', 'T1w', or 'confounds'."
            )

    elif scope == "betas":
        beta_root = "derivatives/betas"
        if suffix == "all":
            candidates.append(f"{beta_root}/{s}_betas{extension}")
        else:
            candidates.append(f"{beta_root}/{s}_beta_{suffix}{extension}")

    else:
        raise ValueError(
            f"Unknown scope: {scope!r}. Use 'raw', 'derivatives', or 'betas'."
        )

    found = _resolve(tree, candidates)
    if found is None:
        _raise_not_found(scope, suffix, sub, task, run, tree, candidates)
    return tree.path(found)


def load_events(
    bids_root,
    subject: str | int,
    task: str,
    run: str | int | None = None,
) -> pd.DataFrame:
    """Load a raw events TSV as a DataFrame."""
    path = get_file(bids_root, subject, "raw", "events", task=task, run=run)
    return pd.read_csv(path, sep="\t")


def load_confounds(
    bids_root,
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
