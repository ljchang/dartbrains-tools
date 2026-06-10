# Sherlock + Paranoia datasets on HuggingFace + dartbrains-tools accessors

**Status:** proposed
**Date:** 2026-05-27
**Author:** Luke Chang (with Claude Opus 4.7)

## Context

`dartbrains-tools` currently ships one dataset accessor: the Pinel Localizer
(`dartbrains/localizer` on HuggingFace Hub, surfaced as
`dartbrains_tools.data`). The DartBrains naturalistic-data course uses two
additional datasets that were previously distributed via DataLad on
`gin.g-node.org`:

- **Sherlock** (Chen et al., 2017, *Nat Neurosci*) — 16 subjects watching
  Sherlock + verbally recalling it. Source: OpenNeuro ds002345. Local copy
  218 GB.
- **Paranoia** (Finn et al., 2018, *Nat Commun*) — 22 subjects listening to a
  three-part ambiguous social narrative. Source: OpenNeuro ds001338. Local
  copy 201 GB.

This spec covers (a) migrating both datasets to the HF Hub under the
`dartbrains` org and (b) adding Python accessors to `dartbrains-tools` that
mirror the existing Localizer API.

## Goals

1. Two new public HF dataset repos: `dartbrains/sherlock`, `dartbrains/paranoia` —
   full local mirror, layout preserved verbatim.
2. New accessor modules `dartbrains_tools.data.sherlock` and
   `dartbrains_tools.data.paranoia` with the same lazy per-file download
   pattern as the existing Localizer accessor.
3. Refactor the single-file `data.py` into a `data/` subpackage so all three
   datasets live side-by-side, without breaking existing import sites.
4. Update the dartbrains book tutorial `content/Download_Data.py` to document
   the new Sherlock + Paranoia accessors. Bump the `dartbrains-tools` pin in
   `~/Github/dartbrains/pyproject.toml` to `>=0.1.4`.

## Non-goals

- Trimming or curating dataset contents. The user has explicitly chosen to
  mirror the full local copy (including fmriprep figures, logs, HTML reports,
  and the `.hdf5` mirrors of `.nii.gz` BOLD).
- A unified cross-dataset abstraction. Each dataset gets its own module with
  dataset-specific helpers (`load_onsets()`, `load_transcript()`, etc.).
- Re-running fmriprep. Data is preserved at its current version (fmriprep
  1.2.6-1).

## Design

### HuggingFace dataset repos

| Repo | Local source | Top-level layout |
|---|---|---|
| `dartbrains/sherlock` | `/Users/lukechang/Downloads/Sherlock/` | `fmriprep/`, `onsets/`, `stimuli/` |
| `dartbrains/paranoia` | `/Users/lukechang/Downloads/Paranoia/` | `fmriprep/`, `stimuli/`, `participants.tsv` |

Both public. Each gets a `README.md` (HF dataset card) with citation, license,
OpenNeuro source, TR, fmriprep version, and the original GIN URL for
provenance.

### Upload mechanism

`huggingface_hub.upload_large_folder(folder_path, repo_id, repo_type="dataset")` —
chunks, parallelizes, checkpoints to `<folder>/.cache/.huggingface/`, resumes
on re-invocation. LFS automatically handles files >10 MB (every BOLD `.nii.gz`).

Per dataset, a tiny driver script:

```python
from huggingface_hub import upload_large_folder
upload_large_folder(
    folder_path="/Users/lukechang/Downloads/Sherlock",
    repo_id="dartbrains/sherlock",
    repo_type="dataset",
    num_workers=8,
)
```

Two scripts run in parallel as background bash processes from this session.
Logs to `/tmp/sherlock_upload.log` and `/tmp/paranoia_upload.log`. Both
processes survive session end; user can `tail -f` to monitor.

### Helper subpackage structure

```
src/dartbrains_tools/data/
    __init__.py        # re-exports localizer API for back-compat
    _hub.py            # _download(repo_id, filename) shared helper
    localizer.py       # existing functions, REPO_ID = "dartbrains/localizer"
    sherlock.py        # new
    paranoia.py        # new
```

`_hub.py`:

```python
from huggingface_hub import hf_hub_download

def _download(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
```

Each per-dataset module declares its own `REPO_ID`, `SUBJECTS`, `TR`, `TASKS`
constants and a thin `_download(filename)` partial.

### Sherlock API

```python
REPO_ID = "dartbrains/sherlock"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 17)]  # sub-01 .. sub-16
TR = 1.5
TASKS = ["sherlockPart1", "sherlockPart2", "freerecall"]

def get_subjects() -> list[str]: ...
def get_tasks() -> list[str]: ...
def get_tr() -> float: ...

def get_file(
    subject: str,
    task: str,
    suffix: Literal["bold", "confounds", "mask", "boldref", "T1w"],
    *,
    denoised: bool = False,
    smoothed: bool = False,
    cropped: bool = False,
    extension: str = ".nii.gz",
) -> str:
    """Returns local cached path to a Sherlock file."""

def load_confounds(subject: str, task: str) -> pd.DataFrame: ...
def load_onsets(kind: Literal["watch", "recall", "crop"]) -> pd.DataFrame: ...
def load_roi_timeseries(subject: str, part: int) -> pd.DataFrame: ...
def get_stimulus_video() -> str: ...  # path to stimuli/stimuli_Sherlock.m4v
```

### Paranoia API

```python
REPO_ID = "dartbrains/paranoia"
SUBJECTS = [...]  # populated from local participants.tsv at module level
TR = 1.0
RUNS = [1, 2, 3]

def get_subjects() -> list[str]: ...
def get_runs() -> list[int]: ...
def get_tr() -> float: ...

def get_file(
    subject: str,
    run: int,
    suffix: Literal["bold", "confounds", "mask", "boldref", "T1w"],
    *,
    denoised: bool = False,
    smoothed: bool = False,
    extension: str = ".nii.gz",
) -> str: ...

def load_confounds(subject: str, run: int) -> pd.DataFrame: ...
def load_participants() -> pd.DataFrame: ...
def load_transcript(story: int) -> str: ...
def load_roi_timeseries(subject: str, run: int) -> pd.DataFrame: ...
def get_stimulus_audio(story: int) -> str: ...  # path to .wav
```

### Back-compat

`data/__init__.py`:

```python
from .localizer import (
    REPO_ID, SUBJECTS, TR, CONDITIONS,
    get_subjects, get_tr, get_file, load_events, load_confounds,
)
from . import localizer, sherlock, paranoia

__all__ = [
    "REPO_ID", "SUBJECTS", "TR", "CONDITIONS",
    "get_subjects", "get_tr", "get_file", "load_events", "load_confounds",
    "localizer", "sherlock", "paranoia",
]
```

Existing `from dartbrains_tools.data import get_subjects` keeps working as a
localizer call. New code is expected to use
`dartbrains_tools.data.sherlock.get_subjects()` etc.

### Testing

Pattern follows existing `tests/test_data.py` — shape tests only, no network:

- `test_data_localizer.py` — existing tests, moved
- `test_data_sherlock.py` — constants, subject list, filename construction
- `test_data_paranoia.py` — same
- `test_data_compat.py` — assert top-level `data` re-exports the localizer API

Filename construction is the failure-prone bit. Each test calls `get_file`
with `_download` monkeypatched to `lambda f: f`, asserts the returned string
matches the expected BIDS-style path.

### Versioning

dartbrains-tools 0.1.3 → 0.1.4. Patch bump per user preference: additive
changes only, existing call sites preserved by `data/__init__.py` re-exports.

## Sequencing

1. Verify HF auth, create the two empty HF dataset repos.
2. Write minimal HF dataset cards (READMEs).
3. Kick off both `upload_large_folder` calls as background bash processes.
   These run for many hours independently of session.
4. While uploads run: refactor `data.py` → `data/` subpackage on a feature
   branch in `~/Github/dartbrains-tools/`. Write tests. Update README and
   CHANGELOG. Bump version to 0.1.4.
5. After uploads complete: smoke-test one `hf_hub_download` per dataset via
   the new accessors, confirm files are reachable.
6. Open PR on `dartbrains-tools` (back-compat preserved, all tests pass).
7. Tag-driven release (`v0.1.4`) once PR merges.
8. Update `~/Github/dartbrains/content/Download_Data.py` to add Sherlock +
   Paranoia documentation sections using the new accessors. Bump the
   `dartbrains-tools` pin in `~/Github/dartbrains/pyproject.toml` to
   `>=0.1.4`. Open PR on `dartbrains`.

## Risks

- **Upload duration.** 419 GB at typical home upload bandwidth (50-200 Mbps)
  is 5-20 hours minimum. Mitigated by `upload_large_folder` resumability;
  re-running picks up from the on-disk checkpoint.
- **HF LFS quota.** Public dataset repos get generous free LFS; 419 GB across
  two repos is well within limits but worth flagging.
- **Local copy drift.** We mirror what's in `~/Downloads/`, not necessarily
  what was on GIN. If the user re-ran something locally, HF becomes the new
  authoritative source — acceptable as long as the team agrees.
- **Filename schema typos.** Long BIDS paths are easy to mistype. Shape tests
  catch generated-string mistakes; a manual one-file smoke test per dataset
  catches schema-vs-reality mistakes after upload completes.

## Open questions

None at design time. Open in implementation if encountered.
