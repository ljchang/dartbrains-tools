# Sherlock + Paranoia Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload Sherlock + Paranoia naturalistic-data fMRI datasets to HuggingFace under the `dartbrains` org, add `dartbrains_tools.data.sherlock` / `data.paranoia` accessor modules (refactoring the existing single-file data module into a subpackage), and update the dartbrains book's Download_Data tutorial to document the new accessors.

**Architecture:** Two new public HF dataset repos mirror the local folder layouts verbatim (`fmriprep/`, `onsets/`/`participants.tsv`, `stimuli/`). The dartbrains-tools package replaces `data.py` with a `data/` subpackage containing a shared `_hub` helper plus one module per dataset (`localizer.py`, `sherlock.py`, `paranoia.py`); `data/__init__.py` re-exports the existing Localizer API for full back-compat. Uploads run via `huggingface_hub.upload_large_folder` as background processes. The dartbrains book tutorial gets two new sections demonstrating the new accessors.

**Tech Stack:** Python 3.11+, `huggingface_hub>=1.0`, `pandas`, `pytest`, `ruff`, `uv`. Existing dartbrains-tools test pattern is shape-only (no network) with `_download` monkeypatched.

**Working directories:**
- `~/Github/dartbrains-tools/` (the helper library — feature branch + PR)
- `~/Github/dartbrains/` (the course book — separate feature branch + PR after dartbrains-tools 0.1.4 ships)

**Phases:**
1. **HF setup + background uploads** (Tasks 1-3) — runs in the foreground session; uploads continue independently for many hours.
2. **dartbrains-tools refactor + new accessors** (Tasks 4-11) — done while uploads run.
3. **Smoke test + tutorial update** (Tasks 12-15) — done after uploads complete and 0.1.4 is released.

---

## Phase 1: HuggingFace setup + background uploads

### Task 1: Create the two empty HF dataset repos

**Files:**
- No source files; uses `huggingface_hub.create_repo`

- [ ] **Step 1: Confirm HF auth**

Run: `~/Github/dartbrains-tools/.venv/bin/python -c "from huggingface_hub import whoami; w = whoami(); print(w.get('name'), [o.get('name') for o in w.get('orgs', [])])"`

Expected output includes `ljchang` and a list containing `dartbrains`.

- [ ] **Step 2: Create `dartbrains/sherlock` repo (idempotent)**

Run:
```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from huggingface_hub import create_repo
create_repo('dartbrains/sherlock', repo_type='dataset', private=False, exist_ok=True)
print('ok: dartbrains/sherlock')
"
```

Expected: `ok: dartbrains/sherlock` (or no error if it already exists).

- [ ] **Step 3: Create `dartbrains/paranoia` repo (idempotent)**

Run:
```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from huggingface_hub import create_repo
create_repo('dartbrains/paranoia', repo_type='dataset', private=False, exist_ok=True)
print('ok: dartbrains/paranoia')
"
```

Expected: `ok: dartbrains/paranoia`.

---

### Task 2: Write minimal HF dataset cards (READMEs)

**Files:**
- Create: `/tmp/sherlock_README.md`
- Create: `/tmp/paranoia_README.md`
- Uploaded via `huggingface_hub.upload_file` (not part of `upload_large_folder` because we want them in place before the bulk upload starts so the dataset page renders something useful immediately)

- [ ] **Step 1: Write Sherlock dataset card**

Write `/tmp/sherlock_README.md`:

````markdown
---
license: cc-by-nc-4.0
task_categories:
  - feature-extraction
tags:
  - fmri
  - neuroimaging
  - naturalistic
  - sherlock
  - bids
pretty_name: Sherlock fMRI Dataset
---

# Sherlock

Naturalistic fMRI dataset: 16 subjects watched ~50 minutes of *Sherlock* across
two scanning runs (Part1, Part2) and then verbally recalled the narrative in
the scanner. TR = 1.5 s.

This repo mirrors the fmriprep-preprocessed dataset originally distributed via
DataLad at <https://gin.g-node.org/ljchang/Sherlock>. fmriprep version 1.2.6-1.

## Layout

```
fmriprep/sub-XX/
  anat/  func/  figures/  log/
onsets/
  Sherlock_Crop_Onsets.csv
  Sherlock_Recall_Scene_n50_Onsets.csv
  Sherlock_Watch_Scene_N50_Onsets.csv
  Sherlock_Segments_1000_NN_2017.xlsx
stimuli/
  stimuli_Sherlock.m4v          # Part1 video (25 min)
  video_text.npy                # text features
```

## Access via dartbrains-tools

```python
from dartbrains_tools.data import sherlock

bold = sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
confounds = sherlock.load_confounds("sub-01", task="sherlockPart1")
```

See <https://dartbrains.org> for the full tutorial.

## Citation

Chen, J., Leong, Y., Honey, C. et al. *Shared memories reveal shared structure
in neural activity across individuals.* Nat Neurosci 20, 115–125 (2017).
<https://doi.org/10.1038/nn.4450>

Original raw data: OpenNeuro [ds002345](https://openneuro.org/datasets/ds002345).

## License

CC-BY-NC-4.0 (matches the upstream OpenNeuro release).
````

- [ ] **Step 2: Upload Sherlock README**

Run:
```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj='/tmp/sherlock_README.md',
    path_in_repo='README.md',
    repo_id='dartbrains/sherlock',
    repo_type='dataset',
    commit_message='Add dataset card',
)
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 3: Write Paranoia dataset card**

Write `/tmp/paranoia_README.md`:

````markdown
---
license: cc-by-nc-4.0
task_categories:
  - feature-extraction
tags:
  - fmri
  - neuroimaging
  - naturalistic
  - paranoia
  - bids
pretty_name: Paranoia fMRI Dataset
---

# Paranoia

Naturalistic fMRI dataset: 22 subjects listened to a three-part ambiguous
social narrative (~22 minutes total) designed to elicit varying levels of
paranoid interpretation. TR = 1.0 s. 3-second fixation before each run.

This repo mirrors the fmriprep-preprocessed dataset originally distributed via
DataLad at <https://gin.g-node.org/ljchang/Paranoia>. fmriprep version
1.2.6-1.

## Layout

```
fmriprep/sub-tbXXXX/
  anat/  func/  figures/
participants.tsv                # demographics + GPTSA score
stimuli/
  stimuli_story1_audio.wav      # 3 audio recordings (one per run)
  stimuli_story2_audio.wav
  stimuli_story3_audio.wav
  paranoia_story1_transcript.txt
  paranoia_story2_transcript.txt
  paranoia_story3_transcript.txt
  stimuli_story_fulltext.txt
```

## Access via dartbrains-tools

```python
from dartbrains_tools.data import paranoia

bold = paranoia.get_file("sub-tb2994", run=1, suffix="bold")
confounds = paranoia.load_confounds("sub-tb2994", run=1)
participants = paranoia.load_participants()
```

See <https://dartbrains.org> for the full tutorial.

## Citation

Finn, E.S., Corlett, P.R., Chen, G. et al. *Trait paranoia shapes inter-subject
synchrony in brain activity during an ambiguous social narrative.* Nat Commun
9, 2043 (2018). <https://doi.org/10.1038/s41467-018-04387-2>

Original raw data: OpenNeuro [ds001338](https://openneuro.org/datasets/ds001338).

## License

CC-BY-NC-4.0 (matches the upstream OpenNeuro release).
````

- [ ] **Step 4: Upload Paranoia README**

Run:
```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj='/tmp/paranoia_README.md',
    path_in_repo='README.md',
    repo_id='dartbrains/paranoia',
    repo_type='dataset',
    commit_message='Add dataset card',
)
print('ok')
"
```

Expected: `ok`.

---

### Task 3: Kick off background uploads via `upload_large_folder`

**Files:**
- Create: `/tmp/upload_sherlock.py`
- Create: `/tmp/upload_paranoia.py`
- Logs: `/tmp/sherlock_upload.log`, `/tmp/paranoia_upload.log`

- [ ] **Step 1: Write upload driver scripts**

Write `/tmp/upload_sherlock.py`:

```python
"""Resumable upload driver for the Sherlock dataset. Safe to re-run."""

from huggingface_hub import upload_large_folder

if __name__ == "__main__":
    upload_large_folder(
        folder_path="/Users/lukechang/Downloads/Sherlock",
        repo_id="dartbrains/sherlock",
        repo_type="dataset",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
        print_report=True,
    )
```

Write `/tmp/upload_paranoia.py`:

```python
"""Resumable upload driver for the Paranoia dataset. Safe to re-run."""

from huggingface_hub import upload_large_folder

if __name__ == "__main__":
    upload_large_folder(
        folder_path="/Users/lukechang/Downloads/Paranoia",
        repo_id="dartbrains/paranoia",
        repo_type="dataset",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
        print_report=True,
    )
```

- [ ] **Step 2: Start Sherlock upload in background**

Use the Bash tool's `run_in_background: true` parameter with:
```bash
~/Github/dartbrains-tools/.venv/bin/python /tmp/upload_sherlock.py > /tmp/sherlock_upload.log 2>&1
```

Note the returned shell id for later polling.

- [ ] **Step 3: Start Paranoia upload in background**

Use the Bash tool's `run_in_background: true` parameter with:
```bash
~/Github/dartbrains-tools/.venv/bin/python /tmp/upload_paranoia.py > /tmp/paranoia_upload.log 2>&1
```

Note the returned shell id.

- [ ] **Step 4: Verify both uploads started (sanity check after ~30 seconds)**

Run: `tail -20 /tmp/sherlock_upload.log /tmp/paranoia_upload.log`

Expected: each log shows `upload_large_folder` discovery output (file listing,
worker startup). No traceback. If a "401 Unauthorized" or "403 Forbidden"
appears: stop, investigate token scope; do not restart blindly.

---

## Phase 2: dartbrains-tools refactor (runs while uploads continue)

### Task 4: Create feature branch + scaffold the `data/` subpackage

**Files:**
- Create: `~/Github/dartbrains-tools/src/dartbrains_tools/data/__init__.py`
- Create: `~/Github/dartbrains-tools/src/dartbrains_tools/data/_hub.py`
- Create: `~/Github/dartbrains-tools/src/dartbrains_tools/data/localizer.py`
- Delete: `~/Github/dartbrains-tools/src/dartbrains_tools/data.py`

- [ ] **Step 1: Confirm working tree is clean and on main**

```bash
cd ~/Github/dartbrains-tools && git status --short && git rev-parse --abbrev-ref HEAD
```

Expected: empty status output, `main`.

- [ ] **Step 2: Create branch**

```bash
cd ~/Github/dartbrains-tools && git checkout -b feature/sherlock-paranoia-datasets
```

Expected: `Switched to a new branch 'feature/sherlock-paranoia-datasets'`.

- [ ] **Step 3: Write `_hub.py`**

Create `~/Github/dartbrains-tools/src/dartbrains_tools/data/_hub.py`:

```python
"""Shared HuggingFace Hub download helper used by all per-dataset modules."""

from huggingface_hub import hf_hub_download


def download(repo_id: str, filename: str) -> str:
    """Download a file from a HuggingFace dataset repo; return local cached path."""
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
```

- [ ] **Step 4: Move existing `data.py` content into `localizer.py`**

Read the current `~/Github/dartbrains-tools/src/dartbrains_tools/data.py` and
write its content (with adjustments below) to
`~/Github/dartbrains-tools/src/dartbrains_tools/data/localizer.py`:

```python
"""
DartBrains Localizer Dataset Access
====================================

Helper functions to download and access the Pinel Localizer dataset
from HuggingFace Hub (dartbrains/localizer).

Files are downloaded on first access and cached locally by huggingface_hub.
"""

import pandas as pd

from ._hub import download

REPO_ID = "dartbrains/localizer"

SUBJECTS = [f"S{i:02d}" for i in range(1, 21)]

TR = 2.4  # seconds, from task-localizer_bold.json

CONDITIONS = [
    "audio_computation",
    "audio_left_hand",
    "audio_right_hand",
    "audio_sentence",
    "horizontal_checkerboard",
    "vertical_checkerboard",
    "video_computation",
    "video_left_hand",
    "video_right_hand",
    "video_sentence",
]


def _download(filename: str) -> str:
    """Download a file from the dartbrains/localizer dataset. Returns local cached path."""
    return download(REPO_ID, filename)


def get_subjects() -> list[str]:
    """Return list of subject IDs (S01-S20)."""
    return list(SUBJECTS)


def get_tr() -> float:
    """Return the repetition time in seconds."""
    return TR


def get_file(subject: str, scope: str, suffix: str, extension: str = ".nii.gz") -> str:
    """Download and return the local path to a dataset file.

    Args:
        subject: Subject ID, e.g. "S01"
        scope: One of "raw", "derivatives", or "betas"
        suffix: BIDS suffix -- "bold", "T1w", "events", "confounds", "mask",
                or a condition name for betas (e.g. "audio_computation"),
                or "all" for the stacked betas file
        extension: File extension including dot, e.g. ".nii.gz", ".tsv"

    Returns:
        Local filesystem path to the cached file.
    """
    s = subject
    sub = f"sub-{s}"

    if scope == "betas":
        if suffix == "all":
            filename = f"derivatives/betas/{s}_betas{extension}"
        else:
            filename = f"derivatives/betas/{s}_beta_{suffix}{extension}"

    elif scope == "raw":
        if suffix == "events":
            filename = f"{sub}/func/{sub}_task-localizer_events.tsv"
        elif suffix == "bold":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        else:
            raise ValueError(f"Unknown raw suffix: {suffix}")

    elif scope == "derivatives":
        if suffix == "bold":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        elif suffix == "T1w":
            filename = f"derivatives/fmriprep/{sub}/anat/{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w{extension}"
        elif suffix == "confounds":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_desc-confounds_regressors.tsv"
        elif suffix == "mask":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask{extension}"
        else:
            raise ValueError(f"Unknown derivatives suffix: {suffix}")
    else:
        raise ValueError(f"Unknown scope: {scope}. Use 'raw', 'derivatives', or 'betas'.")

    return _download(filename)


def load_events(subject: str) -> pd.DataFrame:
    """Download and load the events TSV for a subject as a DataFrame."""
    path = get_file(subject, scope="raw", suffix="events", extension=".tsv")
    return pd.read_csv(path, sep="\t")


def load_confounds(subject: str) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a subject."""
    path = get_file(subject, scope="derivatives", suffix="confounds")
    return pd.read_csv(path, sep="\t")
```

- [ ] **Step 5: Write `data/__init__.py` with back-compat re-exports**

Create `~/Github/dartbrains-tools/src/dartbrains_tools/data/__init__.py`:

```python
"""DartBrains dataset accessors.

This subpackage exposes one module per dataset (``localizer``, ``sherlock``,
``paranoia``). For back-compat, the Localizer API is also re-exported at the
``dartbrains_tools.data`` namespace level, so existing code that does
``from dartbrains_tools.data import get_subjects`` keeps working.

New code is encouraged to use the per-dataset namespace explicitly::

    from dartbrains_tools.data import sherlock
    bold = sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
"""

from . import localizer, paranoia, sherlock
from .localizer import (
    CONDITIONS,
    REPO_ID,
    SUBJECTS,
    TR,
    _download,
    get_file,
    get_subjects,
    get_tr,
    load_confounds,
    load_events,
)

__all__ = [
    "CONDITIONS",
    "REPO_ID",
    "SUBJECTS",
    "TR",
    "_download",
    "get_file",
    "get_subjects",
    "get_tr",
    "load_confounds",
    "load_events",
    "localizer",
    "paranoia",
    "sherlock",
]
```

- [ ] **Step 6: Add empty placeholder `sherlock.py` and `paranoia.py`**

These are filled in by later tasks but `data/__init__.py` imports them, so
they must exist for the package to import cleanly.

Create `~/Github/dartbrains-tools/src/dartbrains_tools/data/sherlock.py`:

```python
"""Sherlock dataset accessor (filled in by Task 6)."""

REPO_ID = "dartbrains/sherlock"
```

Create `~/Github/dartbrains-tools/src/dartbrains_tools/data/paranoia.py`:

```python
"""Paranoia dataset accessor (filled in by Task 7)."""

REPO_ID = "dartbrains/paranoia"
```

- [ ] **Step 7: Delete the old single-file `data.py`**

```bash
rm ~/Github/dartbrains-tools/src/dartbrains_tools/data.py
```

- [ ] **Step 8: Verify the package still imports**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -c "
from dartbrains_tools import data
from dartbrains_tools.data import sherlock, paranoia, localizer, get_subjects
print('subjects:', get_subjects()[:3])
print('sherlock REPO_ID:', sherlock.REPO_ID)
print('paranoia REPO_ID:', paranoia.REPO_ID)
print('localizer REPO_ID:', localizer.REPO_ID)
"
```

Expected: prints three subjects, three repo ids without error.

- [ ] **Step 9: Commit**

```bash
cd ~/Github/dartbrains-tools && \
  git add src/dartbrains_tools/data/ && \
  git rm src/dartbrains_tools/data.py && \
  git commit -m "refactor: split dartbrains_tools.data into per-dataset subpackage

Moves the existing single-file localizer accessor into
data/localizer.py and adds empty data/sherlock.py and
data/paranoia.py placeholders for the new dataset modules.
data/__init__.py re-exports the localizer API for back-compat,
so existing import sites (from dartbrains_tools.data import
get_subjects, ...) keep working.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Move + update existing tests, add back-compat test

**Files:**
- Modify: `~/Github/dartbrains-tools/tests/test_data.py`
- Create: `~/Github/dartbrains-tools/tests/test_data_compat.py`

- [ ] **Step 1: Update `test_data.py` to assert localizer behavior through the new namespace**

Replace `~/Github/dartbrains-tools/tests/test_data.py` entirely with:

```python
"""Shape tests for the localizer dataset module -- does not hit the network."""

from dartbrains_tools.data import localizer


def test_constants():
    assert localizer.REPO_ID == "dartbrains/localizer"
    assert localizer.TR > 0
    assert isinstance(localizer.CONDITIONS, (list, tuple))
    assert len(localizer.CONDITIONS) > 0


def test_get_subjects_returns_list():
    subjects = localizer.get_subjects()
    assert isinstance(subjects, list)
    assert all(isinstance(s, str) for s in subjects)
    assert len(subjects) > 0


def test_get_tr_returns_number():
    assert localizer.get_tr() == localizer.TR


def test_get_file_path_construction(monkeypatch):
    """get_file should construct the right filename without hitting the network."""
    captured = {}

    def fake_download(filename):
        captured["filename"] = filename
        return f"/cache/{filename}"

    monkeypatch.setattr(localizer, "_download", fake_download)

    path = localizer.get_file("S01", scope="derivatives", suffix="bold")
    assert "sub-S01_task-localizer" in captured["filename"]
    assert path.endswith(captured["filename"])
```

- [ ] **Step 2: Create `test_data_compat.py`**

Create `~/Github/dartbrains-tools/tests/test_data_compat.py`:

```python
"""Back-compat: the localizer API stays importable at dartbrains_tools.data.*."""

from dartbrains_tools import data


def test_localizer_reexports():
    assert data.REPO_ID == "dartbrains/localizer"
    assert callable(data.get_subjects)
    assert callable(data.get_tr)
    assert callable(data.get_file)
    assert callable(data.load_events)
    assert callable(data.load_confounds)


def test_subpackage_modules_importable():
    from dartbrains_tools.data import localizer, paranoia, sherlock

    assert localizer.REPO_ID == "dartbrains/localizer"
    assert sherlock.REPO_ID == "dartbrains/sherlock"
    assert paranoia.REPO_ID == "dartbrains/paranoia"
```

- [ ] **Step 3: Run tests**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest tests/test_data.py tests/test_data_compat.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd ~/Github/dartbrains-tools && \
  git add tests/test_data.py tests/test_data_compat.py && \
  git commit -m "test: update data tests for new subpackage, add back-compat test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Implement `sherlock.py`

**Files:**
- Modify: `~/Github/dartbrains-tools/src/dartbrains_tools/data/sherlock.py`
- Create: `~/Github/dartbrains-tools/tests/test_data_sherlock.py`

- [ ] **Step 1: Write failing tests for filename construction**

Create `~/Github/dartbrains-tools/tests/test_data_sherlock.py`:

```python
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
        "fmriprep/sub-01/func/"
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
        "fmriprep/sub-01/func/"
        "sub-01_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_confounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="confounds")
    assert captured["f"] == (
        "fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-confounds_regressors.tsv"
    )


def test_get_file_mask(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="mask")
    assert captured["f"] == (
        "fmriprep/sub-01/func/"
        "sub-01_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )


def test_get_file_t1w_ignores_task(monkeypatch):
    captured = {}
    monkeypatch.setattr(sherlock, "_download", lambda f: captured.setdefault("f", f))
    sherlock.get_file("sub-01", task="sherlockPart1", suffix="T1w")
    assert captured["f"] == (
        "fmriprep/sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest tests/test_data_sherlock.py -v
```

Expected: most tests FAIL (TR, TASKS, SUBJECTS, get_file, load_onsets,
get_stimulus_video don't exist yet).

- [ ] **Step 3: Implement `sherlock.py`**

Replace `~/Github/dartbrains-tools/src/dartbrains_tools/data/sherlock.py` entirely with:

```python
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
    func = f"fmriprep/{sub}/func"
    anat = f"fmriprep/{sub}/anat"

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
        filename = (
            f"{func}/{sub}_{mod_str}task-{task}_space-{_SPACE}_desc-preproc_bold{extension}"
        )
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
    filename = f"fmriprep/{subject}/func/{subject}_Part{part}_Average_ROI_n50.csv"
    return pd.read_csv(_download(filename))


def get_stimulus_video() -> str:
    """Download and return the local path to the Sherlock Part 1 video (.m4v)."""
    return _download("stimuli/stimuli_Sherlock.m4v")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest tests/test_data_sherlock.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Github/dartbrains-tools && \
  git add src/dartbrains_tools/data/sherlock.py tests/test_data_sherlock.py && \
  git commit -m "feat: add dartbrains_tools.data.sherlock accessor

Provides get_file, get_subjects, get_tasks, get_tr, load_confounds,
load_onsets, load_roi_timeseries, get_stimulus_video for the
dartbrains/sherlock HuggingFace dataset (16 subjects, naturalistic
fMRI from Chen et al. 2017).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Implement `paranoia.py`

**Files:**
- Modify: `~/Github/dartbrains-tools/src/dartbrains_tools/data/paranoia.py`
- Create: `~/Github/dartbrains-tools/tests/test_data_paranoia.py`

- [ ] **Step 1: Capture the Paranoia subject list from the local copy**

Run this and copy the output into the test + module (instead of hardcoding
guesses):
```bash
ls -d /Users/lukechang/Downloads/Paranoia/fmriprep/sub-* | xargs -n1 basename | sort > /tmp/paranoia_subjects.txt
wc -l /tmp/paranoia_subjects.txt
head /tmp/paranoia_subjects.txt
```

Expected: 22 lines, each like `sub-tb2994`.

- [ ] **Step 2: Write failing tests**

Create `~/Github/dartbrains-tools/tests/test_data_paranoia.py`:

```python
"""Shape tests for the paranoia dataset module -- does not hit the network."""

import pytest

from dartbrains_tools.data import paranoia


def test_constants():
    assert paranoia.REPO_ID == "dartbrains/paranoia"
    assert paranoia.TR == 1.0
    assert paranoia.RUNS == (1, 2, 3)
    assert len(paranoia.SUBJECTS) == 22
    assert all(s.startswith("sub-tb") for s in paranoia.SUBJECTS)


def test_get_subjects():
    s = paranoia.get_subjects()
    assert isinstance(s, list)
    assert len(s) == 22


def test_get_runs():
    assert paranoia.get_runs() == [1, 2, 3]


def test_get_file_bold(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=1, suffix="bold")
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/"
        "sub-tb2994_task-story_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_bold_denoised_smoothed(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=2, suffix="bold", denoised=True, smoothed=True)
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/"
        "sub-tb2994_denoise_smooth6mm_task-story_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def test_get_file_confounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=3, suffix="confounds")
    assert captured["f"] == (
        "fmriprep/sub-tb2994/func/sub-tb2994_task-story_run-3_desc-confounds_regressors.tsv"
    )


def test_get_file_unknown_run_raises():
    with pytest.raises(ValueError, match="run"):
        paranoia.get_file("sub-tb2994", run=4, suffix="bold")


def test_get_file_unknown_suffix_raises():
    with pytest.raises(ValueError, match="Unknown suffix"):
        paranoia.get_file("sub-tb2994", run=1, suffix="bogus")


def test_load_participants(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f) or "/dev/null")
    monkeypatch.setattr(
        "dartbrains_tools.data.paranoia.pd.read_csv",
        lambda *a, **kw: "sentinel",
    )
    paranoia.load_participants()
    assert captured["f"] == "participants.tsv"


def test_load_transcript(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f) or "/dev/null")
    # Stub Path.read_text via the module-level helper
    monkeypatch.setattr(
        "dartbrains_tools.data.paranoia.Path",
        lambda p: type("P", (), {"read_text": lambda self: "sentinel"})(),
    )
    out = paranoia.load_transcript(2)
    assert captured["f"] == "stimuli/paranoia_story2_transcript.txt"
    assert out == "sentinel"


def test_get_stimulus_audio(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_stimulus_audio(1)
    assert captured["f"] == "stimuli/stimuli_story1_audio.wav"
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest tests/test_data_paranoia.py -v
```

Expected: most tests FAIL (TR, RUNS, get_file etc don't exist yet).

- [ ] **Step 4: Implement `paranoia.py`**

Replace `~/Github/dartbrains-tools/src/dartbrains_tools/data/paranoia.py` entirely with:

```python
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
]

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
    func = f"fmriprep/{sub}/func"
    anat = f"fmriprep/{sub}/anat"

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
        filename = (
            f"{func}/{sub}_task-{_TASK}_run-{run}_space-{_SPACE}_desc-brain_mask{extension}"
        )
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
    filename = f"fmriprep/{subject}/func/{subject}_run-{run}_nodeTimeSeries.csv"
    return pd.read_csv(_download(filename))


def get_stimulus_audio(story: int) -> str:
    """Download and return the local path to a story audio file (1, 2, or 3)."""
    if story not in RUNS:
        raise ValueError(f"Unknown story: {story!r}. Use 1, 2, or 3.")
    return _download(f"stimuli/stimuli_story{story}_audio.wav")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest tests/test_data_paranoia.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/Github/dartbrains-tools && \
  git add src/dartbrains_tools/data/paranoia.py tests/test_data_paranoia.py && \
  git commit -m "feat: add dartbrains_tools.data.paranoia accessor

Provides get_file, get_subjects, get_runs, get_tr, load_confounds,
load_participants, load_transcript, load_roi_timeseries,
get_stimulus_audio for the dartbrains/paranoia HuggingFace dataset
(22 subjects, naturalistic auditory fMRI from Finn et al. 2018).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update README

**Files:**
- Modify: `~/Github/dartbrains-tools/README.md`

- [ ] **Step 1: Update the Modules section and Quick start**

Find and replace the existing `dartbrains_tools.data` bullet:

Old (line ~17):
```
- `dartbrains_tools.data` — load the Pinel Localizer dataset from the Hugging Face Hub.
```

New:
```
- `dartbrains_tools.data.localizer` — load the Pinel Localizer dataset from the Hugging Face Hub. The same API is re-exported at `dartbrains_tools.data` for back-compat.
- `dartbrains_tools.data.sherlock` — load the Sherlock naturalistic-fMRI dataset (Chen et al. 2017).
- `dartbrains_tools.data.paranoia` — load the Paranoia naturalistic-fMRI dataset (Finn et al. 2018).
```

Find the existing Quick start data block:
```python
from dartbrains_tools.data import get_subjects, get_file, load_events

subjects = get_subjects()
bold = get_file("S01", "bold")
events = load_events("S01")
```

Replace with:
```python
# Localizer (default; back-compat — also works as dartbrains_tools.data.localizer)
from dartbrains_tools.data import get_subjects, get_file, load_events

subjects = get_subjects()
bold = get_file("S01", scope="derivatives", suffix="bold")
events = load_events("S01")

# Sherlock
from dartbrains_tools.data import sherlock

bold = sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
onsets = sherlock.load_onsets("watch")

# Paranoia
from dartbrains_tools.data import paranoia

bold = paranoia.get_file("sub-tb2994", run=1, suffix="bold")
participants = paranoia.load_participants()
```

- [ ] **Step 2: Commit**

```bash
cd ~/Github/dartbrains-tools && \
  git add README.md && \
  git commit -m "docs: document new sherlock and paranoia accessors in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Bump version + lint + final test run

**Files:**
- Modify: `~/Github/dartbrains-tools/pyproject.toml`
- Modify: `~/Github/dartbrains-tools/src/dartbrains_tools/__init__.py`

- [ ] **Step 1: Bump version in `pyproject.toml`**

Find `version = "0.1.3"` and replace with `version = "0.1.4"`.

- [ ] **Step 2: Bump `__version__` in `__init__.py`**

In `~/Github/dartbrains-tools/src/dartbrains_tools/__init__.py`, find:
```python
__version__ = "0.1.3"
```
Replace with:
```python
__version__ = "0.1.4"
```

- [ ] **Step 3: Run full test suite**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Run ruff**

```bash
cd ~/Github/dartbrains-tools && ./.venv/bin/python -m ruff check src/ tests/ && ./.venv/bin/python -m ruff format --check src/ tests/
```

Expected: no errors. If formatting check fails, run `./.venv/bin/python -m ruff format src/ tests/` then re-check.

- [ ] **Step 5: Commit version bump**

```bash
cd ~/Github/dartbrains-tools && \
  git add pyproject.toml src/dartbrains_tools/__init__.py && \
  git commit -m "chore: bump version to 0.1.4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Push branch + open PR on dartbrains-tools

**Files:**
- None (git + gh CLI)

- [ ] **Step 1: Push branch**

```bash
cd ~/Github/dartbrains-tools && git push -u origin feature/sherlock-paranoia-datasets
```

Expected: pushed without error.

- [ ] **Step 2: Open PR**

```bash
cd ~/Github/dartbrains-tools && gh pr create --title "Add Sherlock + Paranoia dataset accessors (0.1.4)" --body "$(cat <<'EOF'
## Summary

- Refactor `dartbrains_tools.data` from a single file into a subpackage with one module per dataset (`localizer`, `sherlock`, `paranoia`) plus a shared `_hub` helper. The existing Localizer API stays importable at `dartbrains_tools.data.*` for full back-compat.
- Add `dartbrains_tools.data.sherlock` — accessor for the new `dartbrains/sherlock` HuggingFace dataset (16 subjects, naturalistic fMRI, Chen et al. 2017).
- Add `dartbrains_tools.data.paranoia` — accessor for the new `dartbrains/paranoia` HuggingFace dataset (22 subjects, naturalistic auditory fMRI, Finn et al. 2018).
- Tests are shape-only (no network) and assert the constructed BIDS filenames match the layout uploaded to HF.
- Patch bump to 0.1.4 — additive changes only.

## Test plan

- [ ] `pytest -v` — full suite passes on Python 3.11/3.12/3.13
- [ ] `ruff check src/ tests/` clean
- [ ] `ruff format --check src/ tests/` clean
- [ ] Manual smoke test: `from dartbrains_tools.data import sherlock; sherlock.get_file('sub-01', task='sherlockPart1', suffix='confounds')` — downloads + caches a real file from `dartbrains/sherlock`
- [ ] Manual smoke test: `from dartbrains_tools.data import paranoia; paranoia.load_participants()`
- [ ] Existing localizer code (`from dartbrains_tools.data import get_subjects, get_file, load_events`) still works unchanged

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Phase 3: smoke test + tutorial update (after uploads complete and 0.1.4 ships)

### Task 11: Verify uploads completed and smoke-test the new accessors

**Files:**
- None (live HF + local cache)

- [ ] **Step 1: Confirm uploads finished**

Run: `tail -30 /tmp/sherlock_upload.log /tmp/paranoia_upload.log | grep -i -E "done|complete|error|fail"`

Expected: each log ends with a "done"/"completed" indicator from
`upload_large_folder`, no tracebacks. If anything looks wrong, re-run the
relevant `/tmp/upload_*.py` (resumes from checkpoint).

- [ ] **Step 2: Cross-check file count on HF matches local**

```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
for repo in ['dartbrains/sherlock', 'dartbrains/paranoia']:
    files = api.list_repo_files(repo, repo_type='dataset')
    print(f'{repo}: {len(files)} files')
"
```

Compare with `find /Users/lukechang/Downloads/Sherlock -type f | wc -l` and same
for Paranoia (the HF count should be `local_count - 1` for the missing
`.DS_Store` files we ignored).

- [ ] **Step 3: Smoke-test the Sherlock accessor**

```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from dartbrains_tools.data import sherlock
p = sherlock.get_file('sub-01', task='sherlockPart1', suffix='confounds')
print('confounds path:', p)
df = sherlock.load_confounds('sub-01', task='sherlockPart1')
print('rows:', len(df), 'cols:', len(df.columns))
"
```

Expected: prints a path under `~/.cache/huggingface/`, then prints row/col
counts (hundreds of TRs, dozens of confound columns).

- [ ] **Step 4: Smoke-test the Paranoia accessor**

```bash
~/Github/dartbrains-tools/.venv/bin/python -c "
from dartbrains_tools.data import paranoia
df = paranoia.load_participants()
print('participants:', len(df), 'columns:', list(df.columns))
"
```

Expected: prints 22 participants with columns `participant_id`, `age`, `sex`,
`gptsa_score`.

---

### Task 12: Extend dartbrains book `Download_Data.py` tutorial

**Files:**
- Modify: `~/Github/dartbrains/content/Download_Data.py`
- Modify: `~/Github/dartbrains/pyproject.toml`

- [ ] **Step 1: Create branch on dartbrains book**

```bash
cd ~/Github/dartbrains && git status --short && git rev-parse --abbrev-ref HEAD
```

Expected: clean tree, `main`.

```bash
cd ~/Github/dartbrains && git checkout -b feature/sherlock-paranoia-tutorial
```

- [ ] **Step 2: Bump dartbrains-tools pin**

In `~/Github/dartbrains/pyproject.toml`, find `"dartbrains-tools>=0.1.2"` and
replace with `"dartbrains-tools>=0.1.4"`.

- [ ] **Step 3: Add Sherlock section to `Download_Data.py`**

Open `~/Github/dartbrains/content/Download_Data.py`. After the existing
Localizer section (locate the last localizer cell — likely a confound-loading
or summary cell), append two new sections.

Add a new markdown cell:
```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sherlock Dataset

    The [Sherlock dataset](https://huggingface.co/datasets/dartbrains/sherlock)
    (Chen et al., 2017) contains 16 subjects who watched ~50 minutes of
    *Sherlock* across two scanning runs and then verbally recalled the
    narrative in the scanner. TR = 1.5 s. We use this dataset in the
    naturalistic-data tutorials (intersubject correlation, event
    segmentation, functional alignment).

    Access via `dartbrains_tools.data.sherlock`:
    """)
    return
```

Add a code cell:
```python
@app.cell
def _():
    from dartbrains_tools.data import sherlock

    print(f"Subjects: {sherlock.get_subjects()[:3]} ... ({len(sherlock.get_subjects())} total)")
    print(f"Tasks: {sherlock.get_tasks()}")
    print(f"TR: {sherlock.get_tr()} s")

    # Per-subject preprocessed bold (lazy download, ~800 MB each)
    bold_path = sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
    print(f"\nBOLD path: {bold_path}")

    # Confounds + scene onsets are small
    confounds = sherlock.load_confounds("sub-01", task="sherlockPart1")
    watch_onsets = sherlock.load_onsets("watch")
    print(f"\nConfounds: {confounds.shape}; Watch onsets: {watch_onsets.shape}")
    return sherlock,
```

Add a markdown cell:
```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paranoia Dataset

    The [Paranoia dataset](https://huggingface.co/datasets/dartbrains/paranoia)
    (Finn et al., 2018) contains 22 subjects who listened to a three-part
    ambiguous social narrative (~22 minutes total). TR = 1.0 s. Each subject
    completed 3 story runs.

    Access via `dartbrains_tools.data.paranoia`:
    """)
    return
```

Add a code cell:
```python
@app.cell
def _():
    from dartbrains_tools.data import paranoia

    print(f"Subjects: {paranoia.get_subjects()[:3]} ... ({len(paranoia.get_subjects())} total)")
    print(f"Runs: {paranoia.get_runs()}")
    print(f"TR: {paranoia.get_tr()} s")

    # Demographics + trait paranoia score
    participants = paranoia.load_participants()
    print(f"\nParticipants:\n{participants.head()}")

    # Story 1 transcript (small)
    transcript = paranoia.load_transcript(1)
    print(f"\nStory 1 transcript (first 200 chars):\n{transcript[:200]}")
    return paranoia,
```

- [ ] **Step 4: Verify the notebook can be opened in marimo**

```bash
cd ~/Github/dartbrains && uv run marimo edit content/Download_Data.py --headless &
sleep 5
kill %1 2>/dev/null
```

Expected: marimo starts without import errors. (We're not actually running
the cells — the BOLD download is 800 MB. The book CI will exercise these.)

- [ ] **Step 5: Build the book to confirm no rendering breakage**

```bash
cd ~/Github/dartbrains && uv run marimo-book build -b book.yml --strict 2>&1 | tail -30
```

Expected: build completes; the new Sherlock + Paranoia sections appear in the
output HTML for the Download_Data page.

- [ ] **Step 6: Commit**

```bash
cd ~/Github/dartbrains && \
  git add content/Download_Data.py pyproject.toml uv.lock && \
  git commit -m "docs: add Sherlock + Paranoia download tutorial sections

Documents the new dartbrains_tools.data.sherlock and
dartbrains_tools.data.paranoia accessors. Bumps dartbrains-tools
pin to >=0.1.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(If `uv.lock` was not modified, omit it from the `git add`.)

---

### Task 13: Push + open PR on dartbrains book

**Files:**
- None (git + gh CLI)

- [ ] **Step 1: Push branch**

```bash
cd ~/Github/dartbrains && git push -u origin feature/sherlock-paranoia-tutorial
```

- [ ] **Step 2: Open PR**

```bash
cd ~/Github/dartbrains && gh pr create --title "Add Sherlock + Paranoia download tutorial sections" --body "$(cat <<'EOF'
## Summary

- Extends `content/Download_Data.py` with two new sections documenting the Sherlock and Paranoia naturalistic-fMRI datasets, now hosted on HuggingFace at `dartbrains/sherlock` and `dartbrains/paranoia`.
- Each section demonstrates the new accessors in `dartbrains_tools.data.{sherlock,paranoia}` (added in dartbrains-tools 0.1.4).
- Bumps the `dartbrains-tools` pin from `>=0.1.2` to `>=0.1.4`.

## Test plan

- [ ] CI builds the book without errors (`marimo-book build -b book.yml --strict`)
- [ ] New Sherlock + Paranoia sections render on the Download_Data page
- [ ] Existing Localizer section unaffected (uses back-compat re-exports)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-review checklist

- ✅ Every task has explicit files + paths
- ✅ Every code step has runnable code, not "implement X"
- ✅ Test-first pattern (Tasks 6, 7) shows failing test → impl → passing test
- ✅ Tests are shape-only (no network) matching existing pattern
- ✅ Back-compat preserved: existing `from dartbrains_tools.data import get_subjects` still works
- ✅ Patch version 0.1.4 per user preference
- ✅ Upload phase is sequenced before tutorial phase (tutorial smoke-tests need the data to be live)
- ✅ Phase 2 work runs in parallel with the long upload (separate tasks, no blocking dependency)
- ✅ Each commit message is one focused unit of work
