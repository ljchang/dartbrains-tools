# dartbrains-tools

Helper library and interactive anywidgets for the [DartBrains](https://dartbrains.org)
fMRI course. Extracted from the [book repo](https://github.com/ljchang/dartbrains)
so the widgets and helpers can be installed standalone — including in
[molab](https://molab.marimo.io) and pyodide/WASM marimo notebooks.

## Install

```bash
pip install dartbrains-tools

# Optional: include marimo for notebook_utils.youtube()
pip install "dartbrains-tools[notebook]"
```

## Modules

- `dartbrains_tools.data.localizer` — load the Pinel Localizer dataset from the Hugging Face Hub. The same API is re-exported at `dartbrains_tools.data` for back-compat.
- `dartbrains_tools.data.sherlock` — load the Sherlock naturalistic-fMRI dataset (Chen et al. 2017).
- `dartbrains_tools.data.paranoia` — load the Paranoia naturalistic-fMRI dataset (Finn et al. 2018).
- `dartbrains_tools.mr_simulations` — Bloch equation solvers, signal generators,
  HRF, and Plotly visualization helpers.
- `dartbrains_tools.mr_widgets` — 10 anywidgets for interactive MR physics teaching
  (`PrecessionWidget`, `SpinEnsembleWidget`, `KSpaceWidget`, `ConvolutionWidget`,
  `EncodingWidget`, `CompassWidget`, `NetMagnetizationWidget`, `TransformCubeWidget`,
  `CostFunctionWidget`, `SmoothingWidget`).
- `dartbrains_tools.storage` — course storage behind Dartmouth sign-in: the class copy of the
  data, each student's private space, assignment data released on a schedule, and a durable
  cache. A thin wrapper over [`marimo_grader_client.storage`](https://marimograder.org/students/course-storage/)
  that sets DartBrains' grader as the default and adds `dataset()` for the public Hugging Face
  repos. Falls back to a local directory (`GRADER_STORAGE_ROOT`) for builds and tests.
- `dartbrains_tools.notebook_utils` — marimo helpers: `youtube()`, `image()` (the book's
  figures wherever the notebook runs, including molab), `assignment_card()` (links to a
  chapter's assignment on the grader).

## Quick start

```python
from dartbrains_tools.mr_widgets import PrecessionWidget

w = PrecessionWidget(b0=3.0, flip_angle=90.0)
w  # Interactive 3D Three.js animation in any anywidget host.
```

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

```python
# Course storage (needs a Dartmouth sign-in; public datasets above need none)
from dartbrains_tools import storage

signin = storage.signin_button(); signin                       # in a marimo cell: never blocks
course = storage.course() if storage.connect(signin) else None  # else: public data

path = course.local_path("localizer/sub-S01/func/sub-S01_task-localizer_events.tsv")
storage.private().put("week3/betas.pkl", betas)                # pickle/npy/csv/json/nii.gz by extension
storage.private().get("week3/betas.pkl")
storage.assignment("midterm")                                  # NotReleased before release_at

@storage.cache                                                 # local -> shared cache -> private cache -> compute
def fit(subject): ...

# Outside marimo (scripts, the instructor uploading data):
storage.signin()                                               # device sign-in, cached afterwards
storage.course().sync("~/data/localizer", "localizer")         # instructors: course() is read-write
```

The notebook's `# /// script` block tells the library which grader, course and term it belongs to
(`[tool.grader]`, written by marimo-book's `sync-deps`); `GRADER_SERVER`, `GRADER_COURSE`,
`GRADER_TERM` and `GRADER_OFFERING_ID` override it (the 0.2.x `DARTBRAINS_*` names still work).

## Development

```bash
git clone https://github.com/ljchang/dartbrains-tools
cd dartbrains-tools
uv sync
uv run pytest
uv build
```

## License

MIT. The parent course materials at [dartbrains](https://github.com/ljchang/dartbrains)
remain CC-BY-SA-4.0; this companion library is permissive so it can be reused
in any downstream project.
