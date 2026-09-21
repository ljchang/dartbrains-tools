"""DartBrains storage: the course's copy of the data, each student's own
space, and public datasets -- one API wherever the notebook runs.

This is :mod:`marimo_grader_client.storage` with DartBrains' defaults set
(the grader's URL) and the public datasets added::

    from dartbrains_tools import storage

    signin = storage.signin_button(); signin          # in a marimo cell
    course = storage.course() if storage.connect(signin) else None

    storage.private().put("week3/betas.pkl", betas)
    storage.dataset("localizer").local_path("derivatives/betas/S01_betas.nii.gz")

Everything else -- ``group()``, ``assignment(slug)``, ``@cache``,
``local_path()``, ``GRADER_STORAGE_ROOT`` for a local backend -- is documented
on the client: https://marimograder.org/students/course-storage/.
"""

from __future__ import annotations

import os

from marimo_grader_client import storage as _storage
from marimo_grader_client.storage import *  # noqa: F403 - re-export the whole API
from marimo_grader_client.storage import __all__ as _client_all
from marimo_grader_client.storage import configure

DEFAULT_SERVER = "https://grader.dartbrains.org"

# Names the first releases used (0.2.x); honoured so a student's .env keeps working.
_LEGACY_ENV = {
    "DARTBRAINS_GRADER_TOKEN": "GRADER_TOKEN",
    "DARTBRAINS_GRADER_SERVER": "GRADER_SERVER",
    "DARTBRAINS_OFFERING": "GRADER_OFFERING_ID",
    "DARTBRAINS_COURSE": "GRADER_COURSE",
    "DARTBRAINS_TERM": "GRADER_TERM",
    "DARTBRAINS_RUNTIME": "GRADER_RUNTIME",
    "DARTBRAINS_STORAGE_ROOT": "GRADER_STORAGE_ROOT",
    "DARTBRAINS_CACHE_DIR": "GRADER_CACHE_DIR",
}
for _old, _new in _LEGACY_ENV.items():
    if _old in os.environ and _new not in os.environ:
        os.environ[_new] = os.environ[_old]

configure(server=DEFAULT_SERVER)

__all__ = [*_client_all, "DEFAULT_SERVER", "dataset"]


class _Dataset:
    """A public Hugging Face dataset repo, through the existing loaders' path."""

    def __init__(self, name: str, repo: str) -> None:
        self.name, self.repo = name, repo

    def local_path(self, rel: str) -> str:
        from ..data._hub import download

        return download(self.repo, rel.strip("/"))

    def __repr__(self) -> str:
        return f"<Dataset {self.name} -> hf://datasets/{self.repo}>"


_PUBLIC = {
    "localizer": "dartbrains/localizer",
    "sherlock": "dartbrains/sherlock",
    "paranoia": "dartbrains/paranoia",
}


def dataset(name: str) -> _Dataset:
    """A public dataset; no sign-in. Names: localizer, sherlock, paranoia."""
    if name not in _PUBLIC:
        raise _storage.NoSuchMount(f"unknown public dataset {name!r}; have {', '.join(_PUBLIC)}")
    return _Dataset(name, _PUBLIC[name])
