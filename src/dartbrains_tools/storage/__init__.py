"""DartBrains storage: one API over public datasets, private class data, and
each student's own space -- whichever backend holds it.

::

    from dartbrains_tools import storage

    storage.signin()                          # once; cached afterwards
    course  = storage.course()                # class datasets, read-only
    private = storage.private()               # yours, read-write
    group   = storage.group()                 # your project group's
    exam    = storage.assignment("midterm")   # read-only, only once released

    path = course.local_path("sherlock/sub-01/bold.nii.gz")   # ordinary path
    private.put("week3/betas.pkl", betas)
    betas = private.get("week3/betas.pkl")

    @storage.cache
    def fit(subject): ...                      # computed once per argument set

Public datasets need no sign-in and keep going through Hugging Face::

    storage.dataset("localizer").local_path("derivatives/betas/S01_betas.nii.gz")

Set ``DARTBRAINS_STORAGE_ROOT=/some/dir`` to run everything against a local
directory instead (the book build and tests do this).
"""

from __future__ import annotations

from . import _state
from ._auth import NotSignedIn, Token
from ._cache import cache
from ._fs import NotFound
from ._mount import Mount, ReadOnly
from ._session import NoSuchMount, NotReleased

__all__ = [
    "Mount",
    "NoSuchMount",
    "NotFound",
    "NotReleased",
    "NotSignedIn",
    "ReadOnly",
    "Token",
    "assignment",
    "cache",
    "course",
    "dataset",
    "group",
    "mount",
    "mounts",
    "private",
    "shared_cache",
    "signin",
    "signout",
    "whoami",
]


def signin(server: str | None = None, offering: str | None = None, *, force: bool = False) -> str:
    """Sign in with Dartmouth (device handshake) and remember the token.

    No-op when a valid token is cached, or when DARTBRAINS_STORAGE_ROOT selects
    the local backend. Returns the NetID ("" when local)."""
    b = _state.signin(server, offering, force=force)
    return (b.token.netid or "") if b else ""


def signout() -> None:
    from . import _auth

    _auth.clear()
    _state.reset()


def whoami() -> str | None:
    return _state.broker().token.netid


def mounts() -> list[str]:
    """Logical roots this session may reach."""
    return _state.session().logicals()


def mount(logical: str) -> Mount:
    return _state.session().mount(logical)


def course() -> Mount:
    return mount("/course")


def private() -> Mount:
    return mount("/private")


def shared_cache() -> Mount:
    return mount("/cache/shared")


def group(slug: str | None = None) -> Mount:
    """Your project group's folder. ``slug`` picks one if you are in several."""
    return mount(f"/group/{slug}" if slug else "/group")


def assignment(slug: str) -> Mount:
    """An assignment's protected data; :class:`NotReleased` before its release."""
    return mount(f"/assignments/{slug}")


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
        raise NoSuchMount(f"unknown public dataset {name!r}; have {', '.join(_PUBLIC)}")
    return _Dataset(name, _PUBLIC[name])
