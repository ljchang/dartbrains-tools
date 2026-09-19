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

In a marimo notebook, prefer the button: it never blocks, and a reader who
cannot sign in (not at Dartmouth) just keeps going with public data::

    signin = storage.signin_button(); signin        # one cell
    course = storage.course() if storage.connect(signin) else None

Public datasets need no sign-in and keep going through Hugging Face::

    storage.dataset("localizer").local_path("derivatives/betas/S01_betas.nii.gz")

Set ``DARTBRAINS_STORAGE_ROOT=/some/dir`` to run everything against a local
directory instead (the book build and tests do this).
"""

from __future__ import annotations

import os

from . import _auth, _notebook, _state
from ._auth import NotSignedIn, Token
from ._cache import cache
from ._fs import NotFound
from ._http import HttpError
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
    "connect",
    "course",
    "dataset",
    "group",
    "mount",
    "mounts",
    "private",
    "shared_cache",
    "signin",
    "signin_button",
    "signout",
    "whoami",
]


def signin(server: str | None = None, offering: str | None = None, *, force: bool = False) -> str:
    """Sign in with Dartmouth (device handshake) and remember the token.

    No-op when a valid token is cached, or when DARTBRAINS_STORAGE_ROOT selects
    the local backend. Returns the NetID ("" when local)."""
    b = _state.signin(server, offering, force=force)
    return (b.token.netid or "") if b else ""


def signin_button(server: str | None = None, offering: str | None = None):
    """A "Sign in with Dartmouth" button for a marimo cell.

    The same widget assignments use: the handshake runs in the browser and
    the token lands in the kernel when approved. Nothing blocks, so a reader
    who cannot sign in simply leaves it alone. Pair with :func:`connect`.
    """
    import marimo as mo

    if _state.use_local() or os.environ.get("GRADER_RENDER"):
        return mo.md("*Sign in with Dartmouth* — not available on the static page")
    from marimo_grader_client.widget import GraderWidget

    return mo.ui.anywidget(
        GraderWidget(
            mode="signin",
            server=(_notebook.resolve_server(server) or _auth.DEFAULT_SERVER).rstrip("/"),
            offering_id=_notebook.resolve_offering(offering) or "",
            payload={"client": "dartbrains-tools"},
        )
    )


def connect(button=None, *, offering: str | None = None, quiet: bool = False) -> bool:
    """Open storage if a sign-in is available; ``False`` otherwise. Never prompts.

    Looks at the button's token first, then at a token cached from an earlier
    run. ``False`` also covers a Dartmouth user who is not enrolled in the
    course, so callers can fall back to public data either way.
    """
    if _state.use_local():
        _state.session()
        return True
    token = None
    if button is not None:
        val = getattr(button, "value", None) or {}
        raw = (
            val.get("token")
            or getattr(getattr(button, "widget", None), "token", "")
            or getattr(button, "token", "")  # a bare GraderWidget, e.g. Grader().signin_button()
        )
        if raw:
            server = (
                val.get("server") or _notebook.resolve_server(None) or _auth.DEFAULT_SERVER
            ).rstrip("/")
            token = Token(raw, server)
    if token is None:
        token = _auth.load((_notebook.resolve_server(None) or _auth.DEFAULT_SERVER).rstrip("/"))
    if token is None or not token.valid():
        return False
    try:
        _state.adopt(token, offering)
        _state.session()
    except (HttpError, NotSignedIn) as e:
        if not quiet:
            print(
                f"storage: signed in as {token.netid} but could not open course storage ({e}); using public data"
            )
        _state.reset()
        return False
    return True


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
