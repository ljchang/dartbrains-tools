"""Process-wide session state: sign in once, open the mount table once."""

from __future__ import annotations

import os

from . import _auth, _notebook, _runtime
from ._broker import Broker
from ._session import LocalSession, R2Session

_session = None
_broker: Broker | None = None


def use_local() -> bool:
    return (
        bool(os.environ.get("DARTBRAINS_STORAGE_ROOT"))
        or os.environ.get("DARTBRAINS_STORAGE") == "local"
    )


def signin(
    server: str | None = None, offering: str | None = None, *, force: bool = False, echo=print
) -> Broker | None:
    global _broker, _session
    if use_local():
        return None  # local backend: nothing to sign in to (the book build, tests)
    token = _auth.signin(_notebook.resolve_server(server), force=force, echo=echo)
    if _broker is None or force or _broker.token.value != token.value:
        _broker = _make_broker(token, offering)
        _session = None
    return _broker


def _make_broker(token, offering: str | None) -> Broker:
    return Broker(
        token,
        _notebook.resolve_offering(offering),
        course=_notebook.resolve_course(),
        term=_notebook.resolve_term(),
    )


def adopt(token, offering: str | None = None) -> Broker:
    """Use a token obtained elsewhere (the sign-in button) and remember it."""
    global _broker, _session
    _auth.save(token)
    if _broker is None or _broker.token.value != token.value:
        _broker = _make_broker(token, offering)
        _session = None
    return _broker


def connected() -> bool:
    return _session is not None


def broker() -> Broker:
    if _broker is None:
        return signin()
    if not _broker.token.valid():
        return signin(force=True)
    return _broker


def session():
    global _session
    if _session is None:
        if use_local():
            _session = LocalSession()
        else:
            if _runtime.kind() == "wasm":
                raise RuntimeError(
                    "dartbrains.storage is not available in the browser workbench yet; "
                    "open the notebook in molab or locally"
                )
            b = broker()
            _session = R2Session(
                b,
                b.session(client=f"dartbrains-tools/{_runtime.kind()}"),
                client=f"dartbrains-tools/{_runtime.kind()}",
            )
    return _session


def reset() -> None:
    global _broker, _session
    _broker = None
    _session = None
