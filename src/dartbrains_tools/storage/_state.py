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
) -> Broker:
    global _broker, _session
    if use_local():
        raise RuntimeError("DARTBRAINS_STORAGE_ROOT is set: storage is local, no sign-in needed")
    token = _auth.signin(_notebook.resolve_server(server), force=force, echo=echo)
    if _broker is None or force or _broker.token.value != token.value:
        _broker = Broker(token, _notebook.resolve_offering(offering))
        _session = None
    return _broker


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
