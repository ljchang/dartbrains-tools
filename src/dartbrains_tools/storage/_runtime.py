"""Where is this kernel running, and where may it keep things?

Three runtimes matter and they differ in what survives:

``local``
    A laptop or Discovery. ``~/.config`` and ``~/.cache`` persist.
``molab``
    Persists only files made through its file browser plus ``.env``; anything
    the kernel writes to disk is gone at teardown, and every "Open in molab"
    click is a separate sandbox. So the token goes in ``.env`` (which molab
    also keeps out of forks) and the disk cache is treated as scratch.
``wasm``
    Pyodide in the browser. No ``urllib`` to the grader, no SigV4; the
    storage layer is not available there yet (the public HF path still is).
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

KINDS = ("local", "molab", "wasm")


def kind() -> str:
    override = os.environ.get("DARTBRAINS_RUNTIME")
    if override in KINDS:
        return override
    if sys.platform == "emscripten":
        return "wasm"
    if _looks_like_molab():
        return "molab"
    return "local"


def _looks_like_molab() -> bool:
    # Best effort until the molab spike settles what the sandbox exposes.
    # DARTBRAINS_RUNTIME=molab in the notebook's .env is the reliable override.
    if any(k.upper().startswith("MOLAB") for k in os.environ):
        return True
    try:
        host = socket.gethostname()
    except OSError:
        return False
    return "molab" in host.lower()


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "dartbrains"


def token_file() -> Path:
    """Where the grader token is cached between runs."""
    if kind() == "molab":
        return Path.cwd() / ".env"
    return config_dir() / "token.json"


def cache_dir() -> Path:
    """Local object cache. Durable on a laptop; scratch on molab."""
    override = os.environ.get("DARTBRAINS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if kind() == "molab":
        return Path("/tmp/dartbrains/objects")
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "dartbrains" / "objects"
