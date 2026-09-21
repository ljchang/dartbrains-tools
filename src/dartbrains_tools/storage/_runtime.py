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
import re
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


_POD_HOST = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[0-9a-z]{5}$"
)


def _looks_like_molab() -> bool:
    """molab's sandbox, as observed from inside one (2026-09-21):

        hostname   cc1bc46e-05af-4f5d-a1af-b9b6f129d443-gf7hb   (a pod name)
        cwd        /marimo
        executable /tmp/uv-venv/bin/python
        env        MARIMO_MANAGE_SCRIPT_METADATA=true, MARIMO_CACHE_STORE_ALLOWLIST, ...

    Nothing says "molab", so this counts signals and wants two of them: any
    one could happen on a laptop (a /marimo directory, a uv venv under /tmp),
    two together do not.
    """
    signals = 0
    try:
        if Path.cwd() == Path("/marimo") or Path("/marimo").is_dir():
            signals += 1
    except OSError:
        pass
    if sys.executable.startswith("/tmp/uv-venv/"):
        signals += 1
    if os.environ.get("MARIMO_MANAGE_SCRIPT_METADATA"):
        signals += 1
    try:
        if _POD_HOST.match(socket.gethostname()):
            signals += 1
    except OSError:
        pass
    return signals >= 2


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
