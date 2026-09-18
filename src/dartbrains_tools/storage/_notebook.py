"""Find the grader server and offering the running notebook belongs to.

Published assignment notebooks carry them in the PEP 723 block::

    # /// script
    # grader-server = "https://grader.dartbrains.org"
    # grader-offering-id = "a8e72d80-..."
    # ///

Chapter notebooks do not, so the fall-backs are environment variables and,
last, asking the grader which offerings the signed-in student is enrolled in.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_OPEN, _CLOSE = "# /// script", "# ///"
_LINE = re.compile(r"^#\s*([A-Za-z0-9_.-]+)\s*=\s*(.+?)\s*$")


def notebook_source() -> str | None:
    for candidate in (
        os.environ.get("DARTBRAINS_NOTEBOOK"),
        os.environ.get("GRADER_NOTEBOOK_PATH"),
        _marimo_filename(),
        _main_file(),
    ):
        if candidate:
            p = Path(str(candidate))
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8")
                except OSError:
                    continue
    return None


def _marimo_filename() -> str | None:
    try:
        from marimo._runtime.context import get_context

        name = getattr(get_context(), "filename", None)
        return str(name) if name else None
    except Exception:  # noqa: BLE001 - not inside marimo
        return None


def _main_file() -> str | None:
    try:
        import __main__

        return getattr(__main__, "__file__", None)
    except Exception:  # noqa: BLE001
        return None


def script_metadata(source: str | None = None) -> dict[str, str]:
    """``grader-*`` keys from the notebook's ``# /// script`` block."""
    source = source if source is not None else notebook_source()
    if not source:
        return {}
    out: dict[str, str] = {}
    inside = False
    for line in source.splitlines():
        if line.strip() == _OPEN:
            inside = True
            continue
        if inside and line.strip() == _CLOSE:
            break
        if not inside:
            continue
        m = _LINE.match(line)
        if m and m.group(1).startswith("grader-"):
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def resolve_server(explicit: str | None = None) -> str | None:
    return (
        explicit
        or os.environ.get("DARTBRAINS_GRADER_SERVER")
        or script_metadata().get("grader-server")
    )


def resolve_offering(explicit: str | None = None) -> str | None:
    return (
        explicit
        or os.environ.get("DARTBRAINS_OFFERING")
        or script_metadata().get("grader-offering-id")
    )
