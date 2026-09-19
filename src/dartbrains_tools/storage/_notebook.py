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
    """Grader identity from the notebook's ``# /// script`` block.

    Two spellings, normalised to ``grader-<key>``: the top-level ``grader-*``
    keys a published assignment carries, and the ``[tool.grader]`` table
    marimo-book's ``sync-deps`` writes into chapters (``server``, ``course``,
    ``term``). Top-level keys win when both are present.
    """
    source = source if source is not None else notebook_source()
    if not source:
        return {}
    block: list[str] = []
    inside = False
    for line in source.splitlines():
        if line.strip() == _OPEN:
            inside = True
            continue
        if inside and line.strip() == _CLOSE:
            break
        if inside:
            block.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    if not block:
        return {}
    out: dict[str, str] = {}
    try:
        import tomllib

        table = tomllib.loads("\n".join(block))
    except Exception:  # noqa: BLE001 - fall back to the line scan below
        table = {}
    for k, v in (table.get("tool", {}) or {}).get("grader", {}).items():
        out[f"grader-{k}"] = str(v)
    for k, v in table.items():
        if isinstance(k, str) and k.startswith("grader-"):
            out[k] = str(v)
    if not out:  # a block tomllib could not parse: keep the old line scan
        for line in block:
            m = _LINE.match("# " + line)
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


def resolve_course(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("DARTBRAINS_COURSE") or script_metadata().get("grader-course")


def resolve_term(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("DARTBRAINS_TERM") or script_metadata().get("grader-term")
