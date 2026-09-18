"""The small object-store surface a Mount is built on, with two implementations.

Keys are relative to the mount's prefix. ``ObstoreFS`` roots an ``obstore``
store at the prefix so the store object marimo shows in its Files panel lists
only that subtree; ``LocalFS`` is a directory, used by the marimo-book build,
tests and offline work.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Entry:
    key: str
    size: int
    etag: str | None


class ObjectFS(Protocol):
    native: object | None  # what to hand marimo for the Files panel

    def get_bytes(self, key: str) -> bytes: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> None: ...
    def head(self, key: str) -> Entry | None: ...
    def list(self, prefix: str = "") -> list[Entry]: ...
    def download(self, key: str, dest: Path) -> None: ...
    def upload(self, src: Path, key: str) -> None: ...


class NotFound(FileNotFoundError):
    pass


# --------------------------------------------------------------------------


class ObstoreFS:
    def __init__(self, store) -> None:
        import obstore

        self._obs = obstore
        self.native = store

    def get_bytes(self, key: str) -> bytes:
        try:
            return bytes(self._obs.get(self.native, key).bytes())
        except FileNotFoundError as e:  # obstore's NotFoundError subclasses it
            raise NotFound(key) from e

    def put_bytes(self, key: str, data: bytes) -> None:
        self._obs.put(self.native, key, data)

    def delete(self, key: str) -> None:
        self._obs.delete(self.native, key)

    def head(self, key: str) -> Entry | None:
        try:
            meta = self._obs.head(self.native, key)
        except FileNotFoundError:
            return None
        return Entry(meta["path"], int(meta["size"]), meta.get("e_tag"))

    def list(self, prefix: str = "") -> list[Entry]:
        rows = self._obs.list(self.native, prefix or None).collect()
        return [Entry(r["path"], int(r["size"]), r.get("e_tag")) for r in rows]

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            resp = self._obs.get(self.native, key)
        except FileNotFoundError as e:  # obstore's NotFoundError subclasses it
            raise NotFound(key) from e
        with open(tmp, "wb") as f:
            for chunk in resp:
                f.write(chunk)
        tmp.replace(dest)

    def upload(self, src: Path, key: str) -> None:
        with open(src, "rb") as f:
            self._obs.put(self.native, key, f)


# --------------------------------------------------------------------------


class LocalFS:
    native = None

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key escapes the mount: {key!r}")
        return p

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise NotFound(key)
        return p.read_bytes()

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()

    def head(self, key: str) -> Entry | None:
        p = self._path(key)
        if not p.is_file():
            return None
        return Entry(key, p.stat().st_size, _etag(p))

    def list(self, prefix: str = "") -> list[Entry]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        out = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                key = p.relative_to(self.root).as_posix()
                out.append(Entry(key, p.stat().st_size, _etag(p)))
        return out

    def download(self, key: str, dest: Path) -> None:
        p = self._path(key)
        if not p.is_file():
            raise NotFound(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)

    def upload(self, src: Path, key: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)


def _etag(p: Path) -> str:
    st = p.stat()
    return hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
