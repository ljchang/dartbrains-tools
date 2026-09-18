"""A storage session: the broker's mount table turned into Mount objects.

``R2Session`` builds one ``obstore.store.S3Store`` per mount, rooted at the
mount's prefix. Each store gets a *credential provider* rather than static
keys: obstore calls it whenever the credential it holds is about to expire,
and the provider asks the broker for a fresh session. So an hour-long
credential renews itself for as long as the grader token lasts, without the
notebook noticing.

``LocalSession`` fakes the same mount table on top of a directory
(``DARTBRAINS_STORAGE_ROOT``) for the book build, tests and offline work.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from . import _runtime
from ._fs import LocalFS, ObstoreFS
from ._mount import Mount


class NotReleased(LookupError):
    pass


class NoSuchMount(LookupError):
    pass


class _Session:
    def __init__(self, mounts: list[dict], cache_root: Path) -> None:
        self._specs = {m["logical"]: m for m in mounts}
        self._mounts: dict[str, Mount] = {}
        self.cache_root = cache_root

    def logicals(self) -> list[str]:
        return list(self._specs)

    def mount(self, logical: str) -> Mount:
        if logical not in self._specs:
            if logical.startswith("/assignments/"):
                raise NotReleased(
                    f"{logical} is not available: the assignment has not been released "
                    "(or has closed), or you are not enrolled"
                )
            raise NoSuchMount(f"no such mount {logical!r}; have {', '.join(self._specs)}")
        if logical not in self._mounts:
            spec = self._specs[logical]
            self._mounts[logical] = Mount(
                logical=logical,
                prefix=spec["prefix"],
                mode=spec["mode"],
                fs=self._fs_for(spec),
                cache_root=self.cache_root,
            )
        return self._mounts[logical]

    def _fs_for(self, spec: dict):  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------


class R2Session(_Session):
    def __init__(self, broker, payload: dict, *, client: str | None = None) -> None:
        super().__init__(payload["mounts"], _runtime.cache_dir())
        self.broker = broker
        self.client = client
        self.payload = payload

    # -- credentials -------------------------------------------------------

    def refresh(self) -> dict:
        self.payload = self.broker.session(self.client)
        return self.payload

    def _provider(self, cred: str):
        """An obstore credential provider for one of the two credentials."""

        def provide() -> dict:
            exp = _parse(self.payload["expires_at"])
            if exp is None or (exp - datetime.now(UTC)).total_seconds() < 120:
                self.refresh()
                exp = _parse(self.payload["expires_at"])
            c = self.payload["credentials"][cred]
            return {
                "access_key_id": c["access_key_id"],
                "secret_access_key": c["secret_access_key"],
                "token": c["session_token"],
                "expires_at": exp,
            }

        return provide

    def _fs_for(self, spec: dict):
        from obstore.store import S3Store

        store = S3Store(
            self.payload["bucket"],
            prefix=spec["prefix"].rstrip("/"),
            endpoint=self.payload["endpoint"],
            region=self.payload.get("region", "auto"),
            virtual_hosted_style_request=False,
            credential_provider=self._provider(spec["cred"]),
        )
        return ObstoreFS(store)


# --------------------------------------------------------------------------


STANDARD = ("/course", "/cache/shared", "/private", "/cache/private", "/group")


class LocalSession(_Session):
    """Mounts over a directory tree shaped like the bucket::

    <root>/course/            -> /course          (rw locally; nothing to protect)
    <root>/assignments/<slug> -> /assignments/<slug>
    <root>/users/me/          -> /private
    <root>/groups/me/         -> /group
    <root>/cache/shared/      -> /cache/shared
    <root>/cache/users/me/    -> /cache/private
    """

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        root = Path(
            root
            or os.environ.get("DARTBRAINS_STORAGE_ROOT")
            or _runtime.cache_dir().parent / "local"
        )
        self.root = root
        mounts = [
            {"logical": "/course", "prefix": "course/", "mode": "rw", "cred": "rw"},
            {"logical": "/cache/shared", "prefix": "cache/shared/", "mode": "rw", "cred": "rw"},
            {"logical": "/private", "prefix": "users/me/", "mode": "rw", "cred": "rw"},
            {"logical": "/cache/private", "prefix": "cache/users/me/", "mode": "rw", "cred": "rw"},
            {"logical": "/group", "prefix": "groups/me/", "mode": "rw", "cred": "rw"},
        ]
        assignments = root / "assignments"
        if assignments.is_dir():
            for d in sorted(p for p in assignments.iterdir() if p.is_dir()):
                mounts.append(
                    {
                        "logical": f"/assignments/{d.name}",
                        "prefix": f"assignments/{d.name}/",
                        "mode": "rw",
                        "cred": "rw",
                    }
                )
        super().__init__(mounts, root / ".cache")

    def _fs_for(self, spec: dict):
        return LocalFS(self.root / spec["prefix"].rstrip("/"))


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
