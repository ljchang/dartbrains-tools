"""Sign in to the grader from Python and keep the token between runs.

The grader's notebook token is an 8-hour EdDSA JWT minted after a device
handshake (RFC 8628 shape): we ask for a code, the student approves it in a
browser tab through Dartmouth SSO, and we poll until it is approved. The same
flow the grader CLI and the submit widget use; here it runs in the kernel so
the credentials can be used by Python.

Where the token is kept depends on the runtime (see :mod:`._runtime`): a
0600 JSON file under ``~/.config`` locally, a ``DARTBRAINS_GRADER_TOKEN=`` line
in molab's ``.env`` (persisted, excluded from forks). ``DARTBRAINS_GRADER_TOKEN``
in the environment always wins, which is how the build and tests inject one.
"""

from __future__ import annotations

import base64
import json
import os
import time
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import _runtime
from ._http import HttpError, request

TOKEN_ENV = "DARTBRAINS_GRADER_TOKEN"
DEFAULT_SERVER = "https://grader.dartbrains.org"
CLIENT = "dartbrains-tools"


class NotSignedIn(RuntimeError):
    pass


@dataclass(frozen=True)
class Token:
    value: str
    server: str

    @property
    def claims(self) -> dict:
        return _claims(self.value)

    @property
    def netid(self) -> str | None:
        return self.claims.get("sub")

    @property
    def expires_at(self) -> datetime | None:
        exp = self.claims.get("exp")
        return datetime.fromtimestamp(exp, tz=UTC) if exp else None

    def valid(self, margin: int = 300) -> bool:
        exp = self.expires_at
        return exp is not None and exp.timestamp() - margin > time.time()


def _claims(jwt: str) -> dict:
    """The payload segment, unverified -- only the server verifies. We just
    need ``sub`` and ``exp`` to know whom we are and whether to re-prompt."""
    try:
        seg = jwt.split(".")[1]
        seg += "=" * (-len(seg) % 4)
        return json.loads(base64.urlsafe_b64decode(seg))
    except Exception:  # noqa: BLE001 - any malformed token is simply "no claims"
        return {}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def load(server: str) -> Token | None:
    env = os.environ.get(TOKEN_ENV)
    if env:
        t = Token(env, server)
        return t if t.valid() else None
    path = _runtime.token_file()
    if not path.exists():
        return None
    try:
        if path.name == ".env":
            value = _read_env_line(path)
            saved_server = server
        else:
            data = json.loads(path.read_text())
            value, saved_server = data.get("token"), data.get("server", server)
    except (OSError, ValueError):
        return None
    if not value or saved_server != server:
        return None
    t = Token(value, server)
    return t if t.valid() else None


def save(token: Token) -> Path:
    path = _runtime.token_file()
    if path.name == ".env":
        _write_env_line(path, token.value)
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_private(path, json.dumps({"server": token.server, "token": token.value}))
    return path


def _write_private(path: Path, text: str) -> None:
    """Create with mode 0600 *before* any bytes land, so no umask ever exposes
    a token; an existing file keeps whatever mode it has but gets truncated."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def clear() -> None:
    path = _runtime.token_file()
    if not path.exists():
        return
    if path.name == ".env":
        _write_env_line(path, None)
    else:
        path.unlink()


def _read_env_line(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith(f"{TOKEN_ENV}="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def _write_env_line(path: Path, value: str | None) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [ln for ln in lines if not ln.startswith(f"{TOKEN_ENV}=")]
    if value:
        lines.append(f"{TOKEN_ENV}={value}")
    _write_private(path, "\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------
# The handshake
# --------------------------------------------------------------------------


def device_login(
    server: str,
    *,
    echo=print,
    open_browser: bool | None = None,
    timeout: float = 600,
) -> Token:
    start = request("POST", f"{server}/api/v1/auth/device", {"client": CLIENT})
    url, code = start["verification_url"], start["user_code"]
    echo(f"Sign in with Dartmouth at:\n  {url}\n(code {code})")
    if open_browser is None:
        open_browser = _runtime.kind() == "local"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless; the printed link still works
            pass
    interval = float(start.get("interval", 3))
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        poll = request(
            "POST", f"{server}/api/v1/auth/device/token", {"device_code": start["device_code"]}
        )
        status = poll.get("status")
        if status == "approved":
            echo(f"Signed in as {poll.get('netid')}")
            return Token(poll["access_token"], server)
        if status == "expired":
            raise NotSignedIn("the sign-in code expired before it was approved; try again")
    raise NotSignedIn("timed out waiting for sign-in approval")


def signin(server: str | None = None, *, force: bool = False, echo=print) -> Token:
    """A valid token for ``server``: cached if there is one, else a fresh handshake."""
    server = (server or os.environ.get("DARTBRAINS_GRADER_SERVER") or DEFAULT_SERVER).rstrip("/")
    if not force:
        cached = load(server)
        if cached:
            return cached
    if _runtime.kind() == "wasm":
        raise NotSignedIn("sign-in from the in-browser workbench is not supported yet")
    try:
        token = device_login(server, echo=echo)
    except HttpError as e:
        raise NotSignedIn(f"could not reach the grader at {server}: {e}") from e
    save(token)
    return token
