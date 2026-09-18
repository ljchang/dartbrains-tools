"""The one HTTP helper the storage layer needs: JSON in, JSON out, stdlib only.

Kept dependency-free so importing ``dartbrains_tools.storage`` never drags in
``httpx``/``requests`` -- the only heavyweight import (``obstore``) is deferred
to the moment a bucket is actually opened.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class HttpError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code


def request(
    method: str,
    url: str,
    body: dict | None = None,
    *,
    token: str | None = None,
    timeout: float = 30,
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            err = json.loads(raw).get("error") or {}
        except ValueError:
            err = {}
        raise HttpError(
            e.code,
            err.get("code", "http_error"),
            err.get("message", raw[:200].decode(errors="replace")),
        ) from None
    except urllib.error.URLError as e:
        raise HttpError(0, "unreachable", str(e.reason)) from None
    return json.loads(raw) if raw else {}
