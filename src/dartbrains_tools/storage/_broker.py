"""Client for the grader's storage routes."""

from __future__ import annotations

from ._auth import Token
from ._http import HttpError, request


class Broker:
    def __init__(self, token: Token, offering_id: str | None = None) -> None:
        self.token = token
        self.server = token.server
        self.offering_id = offering_id

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        return request(method, f"{self.server}/api/v1{path}", body, token=self.token.value)

    # -- identity ----------------------------------------------------------

    def me(self) -> dict:
        return self._call("GET", "/auth/me")

    def pick_offering(self) -> str:
        """The one active enrollment, when the notebook did not say which."""
        if self.offering_id:
            return self.offering_id
        me = self.me()
        active = [e for e in me.get("enrollments", []) if e.get("status", "active") == "active"]
        if len(active) == 1:
            self.offering_id = active[0]["offering_id"]
            return self.offering_id
        if not active:
            raise HttpError(403, "not_enrolled", f"{me.get('netid')} is not enrolled in any course")
        names = ", ".join(f"{e.get('course_slug')}/{e.get('term')}" for e in active)
        raise HttpError(
            400,
            "ambiguous_offering",
            f"enrolled in several offerings ({names}); pass offering= or set DARTBRAINS_OFFERING",
        )

    # -- storage -----------------------------------------------------------

    def storage_me(self) -> dict:
        return self._call("GET", f"/offerings/{self.pick_offering()}/storage/me")

    def session(self, client: str | None = None) -> dict:
        return self._call(
            "POST", f"/offerings/{self.pick_offering()}/storage/session", {"client": client}
        )

    def presign(self, path: str, method: str = "GET", expires: int = 900) -> dict:
        return self._call(
            "POST",
            f"/offerings/{self.pick_offering()}/storage/presign",
            {"path": path, "method": method, "expires": expires},
        )
