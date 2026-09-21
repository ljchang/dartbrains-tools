"""dartbrains_tools.storage: the DartBrains wrapper over marimo_grader_client.storage."""

from __future__ import annotations

import pytest

from dartbrains_tools import storage


def test_public_dataset_goes_through_hub(monkeypatch):
    seen = {}

    def fake_download(repo, filename):
        seen["repo"], seen["file"] = repo, filename
        return "/cache/x"

    monkeypatch.setattr("dartbrains_tools.data._hub.download", fake_download)
    assert (
        storage.dataset("localizer").local_path("/derivatives/betas/S01_betas.nii.gz") == "/cache/x"
    )
    assert seen == {"repo": "dartbrains/localizer", "file": "derivatives/betas/S01_betas.nii.gz"}
    with pytest.raises(storage.NoSuchMount):
        storage.dataset("nope")


# --------------------------------------------------------------------------
# R2 session (no network: fake broker; obstore only for store construction)
# --------------------------------------------------------------------------


def _payload(expires_in=3600):
    from datetime import UTC, datetime, timedelta

    exp = (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat()
    return {
        "backend": "r2",
        "endpoint": "https://acct.r2.cloudflarestorage.com",
        "bucket": "dartbrains",
        "region": "auto",
        "expires_at": exp,
        "mounts": [
            {"logical": "/course", "prefix": "course/x/", "mode": "r", "cred": "ro"},
            {"logical": "/private", "prefix": "users/x/abc/", "mode": "rw", "cred": "rw"},
        ],
        "public": [],
        "credentials": {
            "ro": {"access_key_id": "ro-ak", "secret_access_key": "s", "session_token": "t1"},
            "rw": {"access_key_id": "rw-ak", "secret_access_key": "s", "session_token": "t2"},
        },
    }


class FakeBroker:
    def __init__(self):
        self.calls = 0

    def session(self, client=None):
        self.calls += 1
        return _payload()



def test_localizer_filename_matches_get_file(monkeypatch):
    from dartbrains_tools.data import localizer

    seen = []
    monkeypatch.setattr(localizer, "_download", lambda f: seen.append(f) or f"/cache/{f}")
    rel = localizer.filename("S01", "derivatives", "bold")
    assert rel.endswith("sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    assert localizer.get_file("S01", "derivatives", "bold") == f"/cache/{rel}"
    assert (
        localizer.filename("S01", "raw", "events", ".tsv")
        == "sub-S01/func/sub-S01_task-localizer_events.tsv"
    )



def test_assignment_card_uses_tool_grader(monkeypatch):
    pytest.importorskip("marimo")
    from dartbrains_tools.notebook_utils import assignment_card

    monkeypatch.setenv("GRADER_SERVER", "https://grader.example")
    monkeypatch.setenv("GRADER_COURSE", "neuroimaging")
    monkeypatch.setenv("GRADER_TERM", "2026-fall")
    html = assignment_card("glm-single-subject").text
    assert "https://grader.example/a/neuroimaging/2026-fall/glm-single-subject/molab" in html
    assert "student.py" in html and "Glm Single Subject" in html
    monkeypatch.delenv("GRADER_COURSE")
    monkeypatch.setenv("GRADER_NOTEBOOK_PATH", "/nonexistent")
    assert "cannot be built" in assignment_card("glm").text



def test_assignment_card_is_empty_on_the_static_site(monkeypatch):
    pytest.importorskip("marimo")
    from dartbrains_tools.notebook_utils import assignment_card

    monkeypatch.setenv("GRADER_COURSE", "neuroimaging")
    monkeypatch.setenv("GRADER_TERM", "2026-fall")
    monkeypatch.setenv("GRADER_RENDER", "1")
    assert assignment_card("glm").text == ""



def test_wrapper_sets_the_dartbrains_server_and_reexports_the_client_api():
    from marimo_grader_client.storage import _state

    assert _state.DEFAULTS["server"] == "https://grader.dartbrains.org"
    for name in ("signin_button", "connect", "course", "private", "assignment", "cache", "dataset"):
        assert hasattr(storage, name), name


def test_legacy_environment_names_are_honoured(monkeypatch):
    import importlib

    monkeypatch.setenv("DARTBRAINS_STORAGE_ROOT", "/tmp/legacy-root")
    monkeypatch.delenv("GRADER_STORAGE_ROOT", raising=False)
    import dartbrains_tools.storage as s

    importlib.reload(s)
    import os

    assert os.environ["GRADER_STORAGE_ROOT"] == "/tmp/legacy-root"
