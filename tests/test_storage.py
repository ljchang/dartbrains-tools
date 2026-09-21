"""dartbrains_tools.storage without a network: local backend, token cache, mounts, @cache."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dartbrains_tools import storage
from dartbrains_tools.storage import _auth, _notebook, _runtime, _state
from dartbrains_tools.storage._fs import LocalFS
from dartbrains_tools.storage._mount import Mount


def _jwt(sub="f00abc1", exp=None):
    def seg(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'EdDSA'})}.{seg({'sub': sub, 'exp': exp or int(time.time()) + 3600})}.sig"


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    root = tmp_path / "bucket"
    (root / "course").mkdir(parents=True)
    (root / "course" / "hello.txt").write_text("course data")
    (root / "assignments" / "glm").mkdir(parents=True)
    (root / "assignments" / "glm" / "stim.txt").write_text("secret")
    monkeypatch.setenv("DARTBRAINS_STORAGE_ROOT", str(root))
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    _state.reset()
    yield root
    _state.reset()


# --------------------------------------------------------------------------
# notebook metadata, runtime, token cache
# --------------------------------------------------------------------------


def test_script_metadata_parses_grader_keys():
    src = (
        "# /// script\n"
        '# dependencies = ["marimo"]\n'
        '# grader-server = "https://grader.example"\n'
        '# grader-offering-id = "abc-123"\n'
        "# ///\n"
        "import marimo\n"
    )
    meta = _notebook.script_metadata(src)
    assert meta == {"grader-server": "https://grader.example", "grader-offering-id": "abc-123"}
    assert _notebook.script_metadata("no block here") == {}


def test_runtime_override(monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "molab")
    assert _runtime.kind() == "molab"
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "local")
    assert _runtime.kind() == "local"


def test_token_cache_local(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "local")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    server = "https://grader.example"
    assert _auth.load(server) is None

    tok = _auth.Token(_jwt(), server)
    path = _auth.save(tok)
    assert path == tmp_path / "dartbrains" / "token.json"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert _auth.load(server) == tok
    assert _auth.load(server).netid == "f00abc1"
    assert _auth.load("https://other.example") is None, "token is bound to its server"

    _auth.save(_auth.Token(_jwt(exp=int(time.time()) - 10), server))
    assert _auth.load(server) is None, "expired tokens are not returned"

    monkeypatch.setenv("DARTBRAINS_GRADER_TOKEN", _jwt(sub="envuser"))
    assert _auth.load(server).netid == "envuser", "environment wins"

    _auth.clear()
    assert not path.exists() or _auth.load(server) is None


def test_token_cache_molab_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "molab")
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OTHER=1\n")
    server = "https://grader.example"
    tok = _auth.Token(_jwt(), server)
    assert _auth.save(tok) == tmp_path / ".env"
    text = (tmp_path / ".env").read_text()
    assert "OTHER=1\n" in text and f"DARTBRAINS_GRADER_TOKEN={tok.value}\n" in text
    assert _auth.load(server) == tok
    _auth.clear()
    assert (tmp_path / ".env").read_text() == "OTHER=1\n"


# --------------------------------------------------------------------------
# local session + mounts
# --------------------------------------------------------------------------


def test_local_session_mount_table(local_root):
    assert set(storage.mounts()) >= {
        "/course",
        "/private",
        "/group",
        "/cache/shared",
        "/cache/private",
        "/assignments/glm",
    }
    assert storage.course().get("hello.txt") == "course data"
    assert storage.assignment("glm").get_bytes("stim.txt") == b"secret"
    with pytest.raises(storage.NotReleased):
        storage.assignment("midterm")
    with pytest.raises(storage.NoSuchMount):
        storage.mount("/nope")
    assert repr(storage.private()).startswith("<Mount /private (rw)")


def test_put_get_roundtrips_by_extension(local_root):
    p = storage.private()
    arr = np.arange(6).reshape(2, 3)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    p.put("w3/obj.pkl", {"k": arr})
    p.put("w3/arr.npy", arr)
    p.put("w3/table.csv", df)
    p.put("w3/table.tsv", df)
    p.put("w3/meta.json", {"n": np.int64(3), "v": arr[0]})
    p.put("w3/note.txt", "hello")
    p.put("w3/raw.bin", b"\x00\x01")
    with pytest.raises(TypeError, match=r"\.pkl"):
        p.put("w3/unknown.xyz", [1, 2])  # no guessing: say how to store it
    p.put("w3/unknown.xyz", b"ok")  # bytes always go as-is

    np.testing.assert_array_equal(p.get("w3/obj.pkl")["k"], arr)
    np.testing.assert_array_equal(p.get("w3/arr.npy"), arr)
    pd.testing.assert_frame_equal(p.get("w3/table.csv"), df)
    pd.testing.assert_frame_equal(p.get("w3/table.tsv"), df)
    assert p.get("w3/meta.json") == {"n": 3, "v": [0, 1, 2]}
    assert p.get("w3/note.txt") == "hello"
    assert p.get("w3/raw.bin") == b"\x00\x01"
    assert p.get("w3/unknown.xyz") == b"ok"

    assert p.exists("w3/note.txt") and not p.exists("w3/missing")
    assert set(p.ls("w3")) == {
        "w3/obj.pkl",
        "w3/arr.npy",
        "w3/table.csv",
        "w3/table.tsv",
        "w3/meta.json",
        "w3/note.txt",
        "w3/raw.bin",
        "w3/unknown.xyz",
    }
    assert p.glob("w3/*.csv") == ["w3/table.csv"]
    assert p.usage() > 0
    p.delete("w3/raw.bin")
    assert not p.exists("w3/raw.bin")
    with pytest.raises(storage.NotFound):
        p.get("w3/raw.bin")


def test_open_and_local_path_and_sync(local_root, tmp_path):
    p = storage.private()
    with p.open("log.txt", "w") as f:
        f.write("one\n")
    with p.open("log.txt", "a") as f:
        f.write("two\n")
    with p.open("log.txt") as f:
        assert f.read() == "one\ntwo\n"

    path = storage.course().local_path("hello.txt")
    assert Path(path).read_text() == "course data"
    assert Path(path) == local_root / "course" / "hello.txt"

    src = tmp_path / "figs"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.png").write_bytes(b"png")
    (src / "y.txt").write_text("y")
    assert set(p.sync(src, "figures")) == {"figures/a/x.png", "figures/y.txt"}
    assert p.sync(src, "figures") == [], "unchanged files are skipped"

    p.copy_from(storage.course(), "hello.txt", "copied.txt")
    assert p.get("copied.txt") == "course data"


def test_read_only_mount_refuses_writes(tmp_path):
    m = Mount("/course", "course/", "r", LocalFS(tmp_path), tmp_path / "cache")
    with pytest.raises(storage.ReadOnly):
        m.put("x.txt", "no")
    with pytest.raises(storage.ReadOnly):
        m.delete("x.txt")
    with pytest.raises(ValueError):
        m.get_bytes("../escape")


def test_remote_style_local_path_caches_by_etag(tmp_path):
    """The cache path used for R2: download once, reuse while size+etag match."""
    backing = tmp_path / "backing"
    backing.mkdir()
    (backing / "big.bin").write_bytes(b"v1")

    class Remote:  # a LocalFS that hides `.root`, so local_path() treats it as remote
        native = None

        def __init__(self, inner):
            self._i = inner

        def __getattr__(self, n):
            if n == "root":
                raise AttributeError(n)
            return getattr(self._i, n)

    m = Mount("/course", "course/x/", "r", Remote(LocalFS(backing)), tmp_path / "cache")
    first = m.local_path("big.bin")
    assert Path(first).read_bytes() == b"v1" and str(tmp_path / "cache") in first
    (backing / "big.bin").write_bytes(b"v2-longer")
    assert Path(m.local_path("big.bin")).read_bytes() == b"v2-longer", (
        "changed object re-downloaded"
    )


# --------------------------------------------------------------------------
# @cache
# --------------------------------------------------------------------------


def test_cache_decorator_tiers(local_root):
    calls = []

    @storage.cache
    def slow(x, scale=1):
        calls.append((x, scale))
        return np.ones(3) * x * scale

    np.testing.assert_array_equal(slow(2), [2, 2, 2])
    np.testing.assert_array_equal(slow(2), [2, 2, 2])
    assert calls == [(2, 1)], "second call served from cache"
    slow(3)
    slow(2, scale=2)
    assert calls == [(2, 1), (3, 1), (2, 2)]

    # The result landed in the private tier under a stable key.
    key = slow.cache_key(2)
    assert storage.mount("/cache/private").exists(f"slow/{key}.pkl")

    # A result the instructor put in the shared tier is used without computing,
    # even with a cold local cache.
    import pickle
    import shutil

    shutil.rmtree(_state.session().cache_root, ignore_errors=True)
    storage.mount("/cache/private").delete(f"slow/{slow.cache_key(9)}.pkl")
    storage.shared_cache().put_bytes(f"slow/{slow.cache_key(9)}.pkl", pickle.dumps("warmed"))
    assert slow(9) == "warmed"
    assert (9, 1) not in calls


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


def test_r2_session_credential_provider_refreshes(monkeypatch):
    pytest.importorskip("obstore")
    from dartbrains_tools.storage._session import R2Session

    broker = FakeBroker()
    sess = R2Session(broker, _payload(expires_in=60))  # about to expire
    cred = sess._provider("rw")()
    assert broker.calls == 1, "near-expiry payload triggered one refresh"
    assert cred["access_key_id"] == "rw-ak" and cred["token"] == "t2"
    assert sess._provider("ro")()["access_key_id"] == "ro-ak"
    assert broker.calls == 1, "fresh payload reused"

    m = sess.mount("/private")
    assert m.mode == "rw" and m.store is not None  # an obstore S3Store rooted at the prefix
    assert sess.mount("/course").mode == "r"
    with pytest.raises(storage.NotReleased):
        sess.mount("/assignments/midterm")


def test_signin_is_a_noop_on_the_local_backend(local_root):
    assert storage.signin() == ""
    assert storage.private().mode == "rw"


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


def test_connect_without_signin_is_false(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "local")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    monkeypatch.delenv("DARTBRAINS_STORAGE_ROOT", raising=False)
    _state.reset()
    assert storage.connect() is False
    assert storage.connect(button=None) is False


def test_connect_adopts_the_buttons_token_and_opens_a_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "local")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    monkeypatch.delenv("DARTBRAINS_STORAGE_ROOT", raising=False)
    monkeypatch.setenv("DARTBRAINS_GRADER_SERVER", "https://grader.example")
    monkeypatch.setenv("DARTBRAINS_OFFERING", "off-1")
    _state.reset()
    opened = []

    class FakeBroker:
        def __init__(self, token, offering_id=None, **kw):
            self.token, self.offering_id = token, offering_id

        def session(self, client=None):
            opened.append((self.token.netid, self.offering_id, client))
            return _payload()

    monkeypatch.setattr(_state, "Broker", FakeBroker)

    class Button:  # what mo.ui.anywidget(GraderWidget) looks like from Python
        value = {
            "token": _jwt(sub="f00abc1"),
            "netid": "f00abc1",
            "server": "https://grader.example",
        }

    pytest.importorskip("obstore")
    assert storage.connect(Button()) is True
    assert opened == [("f00abc1", "off-1", "dartbrains-tools/local")]
    assert storage.private().prefix == "users/x/abc/"
    # The token was cached, so a later run connects without the button.
    assert _auth.load("https://grader.example").netid == "f00abc1"
    _state.reset()
    assert storage.connect() is True

    # Signed in but not enrolled: the broker refuses; connect() says so and returns False.
    from dartbrains_tools.storage._http import HttpError

    class Refusing(FakeBroker):
        def session(self, client=None):
            raise HttpError(404, "not_found", "offering not found")

    monkeypatch.setattr(_state, "Broker", Refusing)
    _state.reset()
    assert storage.connect(Button(), quiet=True) is False


def test_signin_button_is_static_under_local_backend(local_root):
    pytest.importorskip("marimo")
    el = storage.signin_button()
    assert "Sign in with Dartmouth" in el.text


def test_script_metadata_reads_tool_grader_table():
    src = (
        "# /// script\n"
        '# requires-python = ">=3.11"\n'
        '# dependencies = ["marimo"]\n'
        "#\n"
        "# [tool.grader]\n"
        '# server = "https://grader.example"\n'
        '# course = "neuroimaging"\n'
        '# term = "2026-fall"\n'
        "# ///\n"
        "import marimo\n"
    )
    assert _notebook.script_metadata(src) == {
        "grader-server": "https://grader.example",
        "grader-course": "neuroimaging",
        "grader-term": "2026-fall",
    }


def test_broker_picks_the_offering_by_course_and_term(monkeypatch):
    from dartbrains_tools.storage._broker import Broker
    from dartbrains_tools.storage._http import HttpError

    me = {
        "netid": "f00abc1",
        "enrollments": [
            {"offering_id": "old", "course_slug": "neuroimaging", "term": "2025-fall"},
            {"offering_id": "new", "course_slug": "neuroimaging", "term": "2026-fall"},
        ],
    }
    b = Broker(_auth.Token(_jwt(), "https://g"), course="neuroimaging", term="2026-fall")
    monkeypatch.setattr(b, "me", lambda: me)
    assert b.pick_offering() == "new"
    other = Broker(_auth.Token(_jwt(), "https://g"), course="neuroimaging", term="2027-winter")
    monkeypatch.setattr(other, "me", lambda: me)
    with pytest.raises(HttpError, match="not enrolled"):
        other.pick_offering()


def test_assignment_card_uses_tool_grader(monkeypatch):
    pytest.importorskip("marimo")
    from dartbrains_tools.notebook_utils import assignment_card

    monkeypatch.setenv("DARTBRAINS_GRADER_SERVER", "https://grader.example")
    monkeypatch.setenv("DARTBRAINS_COURSE", "neuroimaging")
    monkeypatch.setenv("DARTBRAINS_TERM", "2026-fall")
    html = assignment_card("glm-single-subject").text
    assert "https://grader.example/a/neuroimaging/2026-fall/glm-single-subject/molab" in html
    assert "student.py" in html and "Glm Single Subject" in html
    monkeypatch.delenv("DARTBRAINS_COURSE")
    monkeypatch.setenv("DARTBRAINS_NOTEBOOK", "/nonexistent")
    assert "cannot be built" in assignment_card("glm").text


def test_connect_accepts_a_bare_grader_widget(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_RUNTIME", "local")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("DARTBRAINS_GRADER_TOKEN", raising=False)
    monkeypatch.delenv("DARTBRAINS_STORAGE_ROOT", raising=False)
    monkeypatch.setenv("DARTBRAINS_GRADER_SERVER", "https://grader.example")
    monkeypatch.setenv("DARTBRAINS_OFFERING", "off-1")
    _state.reset()

    class FakeBroker:
        def __init__(self, token, offering_id=None, **kw):
            self.token, self.offering_id = token, offering_id

        def session(self, client=None):
            return _payload()

    monkeypatch.setattr(_state, "Broker", FakeBroker)

    class Bare:  # marimo_grader_client.GraderWidget as returned by Grader().signin_button()
        token = _jwt(sub="f00abc1")

    pytest.importorskip("obstore")
    assert storage.connect(Bare()) is True
    assert storage.whoami() == "f00abc1"


def test_assignment_card_is_empty_on_the_static_site(monkeypatch):
    pytest.importorskip("marimo")
    from dartbrains_tools.notebook_utils import assignment_card

    monkeypatch.setenv("DARTBRAINS_COURSE", "neuroimaging")
    monkeypatch.setenv("DARTBRAINS_TERM", "2026-fall")
    monkeypatch.setenv("GRADER_RENDER", "1")
    assert assignment_card("glm").text == ""


def test_molab_is_recognised_from_two_sandbox_signals(monkeypatch, tmp_path):
    from dartbrains_tools.storage import _runtime

    monkeypatch.delenv("DARTBRAINS_RUNTIME", raising=False)
    monkeypatch.delenv("MARIMO_MANAGE_SCRIPT_METADATA", raising=False)
    monkeypatch.setattr(_runtime.sys, "platform", "linux")
    monkeypatch.setattr(_runtime.socket, "gethostname", lambda: "laptop.local")
    monkeypatch.setattr(_runtime.sys, "executable", "/usr/bin/python3")
    monkeypatch.chdir(tmp_path)
    assert _runtime.kind() == "local"
    # one signal alone is not enough
    monkeypatch.setenv("MARIMO_MANAGE_SCRIPT_METADATA", "true")
    assert _runtime.kind() == "local"
    # two are: the sandbox interpreter plus the env var
    monkeypatch.setattr(_runtime.sys, "executable", "/tmp/uv-venv/bin/python")
    assert _runtime.kind() == "molab"
    # the pod hostname plus the interpreter, without the env var
    monkeypatch.delenv("MARIMO_MANAGE_SCRIPT_METADATA")
    monkeypatch.setattr(
        _runtime.socket, "gethostname", lambda: "cc1bc46e-05af-4f5d-a1af-b9b6f129d443-gf7hb"
    )
    assert _runtime.kind() == "molab"
    assert _runtime.token_file() == tmp_path / ".env"
