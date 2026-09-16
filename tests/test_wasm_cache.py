"""Tests for the browser-side dataset cache (data/_wasm_cache.py).

The download path itself needs Pyodide and is verified in a browser; what is
testable here is everything around it -- when the cache engages, where it puts
files, and the eviction policy that keeps datasets from crowding a student's
saved notebooks out of the origin's storage.
"""

from __future__ import annotations

import json
import os

import pytest

from dartbrains_tools.data import _hub, _wasm_cache

# --- when the cache engages ---------------------------------------------------


def test_inactive_off_wasm():
    assert not _wasm_cache.active()


def test_active_under_pyodide(monkeypatch):
    monkeypatch.setattr(_wasm_cache.sys, "platform", "emscripten")
    monkeypatch.delenv("DARTBRAINS_WASM_CACHE", raising=False)
    assert _wasm_cache.active()


def test_opt_out_env_var(monkeypatch):
    monkeypatch.setattr(_wasm_cache.sys, "platform", "emscripten")
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE", "0")
    assert not _wasm_cache.active()


def test_hub_dispatches_to_the_cache_when_active(monkeypatch):
    """`_hub.download` must not reach for huggingface_hub in the browser: the
    import itself fails there (no `termios`)."""
    calls = []
    monkeypatch.setattr(_wasm_cache, "active", lambda: True)
    monkeypatch.setattr(_wasm_cache, "download", lambda r, f: calls.append((r, f)) or "/p")

    assert _hub.download("dartbrains/localizer", "sub-S01.nii.gz") == "/p"
    assert calls == [("dartbrains/localizer", "sub-S01.nii.gz")]


def test_hub_uses_huggingface_hub_off_wasm(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kw: seen.update(kw) or "/cached",
    )
    assert _hub.download("dartbrains/localizer", "f.nii.gz") == "/cached"
    assert seen["repo_type"] == "dataset"


def test_data_package_does_not_import_huggingface_hub():
    """Regression: a module-level import here breaks every WASM notebook."""
    import subprocess
    import sys

    code = (
        "import dartbrains_tools.data\n"
        "import sys\n"
        "sys.exit(1 if 'huggingface_hub' in sys.modules else 0)\n"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- where files land ---------------------------------------------------------


def test_cache_root_is_its_own_mount_not_marimos(monkeypatch):
    """The cache must never share marimo's mount: flushing a stale view of it
    deletes notebooks another tab saved (reproduced in Chrome)."""
    monkeypatch.setattr(_wasm_cache, "_ensure_mount", lambda: True)
    assert _wasm_cache.cache_root() == _wasm_cache.CACHE_MOUNT
    assert not _wasm_cache.CACHE_MOUNT.startswith("/marimo")


def test_cache_root_falls_back_without_a_mount(monkeypatch):
    monkeypatch.setattr(_wasm_cache, "_ensure_mount", lambda: False)
    assert _wasm_cache.cache_root() == _wasm_cache.SESSION_ROOT


def test_resolve_url_shape():
    url = _wasm_cache.RESOLVE_URL.format(repo_id="dartbrains/localizer", filename="a/b.nii.gz")
    assert url == "https://huggingface.co/datasets/dartbrains/localizer/resolve/main/a/b.nii.gz"


# --- the eviction budget ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_mb"),
    [
        (None, _wasm_cache.DEFAULT_BUDGET_MB),
        ("512", 512),
        ("nonsense", _wasm_cache.DEFAULT_BUDGET_MB),
    ],
)
def test_budget_from_env(monkeypatch, value, expected_mb):
    if value is None:
        monkeypatch.delenv("DARTBRAINS_WASM_CACHE_MB", raising=False)
    else:
        monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", value)
    assert _wasm_cache._budget_bytes() == expected_mb * 1024 * 1024


def _seed(tmp_path, monkeypatch, sizes_by_age):
    """Write files whose mtimes increase with position: oldest use first."""
    monkeypatch.setattr(_wasm_cache, "cache_root", lambda: str(tmp_path))
    paths = []
    for i, size in enumerate(sizes_by_age):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(b"\0" * size)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
        paths.append(p)
    return paths


def test_make_room_evicts_least_recently_used_first(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "1")
    budget = 1024 * 1024
    oldest, middle, newest = _seed(tmp_path, monkeypatch, [budget // 2, budget // 4, budget // 4])

    _wasm_cache._make_room(budget // 2)

    assert not oldest.exists()
    assert middle.exists() and newest.exists()


def test_make_room_keeps_everything_when_it_fits(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "8")
    kept = _seed(tmp_path, monkeypatch, [1024, 1024])

    _wasm_cache._make_room(1024)

    assert all(p.exists() for p in kept)


def test_a_zero_budget_disables_eviction(tmp_path, monkeypatch):
    """0 means "no ceiling", not "evict everything"."""
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "0")
    kept = _seed(tmp_path, monkeypatch, [4096])

    _wasm_cache._make_room(10 * 1024 * 1024)

    assert kept[0].exists()


# --- syncfs coalescing --------------------------------------------------------


def test_persist_is_a_noop_without_the_mount(monkeypatch):
    monkeypatch.setattr(_wasm_cache.os.path, "isdir", lambda p: False)
    _wasm_cache.persist()  # must not raise off-WASM


# --- the bugs the first review found ------------------------------------------


def test_a_file_larger_than_the_budget_does_not_empty_the_cache(tmp_path, monkeypatch):
    """Evicting everything would still not make it fit, so the cache is left
    alone and the file is kept for this session only."""
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "1")
    kept = _seed(tmp_path, monkeypatch, [512 * 1024])

    _wasm_cache._make_room(4 * 1024 * 1024)

    assert kept[0].exists()


def test_recency_is_tracked_without_touching_the_cached_file(tmp_path, monkeypatch):
    """Bumping a 107 MB file's mtime would make the next reconcile copy all
    107 MB into IndexedDB again, on every cache *hit*."""
    monkeypatch.setattr(_wasm_cache, "CACHE_MOUNT", str(tmp_path))
    monkeypatch.setattr(_wasm_cache, "_lru", {})
    big = tmp_path / "big.nii.gz"
    big.write_bytes(b"\0" * 2048)
    before = big.stat().st_mtime

    _wasm_cache._note_use(str(big))

    assert big.stat().st_mtime == before
    assert "big.nii.gz" in _wasm_cache._lru


def test_eviction_order_follows_the_sidecar_not_mtime(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "1")
    budget = 1024 * 1024
    first, second = _seed(tmp_path, monkeypatch, [budget // 2, budget // 2])
    # `first` is older on disk but was used most recently.
    monkeypatch.setattr(_wasm_cache, "_lru", {first.name: 9e9, second.name: 1.0})

    _wasm_cache._make_room(budget // 2)

    assert first.exists(), "the recently used file should have survived"
    assert not second.exists()


def test_the_lru_sidecar_is_never_itself_evicted(tmp_path, monkeypatch):
    monkeypatch.setattr(_wasm_cache, "cache_root", lambda: str(tmp_path))
    (tmp_path / _wasm_cache.LRU_FILE).write_text("{}")
    (tmp_path / "data.bin").write_bytes(b"\0" * 16)

    assert [os.path.basename(p) for _u, _s, p in _wasm_cache._cached_files()] == ["data.bin"]


def test_a_failed_sync_is_recorded_then_reported_from_a_cell(monkeypatch):
    """A QuotaExceededError is the case the budget exists to prevent. The
    callback runs on the event loop, outside any cell, where a warning reaches
    the browser console and not the reader -- so it is recorded there and
    raised from the next `download`, which does run in a cell."""
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)
    monkeypatch.setattr(_wasm_cache, "_last_sync_error", None)

    _wasm_cache._on_synced("QuotaExceededError")
    assert _wasm_cache.last_sync_error() == "QuotaExceededError"
    assert _wasm_cache._sync_running is False

    with pytest.warns(UserWarning, match="could not persist"):
        _wasm_cache._report_sync_error()

    assert _wasm_cache.last_sync_error() is None, "must not warn on every later call"


def test_a_successful_sync_records_no_error(monkeypatch):
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)
    _wasm_cache._on_synced(None)
    assert _wasm_cache._sync_running is False


def test_url_components_are_escaped(monkeypatch):
    """An unescaped `#` or space silently requests a different path than the
    one hf_hub_download would."""
    seen = {}

    monkeypatch.setattr(_wasm_cache, "cache_root", lambda: "/nowhere")
    monkeypatch.setattr(_wasm_cache, "_note_use", lambda p: None)
    monkeypatch.setattr(_wasm_cache, "_make_room", lambda n: None)
    monkeypatch.setattr(_wasm_cache, "_write", lambda d, b: None)
    monkeypatch.setattr(_wasm_cache, "_fetch", lambda url: seen.update(url=url) or _FakeBuf(10))

    _wasm_cache.download("dartbrains/localizer", "sub-01/a file #2.nii.gz")

    assert "%20" in seen["url"] and "%232" in seen["url"]
    assert "/dartbrains/localizer/resolve/main/sub-01/" in seen["url"]


class _FakeBuf:
    def __init__(self, length):
        self.length = length


def test_mount_failure_degrades_to_a_session_cache(monkeypatch):
    """No IndexedDB is a reason to lose persistence, not to fail the notebook."""
    monkeypatch.setattr(_wasm_cache, "_mount_state", "unavailable")
    assert _wasm_cache.cache_root() == _wasm_cache.SESSION_ROOT
    _wasm_cache.persist()  # must not raise


def test_jsnull_from_a_successful_syncfs_is_not_an_error():
    """Emscripten passes JS `null` on success and Pyodide surfaces it as
    `jsnull` -- falsy but not None. Verified against Pyodide v314 in Chrome:
    treating it as an error warned on every successful flush."""

    class _JsNull:
        def __bool__(self):
            return False

        def __repr__(self):
            return "jsnull"

    fake_null = _JsNull()
    assert not _wasm_cache._sync_failed(None)
    assert not _wasm_cache._sync_failed(fake_null)
    assert _wasm_cache._sync_failed("QuotaExceededError")


def test_a_successful_sync_records_nothing_to_report(monkeypatch):
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)
    monkeypatch.setattr(_wasm_cache, "_last_sync_error", None)
    _wasm_cache._on_synced(None)
    assert _wasm_cache.last_sync_error() is None


# --- the bugs the second review found -----------------------------------------


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    """Both roots on disk, with the mount reported as ready."""
    mount, session = tmp_path / "mount", tmp_path / "session"
    mount.mkdir()
    session.mkdir()
    monkeypatch.setattr(_wasm_cache, "CACHE_MOUNT", str(mount))
    monkeypatch.setattr(_wasm_cache, "SESSION_ROOT", str(session))
    monkeypatch.setattr(_wasm_cache, "_mount_state", "ready")
    monkeypatch.setattr(_wasm_cache, "_ensure_mount", lambda: True)
    monkeypatch.setattr(_wasm_cache, "_lru", {})
    return mount, session


def _stub_download(monkeypatch, size, fetches):
    def fetch(url):
        fetches.append(url)
        return _FakeBuf(size)

    def write(dest, buf):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(b"\0" * buf.length)

    monkeypatch.setattr(_wasm_cache, "_fetch", fetch)
    monkeypatch.setattr(_wasm_cache, "_write", write)
    monkeypatch.setattr(_wasm_cache, "persist", lambda: None)


def test_an_over_budget_file_is_not_re_downloaded_every_call(cache_dirs, monkeypatch):
    """It is written outside the mount, so the hit check has to look there too
    -- otherwise every cell calling get_file re-fetches the whole thing."""
    monkeypatch.setenv("DARTBRAINS_WASM_CACHE_MB", "1")
    fetches = []
    _stub_download(monkeypatch, 4 * 1024 * 1024, fetches)

    with pytest.warns(UserWarning, match="exceeds the"):
        first = _wasm_cache.download("repo/x", "huge.nii.gz")
    second = _wasm_cache.download("repo/x", "huge.nii.gz")

    assert first == second
    assert len(fetches) == 1, "the second call must hit the file already on disk"


def test_nothing_is_written_into_the_mount_while_it_populates(cache_dirs, monkeypatch):
    """A populate is a reconcile in the other direction: it deletes local
    entries IndexedDB lacks. A file written mid-populate would vanish under the
    cell about to read it."""
    mount, session = cache_dirs
    monkeypatch.setattr(_wasm_cache, "_mount_state", "populating")
    fetches = []
    _stub_download(monkeypatch, 1024, fetches)

    dest = _wasm_cache.download("repo/x", "a.nii.gz")

    assert dest.startswith(str(session))
    assert not (mount / "repo/x/a.nii.gz").exists()


def test_a_file_cached_during_populate_is_found_afterwards(cache_dirs, monkeypatch):
    mount, session = cache_dirs
    monkeypatch.setattr(_wasm_cache, "_mount_state", "populating")
    fetches = []
    _stub_download(monkeypatch, 1024, fetches)
    _wasm_cache.download("repo/x", "a.nii.gz")

    monkeypatch.setattr(_wasm_cache, "_mount_state", "ready")
    again = _wasm_cache.download("repo/x", "a.nii.gz")

    assert again.startswith(str(session))
    assert len(fetches) == 1


def test_persist_does_not_dispatch_alongside_an_in_flight_populate(monkeypatch):
    """Two syncfs calls outstanding on one mount reconcile against each other's
    half-applied state; the store-sync can empty the persisted cache."""
    dispatched = []
    monkeypatch.setattr(_wasm_cache, "_ensure_mount", lambda: True)
    monkeypatch.setattr(
        _wasm_cache,
        "_idbfs_syncfs",
        lambda *, populate, on_done: dispatched.append(populate) or True,
    )
    # The state _ensure_mount leaves behind while the populate is in flight.
    monkeypatch.setattr(_wasm_cache, "_mount_state", "populating")
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)

    _wasm_cache.persist()

    assert dispatched == [], "persist must not dispatch alongside the populate"
    assert _wasm_cache._sync_queued is True, "and must not silently drop the flush"


def test_a_queued_persist_survives_the_populate(monkeypatch):
    calls = []
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", True)
    monkeypatch.setattr(_wasm_cache, "_lru", {})
    monkeypatch.setattr(_wasm_cache, "_load_lru", lambda: None)
    monkeypatch.setattr(_wasm_cache, "persist", lambda: calls.append("persist"))

    _wasm_cache._on_populated(None)

    assert calls == ["persist"], "the deferred flush must not be dropped"
    assert _wasm_cache._sync_running is False


def test_a_cache_hit_does_not_flush(cache_dirs, monkeypatch):
    """Flushing reconciles the whole mount against this kernel's boot snapshot,
    so flushing on a read lets a stale tab delete a newer tab's cached file."""
    mount, _session = cache_dirs
    target = mount / "repo/x/a.nii.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0" * 16)
    flushes = []
    monkeypatch.setattr(_wasm_cache, "persist", lambda: flushes.append(1))

    _wasm_cache.download("repo/x", "a.nii.gz")

    assert flushes == []


def test_saving_recency_merges_rather_than_clobbers(cache_dirs, monkeypatch):
    """Another tab may have recorded uses this kernel never saw; dropping them
    sends eviction back to the mtime ordering the sidecar replaces."""
    mount, _session = cache_dirs
    (mount / _wasm_cache.LRU_FILE).write_text('{"other-tab.nii.gz": 111.0}')
    monkeypatch.setattr(_wasm_cache, "_lru", {"mine.nii.gz": 222.0})

    _wasm_cache._save_lru()

    on_disk = json.loads((mount / _wasm_cache.LRU_FILE).read_text())
    assert on_disk == {"other-tab.nii.gz": 111.0, "mine.nii.gz": 222.0}


def test_a_corrupt_sidecar_is_ignored_not_fatal(cache_dirs):
    mount, _session = cache_dirs
    (mount / _wasm_cache.LRU_FILE).write_text("not json at all")
    assert _wasm_cache._read_lru() == {}
