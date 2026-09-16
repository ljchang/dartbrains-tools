"""Tests for the browser-side dataset cache (data/_wasm_cache.py).

The download path itself needs Pyodide and is verified in a browser; what is
testable here is everything around it -- when the cache engages, where it puts
files, and the eviction policy that keeps datasets from crowding a student's
saved notebooks out of the origin's storage.
"""

from __future__ import annotations

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
    monkeypatch.setattr(_wasm_cache, "persist", lambda: None)
    monkeypatch.setattr(_wasm_cache, "_lru", {})
    big = tmp_path / "big.nii.gz"
    big.write_bytes(b"\0" * 2048)
    before = big.stat().st_mtime

    _wasm_cache._note_use(str(big))

    assert big.stat().st_mtime == before
    assert "big.nii.gz" in _wasm_cache._lru
    assert (tmp_path / _wasm_cache.LRU_FILE).exists()


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


def test_a_failed_sync_is_surfaced_not_swallowed(monkeypatch):
    """A QuotaExceededError is the case the budget exists to prevent; dropping
    it silently means re-downloading every visit with no signal."""
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)
    monkeypatch.setattr(_wasm_cache, "_last_sync_error", None)

    with pytest.warns(UserWarning, match="could not persist"):
        _wasm_cache._on_synced("QuotaExceededError")

    assert _wasm_cache.last_sync_error() == "QuotaExceededError"
    assert _wasm_cache._sync_running is False


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


def test_a_successful_sync_does_not_warn(monkeypatch, recwarn):
    monkeypatch.setattr(_wasm_cache, "_sync_running", True)
    monkeypatch.setattr(_wasm_cache, "_sync_queued", False)
    _wasm_cache._on_synced(None)
    assert not [w for w in recwarn if "could not persist" in str(w.message)]
