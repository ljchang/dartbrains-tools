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


def test_cache_root_prefers_the_persistent_mount(monkeypatch):
    monkeypatch.setattr(_wasm_cache.os.path, "isdir", lambda p: p == _wasm_cache.MOUNT)
    assert _wasm_cache.cache_root() == f"/marimo/{_wasm_cache.CACHE_DIR}"


def test_cache_root_falls_back_without_the_mount(monkeypatch):
    monkeypatch.setattr(_wasm_cache.os.path, "isdir", lambda p: False)
    assert _wasm_cache.cache_root().startswith("/tmp/")


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
