"""Persistent dataset cache for notebooks running in the browser (Pyodide/WASM).

Off-WASM this module is inert: :mod:`._hub` calls ``hf_hub_download`` and
huggingface_hub's own cache handles everything. In the browser neither half of
that arrangement survives:

``huggingface_hub`` cannot be imported at all
    ``huggingface_hub.utils._terminal`` imports ``termios``, a Unix-TTY stdlib
    module Pyodide does not ship, so ``import huggingface_hub`` raises
    ``ModuleNotFoundError`` before any download is attempted.
its cache would not outlive the tab
    Pyodide's default filesystem is MEMFS -- per-kernel memory. Every chapter a
    student opened would download the data again, and so would the same chapter
    after a reload.

marimo's WASM runtime mounts IDBFS -- an IndexedDB-backed filesystem -- at
``/marimo``, populating it at kernel boot and persisting it on save. Anything
written there outlives the kernel, the page and the tab, and is visible to
*every* notebook on the origin. So in the browser we skip huggingface_hub
entirely, download straight from the Hub's ``resolve`` URL into
``/marimo/.dartbrains-data``, and ask IDBFS to persist.

Measured in Chrome against the file the ICA chapter loads (``sub-S01``
preprocessed bold, 107 MB)::

    cold download + write         ~5-8 s   (network-bound)
    persist (syncfs)               0.11 s
    populate at every later boot   0.06 s
    warm open (cache hit)            0 s

Two details are load-bearing:

the payload never becomes a Python ``bytes``
    The response arrives as a JS ``ArrayBuffer`` and goes straight to
    ``FS.writeFile``. Reading it through Python costs ~320 MB of WASM heap for
    a 107 MB file, and Pyodide's heap never shrinks back -- expensive in a
    kernel that must then hold a float64 expansion of the same image.
the cache is budgeted
    ``navigator.storage.persisted()`` is false for the site, so the browser may
    evict the origin's storage under pressure -- and that storage also holds
    the student's edited notebooks. The budget (4 GB, override with
    ``DARTBRAINS_WASM_CACHE_MB``) keeps datasets from crowding out work that
    cannot be re-downloaded, evicting least-recently-used files to stay under
    it. It is deliberately well below the ~10.7 GB a Chrome origin is offered
    and can be raised as the heavier chapters (Sherlock, Paranoia) come into
    the browser; watch ``navigator.storage.estimate()`` when doing so. Set
    ``DARTBRAINS_WASM_CACHE=0`` to disable caching entirely.
"""

from __future__ import annotations

import os
import sys

#: Where the cache lives inside marimo's IDBFS mount.
MOUNT = "/marimo"
CACHE_DIR = ".dartbrains-data"

#: Hub files are addressed by branch, not commit: a dataset revision replaces
#: what is cached under the same key, which is what a course wants.
RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

DEFAULT_BUDGET_MB = 4096

# One synchronous fetch that works on either side of the worker boundary.
# `responseType` is only legal on a synchronous XHR off the main thread (marimo
# runs its kernel in a worker); on the main thread it throws, and the
# x-user-defined charset trick gets the bytes through `responseText` intact.
# Either way the conversion happens in JS, so nothing lands in the WASM heap.
_JS_SYNC_FETCH = """
(function (url) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, false);
  try { xhr.responseType = "arraybuffer"; } catch (e) { /* main thread */ }
  if (xhr.responseType !== "arraybuffer") {
    xhr.overrideMimeType("text/plain; charset=x-user-defined");
  }
  xhr.send(null);
  if (xhr.status !== 200) {
    throw new Error("HTTP " + xhr.status + " for " + url);
  }
  if (xhr.response instanceof ArrayBuffer) return new Uint8Array(xhr.response);
  const text = xhr.responseText;
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
  return out;
})
"""

_fetch_binary = None  # the compiled JS function above
_sync_running = False  # IDBFS syncfs is not safe to run concurrently with itself
_sync_queued = False


def active() -> bool:
    """Whether downloads should go through this cache rather than the Hub client."""
    return sys.platform == "emscripten" and os.environ.get("DARTBRAINS_WASM_CACHE") != "0"


def cache_root() -> str:
    """The cache directory -- inside the persistent mount when there is one.

    Without ``/marimo`` (bare Pyodide, or a marimo build that stopped mounting
    IDBFS) the cache still works, it just lives in memory and dies with the
    kernel. Degrading is better than failing.
    """
    base = MOUNT if os.path.isdir(MOUNT) else "/tmp"
    return os.path.join(base, CACHE_DIR)


def download(repo_id: str, filename: str) -> str:
    """Return a local path to ``filename`` from a Hub dataset repo, caching it.

    A hit costs nothing; a miss downloads the file, makes room for it if the
    budget requires, and schedules the write to IndexedDB.
    """
    dest = os.path.join(cache_root(), repo_id, filename)
    if os.path.exists(dest):
        os.utime(dest, None)  # touch: pruning evicts least-recently-used first
        return dest

    buf = _fetch(RESOLVE_URL.format(repo_id=repo_id, filename=filename))
    _make_room(buf.length)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    import pyodide_js

    pyodide_js.FS.writeFile(dest, buf)
    persist()
    return dest


def persist() -> None:
    """Flush the mount to IndexedDB, coalescing overlapping requests.

    ``syncfs`` is asynchronous and we are called from synchronous notebook code,
    so this fires and forgets: the copy runs on the event loop once the cell
    returns. Overlapping calls would interleave their IndexedDB transactions, so
    a request arriving mid-flight is deferred to a single follow-up sync.
    """
    global _sync_running, _sync_queued

    if not os.path.isdir(MOUNT):
        return
    if _sync_running:
        _sync_queued = True
        return

    try:
        import pyodide_js
        from pyodide.ffi import create_once_callable
    except ImportError:  # pragma: no cover - not running under Pyodide
        return

    _sync_running = True
    try:
        pyodide_js.FS.syncfs(False, create_once_callable(_on_synced))
    except Exception:  # pragma: no cover - a failed flush costs a re-download, not data
        _sync_running = False


def _on_synced(err=None) -> None:
    global _sync_running, _sync_queued
    _sync_running = False
    if _sync_queued:
        _sync_queued = False
        persist()


def _fetch(url: str):
    """Fetch ``url`` synchronously; returns a JS ``Uint8Array``, not Python bytes."""
    global _fetch_binary
    if _fetch_binary is None:
        import js

        _fetch_binary = js.eval(_JS_SYNC_FETCH)
    return _fetch_binary(url)


def _budget_bytes() -> int:
    try:
        mb = int(os.environ.get("DARTBRAINS_WASM_CACHE_MB", DEFAULT_BUDGET_MB))
    except ValueError:
        mb = DEFAULT_BUDGET_MB
    return max(mb, 0) * 1024 * 1024


def _cached_files() -> list[tuple[float, int, str]]:
    """``(mtime, size, path)`` for every cached file, oldest use first."""
    root = cache_root()
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            found.append((st.st_mtime, st.st_size, path))
    found.sort()
    return found


def _make_room(incoming: int) -> None:
    """Evict least-recently-used files until ``incoming`` bytes fit in the budget."""
    budget = _budget_bytes()
    if budget <= 0:
        return
    files = _cached_files()
    total = sum(size for _mtime, size, _path in files)
    for _mtime, size, path in files:
        if total + incoming <= budget:
            return
        try:
            os.remove(path)
            total -= size
        except OSError:
            continue
