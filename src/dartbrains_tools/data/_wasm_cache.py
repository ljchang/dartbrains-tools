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

So in the browser we skip huggingface_hub entirely and download straight from
the Hub's ``resolve`` URL into an IndexedDB-backed filesystem, which outlives
the kernel, the page and the tab and is shared by every notebook on the origin.

Measured in Chrome against the file the ICA chapter loads (``sub-S01``
preprocessed bold, 107 MB)::

    cold download + write         ~3-8 s   (network-bound)
    persist (syncfs)               0.11 s
    populate at every later boot   0.06 s
    warm hit, another chapter      0.001 s

Three details are load-bearing.

**The cache gets its own mount.** ``syncfs`` is a *reconcile*, not an append:
IDBFS diffs the kernel's in-memory view of a mount against IndexedDB and
**deletes entries that are missing locally**. A kernel holds the snapshot it
booted with, so flushing a stale one erases anything saved since. Writing the
cache into marimo's own ``/marimo`` mount therefore meant that caching a
dataset in one tab could delete a notebook another tab had just saved --
reproduced, and exactly the work a student cannot get back. The cache lives in
its own mount at :data:`CACHE_MOUNT` and is flushed through
``IDBFS.syncfs(mount, ...)`` rather than the global ``FS.syncfs``, so our
writes can never reconcile the notebook tree. The same staleness still applies
*within* our mount, where the worst case is that someone re-downloads a file --
so flushes are kept to writes, never plain cache hits, and new files stay out
of the mount until its own populate (a reconcile in the other direction) has
landed.

**The payload never becomes a Python ``bytes``.** The response arrives as a JS
``ArrayBuffer`` and goes straight to ``FS.writeFile``. Reading it through
Python costs ~320 MB of WASM heap for a 107 MB file, and Pyodide's heap never
shrinks back -- expensive in a kernel that must then hold a float64 expansion
of the same image.

**The cache is budgeted.** ``navigator.storage.persisted()`` is false for the
site, so the browser may evict the origin's storage under pressure -- and that
storage also holds the student's edited notebooks. The budget (4 GB, override
with ``DARTBRAINS_WASM_CACHE_MB``) keeps datasets from crowding out work that
cannot be re-downloaded, evicting least-recently-used files to stay under it.
It is deliberately well below the ~10.7 GB a Chrome origin is offered and can
be raised as the heavier chapters (Sherlock, Paranoia) come into the browser;
watch ``navigator.storage.estimate()`` when doing so. Set
``DARTBRAINS_WASM_CACHE=0`` to disable caching entirely.

Known limitation: the download uses a synchronous ``XMLHttpRequest``, because
notebook code is synchronous and cannot await. A synchronous request cannot
carry a timeout, so a stalled transfer blocks the kernel until the browser
gives up on the socket, and marimo's interrupt cannot reach it either.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from urllib.parse import quote

try:  # Pyodide >= 0.28 surfaces JS ``null`` as ``jsnull``, not ``None``
    from pyodide.ffi import jsnull as _JS_NULL
except ImportError:  # pragma: no cover - not running under Pyodide
    _JS_NULL = None

#: The cache's own IDBFS mount. Deliberately *not* ``/marimo`` -- see above.
CACHE_MOUNT = "/dartbrains-cache"

#: Where a file goes when it is too large to cache: kernel-local, never synced.
SESSION_ROOT = "/tmp/dartbrains-data"

#: Tracks last use without touching the cached files themselves. Touching a
#: 107 MB file's mtime would make the next reconcile copy all 107 MB into
#: IndexedDB again, so recency lives in this sidecar instead.
LRU_FILE = ".last-used.json"

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
_mount_state = "absent"  # absent | populating | ready | unavailable
_sync_running = False  # IDBFS syncfs is not safe to run concurrently with itself
_sync_queued = False
_last_sync_error: str | None = None
_error_reported = False
_lru: dict[str, float] = {}


def active() -> bool:
    """Whether downloads should go through this cache rather than the Hub client."""
    return sys.platform == "emscripten" and os.environ.get("DARTBRAINS_WASM_CACHE") != "0"


def cache_root() -> str:
    """Where cached files are kept: the persistent mount when there is one.

    Without a working IDBFS mount (bare Pyodide, a browser refusing IndexedDB)
    the cache degrades to kernel-local memory: still useful within one session,
    just not across them. Degrading beats failing.
    """
    return CACHE_MOUNT if _ensure_mount() else SESSION_ROOT


def download(repo_id: str, filename: str) -> str:
    """Return a local path to ``filename`` from a Hub dataset repo, caching it.

    A hit costs nothing; a miss downloads the file, makes room for it if the
    budget requires, and schedules the write to IndexedDB.
    """
    _report_sync_error()

    hit = _find_existing(repo_id, filename)
    if hit is not None:
        hit = _promote(hit, repo_id, filename)
        _note_use(hit)
        return hit

    # `quote` so a filename carrying `#`, `?` or a space addresses the path it
    # names rather than a truncated one -- hf_hub_download escapes these too.
    url = RESOLVE_URL.format(repo_id=quote(repo_id, safe="/"), filename=quote(filename, safe="/"))
    buf = _fetch(url)
    size = buf.length

    root = _writable_root()
    budget = _budget_bytes()
    if root == CACHE_MOUNT and budget and size > budget:
        # Evicting the whole cache still would not make this fit. Keep it for
        # this kernel only rather than clearing everything else for nothing.
        warnings.warn(
            f"{filename} ({size // 1048576} MB) exceeds the "
            f"{budget // 1048576} MB browser cache budget; keeping it for this "
            f"session only. Raise DARTBRAINS_WASM_CACHE_MB to cache it.",
            stacklevel=2,
        )
        root = SESSION_ROOT
    elif root == CACHE_MOUNT:
        _make_room(size)

    dest = os.path.join(root, repo_id, filename)
    _write(dest, buf)
    if root == CACHE_MOUNT:
        _note_use(dest)
        _save_lru()
        persist()
    return dest


def persist() -> None:
    """Flush the cache mount to IndexedDB, coalescing overlapping requests.

    ``syncfs`` is asynchronous and we are called from synchronous notebook code,
    so this fires and forgets: the copy runs on the event loop once the cell
    returns. Two ``syncfs`` calls must never be outstanding on the same mount at
    once -- they would reconcile against each other's half-applied state -- so a
    request arriving while one is in flight (including the populate at mount
    time) is deferred to a single follow-up sync.
    """
    global _sync_running, _sync_queued

    if not _ensure_mount():
        return
    if _sync_running:
        _sync_queued = True
        return

    _sync_running = True
    if not _idbfs_syncfs(populate=False, on_done=_on_synced):
        _sync_running = False


def last_sync_error() -> str | None:
    """The most recent ``syncfs`` failure, or ``None``. Exposed for diagnosis."""
    return _last_sync_error


# --- the IDBFS mount ----------------------------------------------------------


def _ensure_mount() -> bool:
    """Mount the cache's own IDBFS once, and start populating it.

    The populate is asynchronous and cannot be awaited from synchronous notebook
    code, which is why mounting happens at import: marimo runs a notebook's
    imports in an earlier cell than its data loading, and the event loop turns
    in between. But that is an assumption about marimo's runner, not something
    this module can enforce -- so until the populate lands, :func:`_writable_root`
    keeps new files out of this mount. A populate is a reconcile in the other
    direction: it deletes local entries that IndexedDB does not have, which
    would otherwise take a file that had just been downloaded and leave a later
    cell reading a path that no longer exists.
    """
    global _mount_state, _sync_running

    if _mount_state in ("ready", "populating"):
        return True
    if _mount_state == "unavailable" or not active():
        return False

    try:
        import js
        import pyodide_js
        from pyodide.ffi import to_js

        fs = pyodide_js.FS
        os.makedirs(CACHE_MOUNT, exist_ok=True)
        try:
            fs.mount(
                fs.filesystems.IDBFS,
                to_js({"root": "."}, dict_converter=js.Object.fromEntries),
                CACHE_MOUNT,
            )
        except Exception:
            # Most likely EBUSY: something already mounted here (a reimport
            # after `sys.modules` was cleared). An existing mount is a working
            # cache, not a reason to fall back to kernel memory and leave a
            # multi-gigabyte cache invisible -- but only if it really is our
            # IDBFS mount, and we still have to populate, because this module's
            # globals are fresh while the mount may not be.
            if not _is_mounted():
                raise
    except Exception:  # noqa: BLE001 - no IndexedDB at all
        _mount_state = "unavailable"
        return False

    _mount_state = "populating"
    _sync_running = True  # the populate occupies the mount like any other sync
    if not _idbfs_syncfs(populate=True, on_done=_on_populated):
        _sync_running = False
        _mount_state = "ready"  # nothing to populate from; writes still work
        _load_lru()
    return True


def _is_mounted() -> bool:
    """Whether :data:`CACHE_MOUNT` is *our own* IDBFS mount.

    Not "has a mount": in Emscripten every node carries one, inherited from its
    parent, so a plain MEMFS directory -- which ``os.makedirs`` has just created
    -- reports the root mount and would make this trivially true. That turned
    every mount failure (IndexedDB blocked, a private window refusing storage)
    into a false ``ready``, with the session-cache fallback unreachable.
    """
    try:
        import pyodide_js

        fs = pyodide_js.FS
        mount = fs.lookupPath(CACHE_MOUNT).node.mount
        return mount is not None and mount.mountpoint == CACHE_MOUNT
    except Exception:  # noqa: BLE001
        return False


def _idbfs_syncfs(*, populate: bool, on_done) -> bool:
    """Sync *only* the cache mount. Returns whether the call was dispatched."""
    try:
        import pyodide_js
        from pyodide.ffi import create_once_callable

        fs = pyodide_js.FS
        mount = fs.lookupPath(CACHE_MOUNT).node.mount
        fs.filesystems.IDBFS.syncfs(mount, populate, create_once_callable(on_done))
    except Exception:  # noqa: BLE001 - a failed flush costs a re-download, not data
        return False
    return True


def _sync_failed(err) -> bool:
    """Whether a ``syncfs`` callback argument represents a real failure.

    Emscripten passes JS ``null`` on success. Pyodide surfaces that as
    ``jsnull`` -- falsy, but *not* ``None`` -- so a plain ``err is not None``
    check reports every successful flush as an error. Verified in Chrome
    against Pyodide v314: ``v is jsnull`` is true and ``bool(v)`` is false.
    """
    return err is not None and err is not _JS_NULL and bool(err)


def _on_populated(err=None) -> None:
    """Settle the mount once its populate lands.

    A *failed* populate must not make the mount writable. Local would be empty
    while IndexedDB still holds every cached dataset, so the next write's flush
    would reconcile that emptiness outward and delete the whole persisted
    cache -- other tabs' files included. Unwritable keeps new files in
    :data:`SESSION_ROOT` and leaves the stored cache alone.
    """
    global _mount_state, _sync_running, _sync_queued, _last_sync_error

    _sync_running = False
    if _sync_failed(err):
        _last_sync_error = str(err)
        _mount_state = "unavailable"
        _sync_queued = False  # nothing may flush against an unpopulated view
        return

    _mount_state = "ready"
    _load_lru()
    if _sync_queued:
        _sync_queued = False
        persist()


def _on_synced(err=None) -> None:
    global _sync_running, _sync_queued, _last_sync_error, _error_reported

    _sync_running = False
    if _sync_failed(err):
        # Most likely QuotaExceededError -- the case the budget exists to avoid.
        # Recorded rather than warned from here: this runs on the event loop,
        # outside any cell, where a warning reaches the browser console and not
        # the reader. `_report_sync_error` raises it from the next `download`,
        # which does run in a cell.
        _last_sync_error = str(err)
        _error_reported = False
    else:
        # A later flush succeeding supersedes an earlier failure; keeping it
        # would warn about a condition that no longer holds.
        _last_sync_error = None
    if _sync_queued:
        _sync_queued = False
        persist()


def _report_sync_error() -> None:
    """Surface a failed flush from cell context, where the reader can see it.

    Reported once per failure rather than consumed: ``last_sync_error()`` is
    documented for diagnosis, and popping the value here left it returning
    ``None`` after any download.
    """
    global _error_reported

    if _last_sync_error is None or _error_reported:
        return
    _error_reported = True
    warnings.warn(
        f"could not persist the dataset cache to IndexedDB ({_last_sync_error}); "
        f"data will be downloaded again next visit",
        stacklevel=3,
    )


# --- choosing a location ------------------------------------------------------


def _writable_root() -> str:
    """Where a *new* file may be written.

    Only a fully populated mount may take writes. While the populate is in
    flight, its reconcile would delete anything written in the meantime, so new
    files go to kernel-local storage instead -- a slower next visit rather than
    a file vanishing under a cell that is about to read it.
    """
    if _ensure_mount() and _mount_state == "ready":
        return CACHE_MOUNT
    return SESSION_ROOT


def _promote(path: str, repo_id: str, filename: str) -> str:
    """Move a session-only copy into the cache once the mount can take it.

    A file downloaded while the mount was still populating lands in
    :data:`SESSION_ROOT`, and nothing else would ever move it: the kernel would
    keep hitting the session copy and re-download it on the next visit, forever
    and silently. The bytes travel as a JS array, so a 107 MB promotion costs
    no WASM heap.
    """
    if not path.startswith(SESSION_ROOT) or _writable_root() != CACHE_MOUNT:
        return path

    dest = os.path.join(CACHE_MOUNT, repo_id, filename)
    try:
        import pyodide_js

        size = os.path.getsize(path)
        budget = _budget_bytes()
        if budget and size > budget:
            return path
        _make_room(size)
        _write(dest, pyodide_js.FS.readFile(path))
        os.remove(path)
    except Exception:  # noqa: BLE001 - the session copy is still perfectly usable
        return path

    _note_use(dest)
    _save_lru()
    persist()
    return dest


def _find_existing(repo_id: str, filename: str) -> str | None:
    """A cached copy in either root, or ``None``.

    Both are checked because a file can legitimately live in either: too large
    for the budget, or downloaded while the mount was still populating.
    """
    for root in (CACHE_MOUNT, SESSION_ROOT):
        candidate = os.path.join(root, repo_id, filename)
        if os.path.exists(candidate):
            return candidate
    return None


# --- reading and writing ------------------------------------------------------


def _fetch(url: str):
    """Fetch ``url`` synchronously; returns a JS ``Uint8Array``, not Python bytes."""
    global _fetch_binary
    if _fetch_binary is None:
        import js

        _fetch_binary = js.eval(_JS_SYNC_FETCH)
    return _fetch_binary(url)


def _write(dest: str, buf) -> None:
    """Write ``buf`` to ``dest`` atomically.

    ``FS.writeFile`` truncates before it writes, so a failure part-way through
    (MEMFS raises on a large allocation) would leave a short file that every
    later run treats as a valid hit. Writing beside the target and renaming
    means a reader sees either the old file or the whole new one.
    """
    import pyodide_js

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    try:
        pyodide_js.FS.writeFile(part, buf)
        os.replace(part, dest)
    except Exception:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


# --- recency and the eviction budget ------------------------------------------


def _lru_path() -> str:
    return os.path.join(CACHE_MOUNT, LRU_FILE)


def _read_lru() -> dict[str, float]:
    try:
        with open(_lru_path()) as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    out = {}
    for key, value in loaded.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_lru() -> None:
    global _lru
    _lru = _read_lru()


def _note_use(dest: str) -> None:
    """Record in memory that ``dest`` was used.

    Recency is kept in a small sidecar rather than the file's mtime: IDBFS
    decides what to copy by comparing timestamps, so touching a 107 MB file
    would rewrite all 107 MB into IndexedDB on every cache *hit*.

    Nothing is flushed here. A flush reconciles the whole mount against the
    snapshot this kernel booted with, so flushing on a plain cache hit would let
    a stale tab delete a file a newer tab had cached. Recency only decides what
    to evict, which happens when writing -- so the sidecar is saved then.

    The cost is accepted deliberately: a kernel that only ever *hits* the cache
    writes no recency to disk, so across sessions eviction order decays toward
    "oldest download first" rather than true least-recently-used. That makes
    eviction pick a slightly worse victim among files that are all
    re-downloadable. Flushing on reads to avoid it would instead let one stale
    tab delete another's cached data, which is a real loss for the reader, so
    the imprecise ordering is the better trade.
    """
    if not dest.startswith(CACHE_MOUNT):
        return
    _lru[os.path.relpath(dest, CACHE_MOUNT)] = time.time()


def _save_lru() -> None:
    """Merge our recency into the sidecar on disk and write it back.

    Merged rather than overwritten: another tab may have recorded uses this
    kernel never saw, and clobbering them would send eviction back to the mtime
    ordering the sidecar exists to replace.
    """
    merged = _read_lru()
    merged.update(_lru)
    try:
        with open(_lru_path(), "w") as fh:
            json.dump(merged, fh)
    except OSError:
        return
    _lru.update(merged)


def _budget_bytes() -> int:
    try:
        mb = int(os.environ.get("DARTBRAINS_WASM_CACHE_MB", DEFAULT_BUDGET_MB))
    except ValueError:
        mb = DEFAULT_BUDGET_MB
    return max(mb, 0) * 1024 * 1024


def _cached_files() -> list[tuple[float, int, str]]:
    """``(last_used, size, path)`` for every cached file, least recent first."""
    root = cache_root()
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == LRU_FILE:
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = os.path.relpath(path, root)
            found.append((_lru.get(key, st.st_mtime), st.st_size, path))
    found.sort()
    return found


def _make_room(incoming: int) -> None:
    """Evict least-recently-used files until ``incoming`` bytes fit in the budget.

    A file bigger than the whole budget is the caller's problem: evicting
    everything would still not make it fit, so this leaves the cache alone.
    """
    budget = _budget_bytes()
    if budget <= 0 or incoming > budget:
        return
    root = cache_root()
    files = _cached_files()
    total = sum(size for _used, size, _path in files)
    for _used, size, path in files:
        if total + incoming <= budget:
            return
        try:
            os.remove(path)
            total -= size
            _lru.pop(os.path.relpath(path, root), None)
        except OSError:
            continue


if active():  # pragma: no cover - browser only
    _ensure_mount()
