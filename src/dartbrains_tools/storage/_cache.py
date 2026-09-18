"""``@storage.cache``: memoise an expensive step into durable storage.

molab's disk does not survive teardown and every chapter is its own sandbox,
so the only cache that is warm on a student's *second* run of anything lives
in the bucket. Lookup order:

1. the local disk (fast; scratch on molab)
2. ``/cache/shared`` -- results the instructor pre-computed with the same code
3. ``/cache/private`` -- the student's own earlier results
4. run the function, then write the result to 1 and 3

The key covers the function's qualified name, its source text, and its
arguments, so editing the function or calling it differently is a miss.
Results are pickled; keep them to arrays, DataFrames, dicts and the like.
Unpickling is limited to the two tiers above: the student's own private cache
and the shared one only the instructor can write -- never another student's.

Do not wrap a cell that draws a figure: like ``mo.persistent_cache`` this
memoises the *return value* only, and matplotlib side effects will not
replay on a hit.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import pickle

from . import _state
from ._fs import NotFound

_VERSION = "1"  # bump to invalidate every cached result


def _key(func, args, kwargs) -> str:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        src = ""
    try:
        payload = pickle.dumps((args, sorted(kwargs.items())), protocol=4)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"@storage.cache arguments must be picklable: {e}") from e
    h = hashlib.sha256()
    for part in (_VERSION, func.__module__ or "", func.__qualname__, src):
        h.update(part.encode())
        h.update(b"\0")
    h.update(payload)
    return h.hexdigest()[:32]


def cache(func=None, *, shared: bool = True):
    """Decorator. ``shared=False`` skips the instructor-warmed tier."""

    def wrap(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            key = _key(f, args, kwargs)
            rel = f"{f.__name__}/{key}.pkl"
            session = _state.session()
            local = session.cache_root / "derived" / rel
            if local.is_file():
                return pickle.loads(local.read_bytes())
            tiers = ["/cache/shared", "/cache/private"] if shared else ["/cache/private"]
            for logical in tiers:
                if logical not in session.logicals():
                    continue
                try:
                    data = session.mount(logical).get_bytes(rel)
                except NotFound:
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)
                return pickle.loads(data)
            result = f(*args, **kwargs)
            data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            if "/cache/private" in session.logicals():
                session.mount("/cache/private").put_bytes(rel, data)
            return result

        inner.cache_key = lambda *a, **k: _key(f, a, k)  # type: ignore[attr-defined]
        return inner

    return wrap(func) if func is not None else wrap
