"""Shared HuggingFace Hub download helper used by all per-dataset modules."""

from . import _wasm_cache


def download(repo_id: str, filename: str) -> str:
    """Download a file from a HuggingFace dataset repo; return local cached path.

    In the browser this goes through :mod:`._wasm_cache`, which keeps the file
    in marimo's persistent filesystem so it is downloaded once per student
    rather than once per chapter per visit.
    """
    if _wasm_cache.active():
        return _wasm_cache.download(repo_id, filename)

    # Imported here, not at module scope: huggingface_hub imports `termios`,
    # which Pyodide does not ship, so a top-level import would make every
    # dataset module unimportable in WASM -- including the branch above that
    # exists precisely to avoid needing it.
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
