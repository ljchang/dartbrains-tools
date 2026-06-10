"""Shared HuggingFace Hub download helper used by all per-dataset modules."""

from huggingface_hub import hf_hub_download


def download(repo_id: str, filename: str) -> str:
    """Download a file from a HuggingFace dataset repo; return local cached path."""
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
