"""Thin HuggingFace Hub I/O for the generator. Network only -- no logic."""

from __future__ import annotations

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

_api = HfApi()


def list_files(repo: str) -> list[str]:
    return list_repo_files(repo, repo_type="dataset")


def read_text(repo: str, path: str) -> str:
    local = hf_hub_download(repo, path, repo_type="dataset")
    with open(local, encoding="utf-8") as fh:
        return fh.read()


def upload_files(repo: str, files: dict[str, str], branch: str, message: str) -> str:
    """Upload each {repo_path: text} on *branch*, creating the branch/PR."""
    from huggingface_hub import CommitOperationAdd

    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=text.encode("utf-8"))
        for p, text in files.items()
    ]
    info = _api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=message,
        create_pr=True,
    )
    return info.pr_url or ""
