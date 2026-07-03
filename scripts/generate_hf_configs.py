#!/usr/bin/env python
"""Generate uniform path-index load_dataset configs for the dartbrains datasets.

Usage:
    python scripts/generate_hf_configs.py index --repo dartbrains/localizer --dry-run
    python scripts/generate_hf_configs.py index --repo dartbrains/localizer
    python scripts/generate_hf_configs.py check --repo dartbrains/localizer
"""

from __future__ import annotations

import argparse
import sys

from hf_configs import hub
from hf_configs.index import (
    build_index,
    render_readme_configs,
    replace_configs_block,
    rows_to_csv,
)
from hf_configs.specs import DATASETS


def _build_outputs(repo: str) -> dict[str, str]:
    """Return {repo_path: text} for every generated file (CSVs + participants)."""
    spec = DATASETS[repo]
    files = hub.list_files(repo)
    out: dict[str, str] = {}
    for name, cfg in spec["configs"].items():
        if "content" in cfg:
            raw = hub.read_text(repo, cfg["content"])
            out[cfg["content_out"]] = _tsv_to_csv(raw)
            continue
        rows = build_index(files, cfg)
        if not rows:
            print(f"  WARNING: config {name!r} matched 0 files", file=sys.stderr)
        out[f"{name}.csv"] = rows_to_csv(rows)
    return out


def _tsv_to_csv(text: str) -> str:
    import csv
    import io

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in reader:
        writer.writerow(row)
    return buf.getvalue()


def _rewrite_readme(repo: str, configs_yaml: str) -> str:
    """Replace the frontmatter `configs:` block in the repo README."""
    return replace_configs_block(hub.read_text(repo, "README.md"), configs_yaml)


def cmd_index(args):
    repo = args.repo
    outputs = _build_outputs(repo)
    outputs["README.md"] = _rewrite_readme(repo, render_readme_configs(DATASETS[repo]))
    if args.dry_run:
        for path, text in outputs.items():
            preview = text if path.endswith("README.md") else "\n".join(text.splitlines()[:4])
            print(f"\n===== {path} ({len(text.splitlines())} lines) =====")
            print(preview)
        return
    url = hub.upload_files(
        repo, outputs,
        message="Generate uniform path-index load_dataset configs",
    )
    print(f"Opened PR: {url}")


def cmd_check(args):
    from datasets import load_dataset

    repo = args.repo
    files = hub.list_files(repo)
    ok = True
    for name, cfg in DATASETS[repo]["configs"].items():
        if "content" in cfg:
            continue
        expected = len(build_index(files, cfg))
        got = len(load_dataset(repo, name, split="train"))
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {name:12s} expected={expected:4d} got={got:4d} {status}")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    pi = sub.add_parser("index", help="generate + (optionally) upload configs")
    pi.add_argument("--repo", required=True, choices=list(DATASETS))
    pi.add_argument("--dry-run", action="store_true")
    pi.set_defaults(func=cmd_index)
    pc = sub.add_parser("check", help="verify row counts via load_dataset")
    pc.add_argument("--repo", required=True, choices=list(DATASETS))
    pc.set_defaults(func=cmd_check)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
