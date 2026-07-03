"""Build path-index rows and render the README `configs:` block (pure)."""

from __future__ import annotations

import io
import csv as _csv
import re

from .labels import parse_bids_entities


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a glob to a full-match regex.

    ``**`` matches any characters including ``/``; ``*`` matches any run of
    characters except ``/``; ``?`` matches a single non-``/`` character.
    """
    out = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def build_index(files: list[str], config: dict) -> list[dict]:
    """Select files matching ``config['glob']`` and attach label columns.

    Base labels come from :func:`parse_bids_entities`; if ``config['labels']``
    is provided it is called per path and its keys override/augment the base.
    Rows are ``{'path': f, **labels}`` sorted by path.
    """
    rx = glob_to_regex(config["glob"])
    labeler = config.get("labels")
    rows = []
    for f in sorted(files):
        if not rx.match(f):
            continue
        labels = parse_bids_entities(f)
        if labeler is not None:
            labels = {**labels, **labeler(f)}
        rows.append({"path": f, **labels})
    return rows


def rows_to_csv(rows: list[dict]) -> str:
    """Serialize rows to comma-CSV. Header = 'path' + union of label keys
    (first-seen order); missing cells are empty."""
    fields = ["path"]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def render_readme_configs(dataset: dict) -> str:
    """Render the README frontmatter `configs:` block for a dataset spec."""
    lines = ["configs:"]
    default = dataset.get("default")
    for name, cfg in dataset["configs"].items():
        lines.append(f"  - config_name: {name}")
        if name == default:
            lines.append("    default: true")
        target = cfg["content_out"] if "content" in cfg else f"{name}.csv"
        lines.append("    data_files:")
        lines.append("      - split: train")
        lines.append(f"        path: {target}")
    return "\n".join(lines) + "\n"
