#!/usr/bin/env python3
"""Validate expected STOP pipeline artifact files."""

from __future__ import annotations

import argparse
from pathlib import Path


ARTIFACT_COLUMNS = {
    "selected": {"question", "target", "best_seed_response", "think", "content"},
    "segmented": {"id", "paragraphs", "question", "target"},
    "taxonomy": {"id", "taxonomy"},
    "conclusion": {"id", "conclusion"},
    "training": {"messages"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact spec in TYPE=PATH format. Known types: selected, segmented, taxonomy, conclusion, training.",
    )
    return parser.parse_args()


def parse_artifact(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected TYPE=PATH format, got: {spec}")
    kind, path = spec.split("=", 1)
    if kind not in ARTIFACT_COLUMNS:
        raise ValueError(f"Unknown artifact type: {kind}")
    return kind, Path(path)


def main() -> int:
    args = parse_args()

    import pandas as pd

    failures = []

    for spec in args.artifact:
        kind, path = parse_artifact(spec)
        if not path.exists():
            failures.append(f"{kind}: missing file {path}")
            continue

        df = pd.read_json(path, lines=True)
        if df.empty:
            failures.append(f"{kind}: empty file {path}")
            continue

        missing = ARTIFACT_COLUMNS[kind] - set(df.columns)
        if missing:
            failures.append(f"{kind}: missing columns {sorted(missing)} in {path}")
            continue

        print(f"{kind}: {len(df)} rows OK ({path})")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
