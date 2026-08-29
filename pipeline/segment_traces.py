#!/usr/bin/env python3
"""Segment selected reasoning traces into numbered nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import PROJECT_ROOT, split_text_into_paragraphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "self_distill_best_of_N_seed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "pruned_data_pipeline"
        / "data"
        / "self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl",
    )
    parser.add_argument("--column", default="think", help="Text column to segment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd

    df = pd.read_json(args.input, lines=True)
    if args.column not in df.columns:
        raise SystemExit(f"ERROR: Missing column {args.column!r} in {args.input}")

    df = df.copy()
    df[args.column] = df[args.column].fillna("")
    df["paragraphs"] = df[args.column].apply(split_text_into_paragraphs)
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    required_cols = [
        "id",
        "paragraphs",
        "question",
        "target",
        "no_last_thinking_sentence",
        "last_thinking_sentence",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: Missing columns after segmentation: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[required_cols].to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(df)} segmented traces: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
