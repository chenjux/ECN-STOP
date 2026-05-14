#!/usr/bin/env python3
"""Collect Gemini batch results into normalized annotation JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import extract_gemini_text, extract_jsonl_block, parse_request_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input-dir", type=Path, help="Directory containing *_result.jsonl files.")
    sources.add_argument("--result-file", action="append", type=Path, help="Result JSONL file. Repeatable.")
    parser.add_argument("--glob", default="*_result.jsonl", help="Glob used with --input-dir.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["taxonomy", "conclusion"], required=True)
    parser.add_argument(
        "--join",
        type=Path,
        help="Optional JSONL file to left-join on id, usually the segmented traces or taxonomy file.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows whose extracted JSONL annotation is missing.",
    )
    return parser.parse_args()


def result_files(args: argparse.Namespace) -> list[Path]:
    if args.result_file:
        return sorted(args.result_file)
    return sorted(args.input_dir.glob(args.glob))


def main() -> int:
    args = parse_args()

    import pandas as pd

    files = result_files(args)
    if not files:
        raise SystemExit("ERROR: No result files found.")

    frames = [pd.read_json(path, lines=True) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "key" not in df.columns or "response" not in df.columns:
        raise SystemExit("ERROR: Result files must contain key and response columns.")

    annotation_col = "taxonomy" if args.kind == "taxonomy" else "conclusion"
    response_col = f"{annotation_col}_response"

    df["id"] = df["key"].apply(parse_request_id)
    df[response_col] = df["response"]
    df[annotation_col] = df["response"].apply(lambda response: extract_jsonl_block(extract_gemini_text(response)))
    df = df.drop(columns=["key", "response"])

    if args.drop_missing:
        df = df[df[annotation_col].notna()].copy()

    if args.join:
        join_df = pd.read_json(args.join, lines=True)
        if "id" not in join_df.columns:
            raise SystemExit(f"ERROR: Join file must contain id column: {args.join}")
        df = df.merge(join_df, on="id", how="left")
        if args.drop_missing:
            df = df[df[annotation_col].notna()].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(df)} {args.kind} annotations: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
