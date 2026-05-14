#!/usr/bin/env python3
"""Build Google batch request JSONL files for taxonomy or conclusion annotation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pipeline.common import PROJECT_ROOT
from pipeline.prompts import build_conclusion_prompt, build_taxonomy_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["taxonomy", "conclusion"], required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--prefix", help="Output filename prefix.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_request(row, key: str, prompt: str, temperature: float) -> dict[str, object]:
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"temperature": temperature},
        },
    }


def main() -> int:
    args = parse_args()

    import pandas as pd

    if args.batch_size <= 0:
        raise SystemExit("ERROR: --batch-size must be positive.")

    df = pd.read_json(args.input, lines=True)
    prompt_builder = build_taxonomy_prompt if args.task == "taxonomy" else build_conclusion_prompt
    prefix = args.prefix or f"my-batch-request-{args.task}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_batches = math.ceil(len(df) / args.batch_size)
    for batch_index in range(num_batches):
        start = batch_index * args.batch_size
        end = min((batch_index + 1) * args.batch_size, len(df))
        batch_df = df.iloc[start:end]
        output_path = args.output_dir / f"{prefix}-{batch_index + 1}.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for row_index, row in batch_df.iterrows():
                request_id = row.get("id", row_index + 1)
                prompt = prompt_builder(row)
                request = build_request(row, f"request-{request_id}", prompt, args.temperature)
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"Wrote rows {start}-{end - 1}: {output_path}")

    print(f"Total rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
