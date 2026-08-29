#!/usr/bin/env python3
"""Create self-distillation prompt JSONL chunks from PRM12K metadata."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pipeline.common import PROJECT_ROOT, build_chat_records, write_jsonl_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "prm12k.csv",
        help="CSV containing a question column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data",
        help="Directory for generated JSONL chunks.",
    )
    parser.add_argument("--prefix", default="self_distill_questions_dataset")
    parser.add_argument("--chunk-size", type=int, default=995)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd

    if args.chunk_size <= 0:
        raise SystemExit("ERROR: --chunk-size must be positive.")

    df = pd.read_csv(args.input_csv)
    if "question" not in df.columns:
        raise SystemExit(f"ERROR: Missing question column in {args.input_csv}")

    records = build_chat_records(df["question"].tolist())
    num_chunks = math.ceil(len(records) / args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(num_chunks):
        start = index * args.chunk_size
        end = min((index + 1) * args.chunk_size, len(records))
        output_path = args.output_dir / f"{args.prefix}_{index + 1}.jsonl"
        count = write_jsonl_records(records[start:end], output_path)
        print(f"Wrote {count} records: {output_path}")

    print(f"Total records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
