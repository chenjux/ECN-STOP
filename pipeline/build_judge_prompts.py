#!/usr/bin/env python3
"""Build answer-check prompt JSONL files for each sampled seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import (
    DEFAULT_TOKENIZER_DIR,
    PROJECT_ROOT,
    build_token_calculator,
    normalize_inference_df,
    parse_key_value_specs,
)
from pipeline.prompts import build_judge_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "prm12k.csv",
        help="CSV containing question and target columns.",
    )
    parser.add_argument(
        "--seed-output",
        action="append",
        required=True,
        help="Sample output in SEED=PATH format. Repeat for each seed.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Tokenizer used for response token counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "judge_prompts",
    )
    parser.add_argument("--prefix", default="self_distill_seed")
    parser.add_argument("--tail-n", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd

    seed_outputs = parse_key_value_specs(args.seed_output, Path)
    token_calculator = build_token_calculator(args.tokenizer_dir)

    metadata = pd.read_csv(args.metadata_csv)
    required = {"question", "target"}
    missing = required - set(metadata.columns)
    if missing:
        raise SystemExit(f"ERROR: Missing metadata columns: {sorted(missing)}")

    target_by_question = metadata.set_index("question")["target"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed, path in sorted(seed_outputs.items()):
        df = normalize_inference_df(pd.read_json(path, lines=True), token_calculator)
        df["target"] = df["question"].map(target_by_question)
        if df["target"].isna().any():
            missing_count = int(df["target"].isna().sum())
            raise SystemExit(f"ERROR: Seed {seed} has {missing_count} questions without target.")

        out_df = pd.DataFrame()
        out_df["messages"] = df.apply(
            lambda row: [
                {
                    "role": "user",
                    "content": build_judge_prompt(row["target"], row["response"], args.tail_n),
                }
            ],
            axis=1,
        )
        out_df["question"] = df["question"]
        out_df["target"] = df["target"]

        output_path = args.output_dir / f"{args.prefix}{seed}_check.jsonl"
        out_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print(f"Wrote {len(out_df)} judge prompts: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
