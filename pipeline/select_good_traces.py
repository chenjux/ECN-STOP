#!/usr/bin/env python3
"""Select one correct self-distilled trace per question from multiple seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import (
    DEFAULT_TOKENIZER_DIR,
    PROJECT_ROOT,
    build_token_calculator,
    extract_last_thinking_signal,
    normalize_inference_df,
    parse_bool,
    parse_key_value_specs,
    split_by_content_part,
    split_by_think_part,
)


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
        "--check-result",
        action="append",
        required=True,
        help="Answer-check output in SEED=PATH format. Repeat for each seed.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Tokenizer used for response token counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "self_distill_best_of_N_seed.jsonl",
    )
    parser.add_argument(
        "--selection",
        choices=["median_correct", "shortest_correct"],
        default="median_correct",
        help="How to choose among correct seed responses.",
    )
    return parser.parse_args()


def choose_seed(row, seeds: list[int], strategy: str) -> str:
    candidates = []
    for seed in seeds:
        flag_col = f"seed{seed}_TrueOrFalse"
        token_col = f"seed{seed}_tokens"
        if bool(row.get(flag_col)):
            candidates.append(
                {
                    "seed_name": f"seed{seed}",
                    "token_count": row.get(token_col),
                }
            )

    if not candidates:
        return "Not exist"

    candidates.sort(key=lambda item: item["token_count"])
    if strategy == "shortest_correct":
        return candidates[0]["seed_name"]

    return candidates[len(candidates) // 2]["seed_name"]


def extract_best_response(row) -> object | None:
    seed_name = row["best_seed"]
    if seed_name == "Not exist":
        return None
    return row.get(f"{seed_name}_response")


def main() -> int:
    args = parse_args()

    import pandas as pd

    seed_outputs = parse_key_value_specs(args.seed_output, Path)
    check_results = parse_key_value_specs(args.check_result, Path)
    if set(seed_outputs) != set(check_results):
        raise SystemExit("ERROR: --seed-output and --check-result seeds must match.")

    seeds = sorted(seed_outputs)
    token_calculator = build_token_calculator(args.tokenizer_dir)
    metadata = pd.read_csv(args.metadata_csv)
    required = {"question", "target"}
    missing = required - set(metadata.columns)
    if missing:
        raise SystemExit(f"ERROR: Missing metadata columns: {sorted(missing)}")
    if metadata["question"].duplicated().any():
        raise SystemExit("ERROR: Metadata contains duplicate questions; cannot merge safely.")

    merged = metadata[["question", "target"]].copy()

    for seed in seeds:
        seed_df = normalize_inference_df(pd.read_json(seed_outputs[seed], lines=True), token_calculator)
        if seed_df["question"].duplicated().any():
            raise SystemExit(f"ERROR: Seed {seed} output contains duplicate questions.")

        check_df = pd.read_json(check_results[seed], lines=True)
        if len(check_df) != len(seed_df):
            raise SystemExit(
                f"ERROR: Seed {seed} check rows ({len(check_df)}) do not match sample rows ({len(seed_df)})."
            )
        if "response" not in check_df.columns:
            raise SystemExit(f"ERROR: Seed {seed} check result missing response column.")

        seed_df[f"seed{seed}_TrueOrFalse"] = [parse_bool(value) for value in check_df["response"].tolist()]
        seed_df = seed_df.rename(
            columns={
                "response": f"seed{seed}_response",
                "tokens": f"seed{seed}_tokens",
            }
        )
        merged = merged.merge(seed_df, on="question", how="left")

    merged["best_seed"] = merged.apply(lambda row: choose_seed(row, seeds, args.selection), axis=1)
    merged["best_seed_response"] = merged.apply(extract_best_response, axis=1)
    selected = merged[merged["best_seed_response"].notna()].copy()

    selected["last_thinking_sentence"] = selected["best_seed_response"].apply(extract_last_thinking_signal)
    selected["think"] = selected["best_seed_response"].apply(split_by_think_part)
    selected["content"] = selected["best_seed_response"].apply(split_by_content_part)
    selected["no_last_thinking_sentence"] = selected["last_thinking_sentence"].apply(lambda value: value == "")

    output_cols = [
        "question",
        "target",
        "best_seed",
        "best_seed_response",
        "last_thinking_sentence",
        "think",
        "content",
        "no_last_thinking_sentence",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected[output_cols].to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(selected)} selected traces: {args.output}")
    print(f"Dropped {len(merged) - len(selected)} questions without a correct seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
