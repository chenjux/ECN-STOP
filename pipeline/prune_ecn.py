#!/usr/bin/env python3
"""Create ECN-pruned Swift training data from selected traces and annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.common import PROJECT_ROOT, node_id_to_index, parse_jsonl_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=PROJECT_ROOT / "pruned_data_pipeline" / "data" / "self_distill_best_of_N_seed.jsonl",
    )
    parser.add_argument(
        "--segmented",
        type=Path,
        default=PROJECT_ROOT
        / "pruned_data_pipeline"
        / "data"
        / "self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl",
    )
    parser.add_argument("--conclusions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "ecn_self_distill_qwen.jsonl",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        help="Optional JSONL with intermediate pruning columns.",
    )
    parser.add_argument(
        "--allow-intermediate-correct",
        action="store_true",
        help="Match the original notebook behavior by accepting any correct conclusion, even if it is not typed as answering.",
    )
    return parser.parse_args()


def is_answering_conclusion(item: dict[str, object]) -> bool:
    return "answering" in str(item.get("type", "")).lower()


def find_first_correct_answering(jsonl_text: object, allow_intermediate_correct: bool = False) -> str | None:
    for obj in parse_jsonl_objects(jsonl_text):
        if obj.get("is_correct") != 1:
            continue
        if allow_intermediate_correct or is_answering_conclusion(obj):
            node = obj.get("conclusion_node")
            return str(node) if node else None
    return None


def strip_node_prefix(paragraph: object) -> str:
    return str(paragraph).split(":", 1)[-1].strip()


def build_processed_text(row) -> str:
    if not row["need_pruned"]:
        return f"{row['think']}</think>{row['content']}"

    paragraphs = row.get("paragraphs") or []
    first_n = int(row.get("first_node_index") or 0)
    cleaned = [strip_node_prefix(paragraph) for paragraph in paragraphs[:first_n]]
    final_text = "\n".join(cleaned)

    if not bool(row.get("no_last_thinking_sentence", False)):
        last_sentence = str(row.get("last_thinking_sentence", "") or "").strip()
        if last_sentence:
            final_text += "\n" + last_sentence

    return final_text + "</think>" + str(row.get("content", "") or "")


def main() -> int:
    args = parse_args()

    import pandas as pd

    selected = pd.read_json(args.selected, lines=True).reset_index(drop=True)
    segmented = pd.read_json(args.segmented, lines=True)
    conclusions = pd.read_json(args.conclusions, lines=True)

    if "id" not in selected.columns:
        selected["id"] = selected.index + 1
    if "id" not in segmented.columns or "paragraphs" not in segmented.columns:
        raise SystemExit("ERROR: Segmented file must contain id and paragraphs columns.")
    if "id" not in conclusions.columns:
        raise SystemExit("ERROR: Conclusions file must contain id column.")

    conclusion_col = "conclusion" if "conclusion" in conclusions.columns else "jsonl"
    if conclusion_col not in conclusions.columns:
        raise SystemExit("ERROR: Conclusions file must contain conclusion or jsonl column.")

    df = selected.merge(segmented[["id", "paragraphs"]], on="id", how="left")
    df = df.merge(conclusions[["id", conclusion_col]], on="id", how="left")
    df = df.rename(columns={conclusion_col: "conclusion"})
    df["total_nodes"] = df["paragraphs"].apply(lambda value: len(value) if isinstance(value, list) else 0)
    df["first_correct_nodes"] = df["conclusion"].apply(
        lambda text: find_first_correct_answering(text, args.allow_intermediate_correct)
    )
    df = df[df["first_correct_nodes"].notna()].copy()
    df["first_node_index"] = df["first_correct_nodes"].apply(node_id_to_index)
    df = df[df["first_node_index"].notna()].copy()
    df["need_pruned"] = df["total_nodes"] != df["first_node_index"]
    df["processed_text"] = df.apply(build_processed_text, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            record = {
                "messages": [
                    {"role": "user", "content": row.get("question", "")},
                    {"role": "assistant", "content": row.get("processed_text", "")},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.debug_output, orient="records", lines=True, force_ascii=False)

    print(f"Wrote {len(df)} ECN-pruned training rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
