#!/usr/bin/env python3
"""Summarize EvalScope reports and response token lengths."""

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[1] / "pruned_data_pipeline" / "ds_tokenizer"
)


def build_token_calculator(tokenizer_dir: Path) -> Callable[[object], int]:
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )

    def token_calculator(value: object) -> int:
        if isinstance(value, str):
            return len(tokenizer.encode(value))
        if isinstance(value, list):
            return sum(len(tokenizer.encode(item)) for item in value if isinstance(item, str))
        return 0

    return token_calculator


def extract_response(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""

    first_message = messages[0]
    if not isinstance(first_message, dict):
        return ""

    content = first_message.get("content", "")
    return content if isinstance(content, str) else ""


def compute_avg_tokens_from_jsonl(
    jsonl_files: Iterable[Path],
    token_calculator: Callable[[object], int],
) -> float | None:
    import pandas as pd

    all_tokens: list[int] = []

    for path in jsonl_files:
        try:
            df = pd.read_json(path, lines=True)
        except ValueError as exc:
            print(f"[WARN] Failed to read predictions {path}: {exc}")
            continue

        if "messages" not in df.columns:
            print(f"[WARN] Missing messages column in {path}")
            continue

        tokens = df["messages"].apply(extract_response).apply(token_calculator)
        all_tokens.extend(tokens.tolist())

    if not all_tokens:
        return None

    return sum(all_tokens) / len(all_tokens)


def detect_dataset_and_files(model_pred_dir: Path) -> tuple[str | None, list[Path]]:
    files = sorted(model_pred_dir.glob("*.jsonl"))
    names = {path.name for path in files}

    math_files = [path for path in files if path.name.startswith("math_500_Level")]
    if math_files:
        return "math_500", math_files

    if "aime24_default.jsonl" in names:
        return "aime_2024", [model_pred_dir / "aime24_default.jsonl"]

    if "gsm8k_main.jsonl" in names:
        return "gsm8k", [model_pred_dir / "gsm8k_main.jsonl"]

    return None, []


def extract_accuracy_from_reports(run_dir: Path) -> tuple[str | None, str | None, float | None]:
    reports_root = run_dir / "reports"
    if not reports_root.exists():
        return None, None, None

    model_dirs = [path for path in reports_root.iterdir() if path.is_dir()]
    if len(model_dirs) != 1:
        print(
            f"[WARN] {run_dir.name}: expected 1 reports model dir, found {len(model_dirs)}"
        )
        return None, None, None

    model_dir = model_dirs[0]
    report_files = sorted(model_dir.glob("*.json"))
    if not report_files:
        return None, None, None

    report_path = report_files[0]
    try:
        data = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {report_path}: {exc}")
        return None, None, None

    return (
        data.get("model_name", model_dir.name),
        data.get("dataset_name"),
        data.get("score"),
    )


def process_single_run(
    run_dir: Path,
    token_calculator: Callable[[object], int],
) -> dict[str, object] | None:
    model_name, report_dataset, accuracy = extract_accuracy_from_reports(run_dir)
    if accuracy is None:
        print(f"[WARN] Missing accuracy in {run_dir.name}")
        return None

    predictions_root = run_dir / "predictions"
    if not predictions_root.exists():
        print(f"[WARN] Missing predictions directory in {run_dir.name}")
        return None

    model_dirs = [path for path in predictions_root.iterdir() if path.is_dir()]
    if len(model_dirs) != 1:
        print(
            f"[WARN] {run_dir.name}: expected 1 predictions model dir, found {len(model_dirs)}"
        )
        return None

    model_pred_dir = model_dirs[0]
    prediction_dataset, jsonl_files = detect_dataset_and_files(model_pred_dir)
    if not jsonl_files:
        print(f"[WARN] No supported prediction files in {model_pred_dir}")
        return None

    return {
        "run_id": run_dir.name,
        "model_name": model_name or model_pred_dir.name,
        "dataset": prediction_dataset or report_dataset,
        "accuracy": accuracy,
        "avg_token_length": compute_avg_tokens_from_jsonl(jsonl_files, token_calculator),
    }


def process_all_runs(
    outputs_root: Path,
    token_calculator: Callable[[object], int],
):
    import pandas as pd

    rows = []

    for run_dir in sorted(outputs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        row = process_single_run(run_dir, token_calculator)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize EvalScope run accuracy and average response tokens.",
    )
    parser.add_argument(
        "outputs_root",
        nargs="?",
        type=Path,
        default=Path("outputs"),
        help="Directory containing EvalScope run outputs.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Tokenizer directory used to count response tokens.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for writing the summary table as CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.outputs_root.exists():
        raise SystemExit(f"Outputs directory does not exist: {args.outputs_root}")
    if not args.tokenizer_dir.exists():
        raise SystemExit(f"Tokenizer directory does not exist: {args.tokenizer_dir}")

    token_calculator = build_token_calculator(args.tokenizer_dir)
    df = process_all_runs(args.outputs_root, token_calculator)

    if df.empty:
        print("No completed runs found.")
    else:
        print(df.to_string(index=False))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote summary CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
