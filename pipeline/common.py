#!/usr/bin/env python3
"""Shared utilities for the STOP data pipeline."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "pruned_data_pipeline" / "ds_tokenizer"


def parse_key_value_specs(specs: Iterable[str], value_type: Callable = Path) -> dict[int, object]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected SEED=PATH format, got: {spec}")
        key, value = spec.split("=", 1)
        try:
            seed = int(key)
        except ValueError as exc:
            raise ValueError(f"Seed must be an integer in spec: {spec}") from exc
        parsed[seed] = value_type(value)
    return parsed


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_first_message_content(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    first_message = messages[0]
    if not isinstance(first_message, dict):
        return ""
    content = first_message.get("content", "")
    return content if isinstance(content, str) else ""


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False

    match = re.search(r"\b(true|false|yes|no)\b", normalized)
    if not match:
        return False
    return match.group(1) in {"true", "yes"}


def token_count(value: object, tokenizer: object) -> int:
    if isinstance(value, str):
        return len(tokenizer.encode(value))
    if isinstance(value, list):
        return sum(len(tokenizer.encode(item)) for item in value if isinstance(item, str))
    return 0


def build_token_calculator(tokenizer_dir: Path) -> Callable[[object], int]:
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )
    return lambda value: token_count(value, tokenizer)


def normalize_inference_df(df, token_calculator: Callable[[object], int]):
    missing = {"messages", "response"} - set(df.columns)
    if missing:
        raise ValueError(f"Inference output missing columns: {sorted(missing)}")

    normalized = df.copy()
    normalized["question"] = normalized["messages"].apply(extract_first_message_content)
    normalized["tokens"] = normalized["response"].apply(token_calculator)
    normalized = normalized.drop(columns=["labels", "logprobs"], errors="ignore")
    return normalized[["question", "response", "tokens"]]


def build_chat_records(questions: Iterable[str]) -> list[dict[str, object]]:
    return [
        {"messages": [{"role": "user", "content": str(question)}]}
        for question in questions
    ]


def write_jsonl_records(records: Iterable[dict[str, object]], output_path: Path) -> int:
    ensure_parent(output_path)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_last_thinking_signal(text: object) -> str:
    if not isinstance(text, str):
        return ""

    parts = text.split("</think>")
    if len(parts) < 2:
        return "No </think> Label"

    think_content = parts[0]
    target_phrase = "**Final Answer**"
    idx = think_content.find(target_phrase)
    if idx == -1:
        return ""

    pre_text = think_content[:idx]
    start_pos = max(pre_text.rfind("\n"), pre_text.rfind("."))
    if start_pos != -1:
        return think_content[start_pos + 1 :].strip()
    return think_content[idx:].strip()


def split_by_think_part(text: object) -> str:
    if not isinstance(text, str) or "</think>" not in text:
        return "error"
    return text[: text.find("</think>")]


def split_by_content_part(text: object) -> str:
    if not isinstance(text, str) or "</think>" not in text:
        return "error"
    marker = "</think>"
    return text[text.find(marker) + len(marker) :].lstrip()


def split_text_into_paragraphs(text: object) -> list[str]:
    if not isinstance(text, str):
        text = ""
    text = text.replace("\\n", "\n")
    heuristic_patterns = [
        r"\bHmm\b",
        r"\bWait\b",
        r"\bBut\b",
        r"\bSo\b",
        r"\bAlternatively\b",
        r"\bNow\b",
    ]

    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        is_trigger = any(re.search(pattern, stripped, re.IGNORECASE) for pattern in heuristic_patterns)
        if is_trigger and len(current_paragraph) >= 4:
            paragraphs.append(" ".join(current_paragraph).strip())
            current_paragraph = []

        current_paragraph.append(stripped)

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph).strip())

    return [f"N{index + 1}: {paragraph}" for index, paragraph in enumerate(paragraphs)]


def extract_jsonl_block(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"```jsonl\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    return stripped if stripped.startswith("{") else None


def extract_gemini_text(response: object) -> str:
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def parse_request_id(value: object) -> int | None:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def parse_jsonl_objects(text: object) -> list[dict[str, object]]:
    result = []
    for line in str(text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parsed = None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

        if isinstance(parsed, dict):
            result.append(parsed)

    return result


def node_id_to_index(node_id: object) -> int | None:
    match = re.search(r"(\d+)$", str(node_id))
    return int(match.group(1)) if match else None
