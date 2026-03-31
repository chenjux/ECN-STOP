import pandas as pd
import transformers
import os
from pathlib import Path

# ====== 初始化 tokenizer ======
chat_tokenizer_dir = Path(__file__).resolve().parent.parent / "pruned_data_pipeline" / "ds_tokenizer"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)

def token_calculator(paragraphs):
    if isinstance(paragraphs, str):
        return len(tokenizer.encode(paragraphs))
    elif isinstance(paragraphs, list):
        return sum(len(tokenizer.encode(p)) for p in paragraphs if isinstance(p, str))
    else:
        return 0  # Fallback for unexpected types

# =========================================================
# Token computation
# =========================================================

#!/usr/bin/env python3
import os
import json
from pathlib import Path
import pandas as pd


# =========================================================
# Token computation from jsonl
# =========================================================

def compute_avg_tokens_from_jsonl(jsonl_files, token_calculator):
    all_tokens = []

    for path in jsonl_files:
        df = pd.read_json(path, lines=True)

        def extract_response(messages):
            if not messages or "content" not in messages[0]:
                return ""
            return messages[0]["content"]

        df["response"] = df["messages"].apply(extract_response)
        df["tokens"] = df["response"].apply(token_calculator)

        all_tokens.extend(df["tokens"].tolist())

    if not all_tokens:
        return None

    return sum(all_tokens) / len(all_tokens)


# =========================================================
# Detect dataset + prediction files
# =========================================================

def detect_dataset_and_files(model_pred_dir):
    files = list(model_pred_dir.glob("*.jsonl"))
    names = [f.name for f in files]

    # math_500: 5 levels
    math_files = sorted([f for f in files if f.name.startswith("math_500_Level")])
    if math_files:
        return "math_500", math_files

    # AIME
    if "aime24_default.jsonl" in names:
        return "aime_2024", [model_pred_dir / "aime24_default.jsonl"]

    # GSM8K
    if "gsm8k_main.jsonl" in names:
        return "gsm8k", [model_pred_dir / "gsm8k_main.jsonl"]

    return None, []


# =========================================================
# Extract accuracy from reports
# =========================================================

def extract_accuracy_from_reports(run_dir):
    """
    reports/
      └── <model_dir>/
          └── *.json
    """
    reports_root = run_dir / "reports"
    if not reports_root.exists():
        return None, None, None

    model_dirs = [d for d in reports_root.iterdir() if d.is_dir()]
    if len(model_dirs) != 1:
        print(f"[WARN] {run_dir.name}: expected 1 reports model dir, found {len(model_dirs)}")
        return None, None, None

    model_dir = model_dirs[0]
    report_files = list(model_dir.glob("*.json"))
    if not report_files:
        return None, None, None

    report_path = report_files[0]
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to read {report_path}: {e}")
        return None, None, None

    model_name = data.get("model_name", model_dir.name)
    dataset_name = data.get("dataset_name")
    accuracy = data.get("score")

    return model_name, dataset_name, accuracy


# =========================================================
# Process ONE run (1 model)
# =========================================================

def process_single_run(run_dir, token_calculator):
    run_dir = Path(run_dir)

    # ---------- accuracy ----------
    model_name_r, dataset_r, accuracy = extract_accuracy_from_reports(run_dir)
    if accuracy is None:
        print(f"[WARN] Missing accuracy in {run_dir.name}")
        return None

    # ---------- predictions ----------
    predictions_root = run_dir / "predictions"
    if not predictions_root.exists():
        return None

    model_dirs = [d for d in predictions_root.iterdir() if d.is_dir()]
    if len(model_dirs) != 1:
        print(f"[WARN] {run_dir.name}: expected 1 predictions model dir, found {len(model_dirs)}")
        return None

    model_pred_dir = model_dirs[0]

    dataset_p, jsonl_files = detect_dataset_and_files(model_pred_dir)
    if not jsonl_files:
        print(f"[WARN] No prediction files in {model_pred_dir}")
        return None

    avg_tokens = compute_avg_tokens_from_jsonl(
        jsonl_files=jsonl_files,
        token_calculator=token_calculator,
    )

    return {
        "run_id": run_dir.name,
        "model_name": model_name_r or model_pred_dir.name,
        "dataset": dataset_p or dataset_r,
        "accuracy": accuracy,
        "avg_token_length": avg_tokens,
    }


# =========================================================
# Process ALL runs under outputs_qwen
# =========================================================

def process_all_runs(outputs_root, token_calculator):
    outputs_root = Path(outputs_root)
    rows = []

    for run_dir in sorted(outputs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        row = process_single_run(run_dir, token_calculator)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)



# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":


    outputs_root = "/Users/dexter/Desktop/acl_data/tokenskip"

    df = process_all_runs(outputs_root, token_calculator)
    print(df)
