#!/usr/bin/env python3
"""Launch vLLM servers and evaluate them with EvalScope."""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL_LIST = [
    (
        "./model/no_thinking_8192_seed1_2026_0103_055650/"
        "v0-20260103-055657/checkpoint-118-merged",
        "no_thinking_8192_seed1_qwen",
    ),
    (
        "./model/random_pruned_8192_seed1_2026_0103_060800/"
        "v0-20260103-060808/checkpoint-98-merged",
        "random_pruned_8192_qwen",
    ),
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_model_specs(specs: list[str] | None) -> list[tuple[str, str]]:
    if not specs:
        return DEFAULT_MODEL_LIST

    models = []
    for spec in specs:
        if "=" not in spec:
            raise argparse.ArgumentTypeError(
                f"Model spec must use MODEL_PATH=MODEL_NAME format: {spec}"
            )
        model_path, model_name = spec.rsplit("=", 1)
        models.append((model_path, model_name))

    return models


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def wait_server(url: str, timeout: int = 300, interval: int = 5) -> bool:
    import requests

    print("Waiting for server", end="", flush=True)
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/models", timeout=2)
            if response.status_code == 200:
                print(" ready")
                return True
        except requests.RequestException:
            pass

        time.sleep(interval)
        print(".", end="", flush=True)

    print(" failed")
    return False


def run_eval(
    api_url: str,
    model_name: str,
    datasets: list[str],
    eval_batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> bool:
    from evalscope import TaskConfig, run_task

    print(f"Evaluating {model_name} on {', '.join(datasets)}")
    config = TaskConfig(
        api_url=api_url,
        model=model_name,
        eval_type="openai_api",
        datasets=datasets,
        eval_batch_size=eval_batch_size,
        generation_config={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    )

    try:
        run_task(config)
    except Exception as exc:
        print(f"Evaluation failed for {model_name}: {exc}")
        return False

    return True


def start_vllm_server(
    model_path: str,
    model_name: str,
    host: str,
    port: int,
    gpu_id: str,
    seed: int,
    enforce_eager: bool,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--seed",
        str(seed),
        "--disable-log-requests",
    ]

    if enforce_eager:
        cmd.append("--enforce-eager")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    try:
        return subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()


def kill_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EvalScope benchmarks against one or more vLLM servers.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model spec in MODEL_PATH=MODEL_NAME format. Can be repeated.",
    )
    parser.add_argument("--gpu-id", default="0", help="CUDA device id for each vLLM run.")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host.")
    parser.add_argument("--base-port", type=int, default=8801, help="First vLLM port.")
    parser.add_argument(
        "--datasets",
        default="math_500",
        help="Comma-separated EvalScope datasets, for example math_500,gsm8k,aime24.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument("--cooldown-seconds", type=int, default=15)
    parser.add_argument("--log-dir", type=Path, default=Path("vllm_logs"))
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Do not pass --enforce-eager to vLLM.",
    )
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        models = parse_model_specs(args.model)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    datasets = split_csv(args.datasets)
    if not datasets:
        raise SystemExit("ERROR: --datasets must contain at least one dataset name.")

    failed_models: list[str] = []

    for index, (model_path, model_name) in enumerate(models, start=1):
        port = args.base_port + index - 1
        api_url = f"http://{args.host}:{port}/v1"
        log_path = args.log_dir / f"{safe_filename(model_name)}.log"
        process = None

        print()
        print("=" * 72)
        print(f"[{index}/{len(models)}] {model_name} on port {port}")
        print("=" * 72)

        try:
            process = start_vllm_server(
                model_path=model_path,
                model_name=model_name,
                host=args.host,
                port=port,
                gpu_id=args.gpu_id,
                seed=args.seed,
                enforce_eager=args.enforce_eager,
                log_path=log_path,
            )

            if wait_server(api_url, timeout=args.server_timeout):
                ok = run_eval(
                    api_url=api_url,
                    model_name=model_name,
                    datasets=datasets,
                    eval_batch_size=args.eval_batch_size,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                if not ok:
                    failed_models.append(model_name)
            else:
                failed_models.append(model_name)
                print(f"Server log: {log_path}")

        except Exception as exc:
            failed_models.append(model_name)
            print(f"Error while evaluating {model_name}: {exc}")

        finally:
            kill_process(process)
            print(f"Cooling down {args.cooldown_seconds}s")
            time.sleep(args.cooldown_seconds)

    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
