import os
import time
import subprocess
import requests
import signal
from evalscope import TaskConfig, run_task
import shlex

# ================= Setup =================
#Example model list
MODEL_LIST = [
('./model/no_thinking_8192_seed1_2026_0103_055650/v0-20260103-055657/checkpoint-118-merged', 'no_thinking_8192_seed1_qwen'),
('./model/random_pruned_8192_seed1_2026_0103_060800/v0-20260103-060808/checkpoint-98-merged', 'random_pruned_8192_qwen'),
]

GPU_ID = "0"
BASE_PORT = 8801

# ================= function =================
def wait_server(url, timeout=300):
    """等待服务器启动"""
    print("Waiting for server", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/models", timeout=2).status_code == 200:
                print(" Ready!")
                return True
        except:
            pass
        time.sleep(5)
        print(".", end="", flush=True)
    print(" Failed!")
    return False

def run_eval(api_url, model_name):
    """运行评测"""
    print(f"Evaluating {model_name}...")
    config = TaskConfig(
        api_url=api_url,
        model=model_name,
        eval_type='openai_api',
        datasets=['math_500'], #aime24 math_500 'math_500','gsm8k'
        eval_batch_size=32,  # 
        generation_config={
            'max_tokens': 8192,
            'temperature': 0,
            'top_p': 0.95,
            'seed': 1,
        },
    )
    try:
        run_task(config)
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False

def kill_process(process):
    """杀死进程"""
    if process and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=20)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass

# ================= 主程序 =================
def main():
    os.makedirs("vllm_logs", exist_ok=True)
    
    for i, (model_path, model_name) in enumerate(MODEL_LIST):
        port = BASE_PORT + i
        url = f"http://127.0.0.1:{port}/v1"
        process = None
        
        print(f"\n{'='*50}\n[{i+1}/{len(MODEL_LIST)}] {model_name} (Port {port})\n{'='*50}")
        
        try:
            # 启动 vLLM
            cmd = (f"python -m vllm.entrypoints.openai.api_server "
                   f"--model {model_path} --served-model-name {model_name} "
                   f"--port {port} --trust-remote-code --seed 1 "
                   f"--disable-log-requests --enforce-eager")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = GPU_ID
            
            with open(f"vllm_logs/{model_name}.log", 'w') as f:
                process = subprocess.Popen(
                    shlex.split(cmd), stdout=f, stderr=f, 
                    env=env, start_new_session=True
                )
            
            # 等待并评测
            if wait_server(url):
                run_eval(url, model_name)
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            # 清理
            kill_process(process)
            print("Cooling down 15s...")
            time.sleep(15)

if __name__ == "__main__":
    main()