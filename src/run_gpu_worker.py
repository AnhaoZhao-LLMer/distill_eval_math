from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def chunk_models(models: list[dict], worker_count: int) -> list[list[dict]]:
    chunk_size = math.ceil(len(models) / worker_count)
    return [models[idx : idx + chunk_size] for idx in range(0, len(models), chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--datasets-config", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_payload = load_yaml(Path(args.models_config))
    datasets_payload = load_yaml(Path(args.datasets_config))
    model_chunks = chunk_models(models_payload["models"], args.worker_count)
    my_models = model_chunks[args.worker_index] if args.worker_index < len(model_chunks) else []
    global_cfg = datasets_payload["global"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONPATH"] = str(repo_root)

    for model in my_models:
        model_path = Path(args.model_root).resolve() / model["alias"]
        for dataset in datasets_payload["datasets"]:
            output_dir = Path(args.results_root).resolve() / model["alias"] / dataset["name"]
            cmd = [
                args.python_bin,
                "-m",
                "src.run_dataset_eval",
                "--model_name_or_path",
                str(model_path),
                "--model_alias",
                model["alias"],
                "--dataset_name",
                dataset["name"],
                "--dataset_path",
                str((repo_root / dataset["path"]).resolve()),
                "--output_dir",
                str(output_dir),
                "--n",
                str(dataset["n"]),
                "--k",
                str(dataset["k"]),
                "--temperature",
                str(global_cfg["temperature"]),
                "--top_p",
                str(global_cfg["top_p"]),
                "--top_k",
                str(global_cfg["top_k"]),
                "--max_new_tokens",
                str(global_cfg["max_new_tokens"]),
                "--tensor_parallel_size",
                str(global_cfg["tensor_parallel_size"]),
                "--gpu_memory_utilization",
                str(global_cfg["gpu_memory_utilization"]),
            ]
            if global_cfg.get("enable_thinking", False):
                cmd.append("--enable_thinking")
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            print(f"[worker {args.worker_index}] running {model['alias']} on {dataset['name']}")
            subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()

