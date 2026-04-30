from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import swanlab
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--project", default="distill_eval_vllm")
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    models_payload = load_yaml(Path(args.models_config))

    for model in models_payload["models"]:
        model_df = summary_df[summary_df["model_alias"] == model["alias"]]
        if model_df.empty:
            continue

        swanlab.init(
            project=args.project,
            experiment_name=model["alias"],
            config={
                "model_id": model["repo_id"],
                "scheme": model["scheme"],
                "teacher_size": model["teacher_size"],
                "maxlen": model["maxlen"],
                "backend": "vllm",
            },
        )
        metrics = {}
        for _, row in model_df.iterrows():
            dataset_name = row["dataset"]
            n_value = int(row["n"])
            metrics[f"{dataset_name}/avg@{n_value}"] = row["avg_at_n"]
            metrics[f"{dataset_name}/pass@{n_value}"] = row["pass_at_n"]
            metrics[f"{dataset_name}/avg_len"] = row["avg_len"]
        swanlab.log(metrics)
        if hasattr(swanlab, "finish"):
            swanlab.finish()


if __name__ == "__main__":
    main()
