from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--datasets-config", required=True)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    models_payload = load_yaml(Path(args.models_config))
    datasets_payload = load_yaml(Path(args.datasets_config))
    results_root = Path(args.results_root).resolve()

    rows: list[dict] = []
    for model in models_payload["models"]:
        for dataset in datasets_payload["datasets"]:
            metrics_path = results_root / model["alias"] / dataset["name"] / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_alias": model["alias"],
                    "repo_id": model["repo_id"],
                    "scheme": model["scheme"],
                    "teacher_size": model["teacher_size"],
                    "maxlen": model["maxlen"],
                    "dataset": dataset["name"],
                    "n": metrics["n"],
                    "k": metrics["k"],
                    "avg_at_n": metrics["avg_at_n"],
                    "pass_at_n": metrics["pass_at_n"],
                    "avg_len": metrics["avg_len"],
                    "count": metrics["count"],
                    "elapsed_sec": metrics["elapsed_sec"],
                    "metrics_path": str(metrics_path),
                }
            )

    summary_df = pd.DataFrame(rows).sort_values(["model_alias", "dataset"]).reset_index(drop=True)
    summary_per_model_dataset = results_root / "summary_per_model_dataset.csv"
    summary_df.to_csv(summary_per_model_dataset, index=False)

    wide_rows: list[dict] = []
    for model in models_payload["models"]:
        row = {
            "model_alias": model["alias"],
            "repo_id": model["repo_id"],
            "scheme": model["scheme"],
            "teacher_size": model["teacher_size"],
            "maxlen": model["maxlen"],
        }
        model_df = summary_df[summary_df["model_alias"] == model["alias"]]
        for _, dataset_row in model_df.iterrows():
            n_value = int(dataset_row["n"])
            dataset_name = dataset_row["dataset"]
            row[f"{dataset_name}_avg@{n_value}"] = dataset_row["avg_at_n"]
            row[f"{dataset_name}_pass@{n_value}"] = dataset_row["pass_at_n"]
            row[f"{dataset_name}_avg_len"] = dataset_row["avg_len"]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows).sort_values("model_alias").reset_index(drop=True)
    summary_wide = results_root / "summary_wide.csv"
    wide_df.to_csv(summary_wide, index=False)

    print(summary_per_model_dataset)
    print(summary_wide)


if __name__ == "__main__":
    main()

