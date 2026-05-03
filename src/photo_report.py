from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DATASETS = ["gsm8k", "math500", "amc23", "aime24"]


def short_title(alias: str) -> str:
    if alias == "grpo_base":
        return "GRPO base"

    grpo_match = re.fullmatch(
        r"grpo_(student_forward|student_reverse|teacher_forward|teacher_reverse)_"
        r"studentqwen3_0\.6b_base_teacherqwen3_(4b|8b)_maxlen(128|4096)",
        alias,
    )
    if grpo_match:
        scheme, teacher_size, maxlen = grpo_match.groups()
        return f"GRPO {scheme} T{teacher_size.upper()} L{maxlen}"

    distill_match = re.fullmatch(
        r"(student_forward|student_reverse|teacher_forward|teacher_reverse)_"
        r"student0\.6B_teacher(4B|8B)_maxlen(128|4096)",
        alias,
    )
    if distill_match:
        scheme, teacher_size, maxlen = distill_match.groups()
        return f"{scheme} T{teacher_size} L{maxlen}"

    return alias


def has_any_metrics(model_dir: Path) -> bool:
    return any((model_dir / dataset / "metrics.json").exists() for dataset in DATASETS)


def format_model_block(model_dir: Path) -> str:
    lines = [
        short_title(model_dir.name),
        "dataset    avg    pass     len",
    ]

    for dataset in DATASETS:
        metrics_path = model_dir / dataset / "metrics.json"
        if not metrics_path.exists():
            lines.append(f"{dataset:<8}  MISS")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        avg = 100 * float(metrics["avg_at_n"])
        pass_at_n = 100 * float(metrics["pass_at_n"])
        avg_len = float(metrics["avg_len"])

        lines.append(f"{dataset:<8} {avg:6.2f} {pass_at_n:7.2f} {avg_len:7.2f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_path = Path(args.output) if args.output else results_root / "photo_report.txt"

    model_dirs = sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and has_any_metrics(path)
    )

    blocks = [format_model_block(model_dir) for model_dir in model_dirs]
    output_path.write_text("\n\n\n\n".join(blocks) + "\n", encoding="utf-8")

    print(output_path)


if __name__ == "__main__":
    main()