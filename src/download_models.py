from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from modelscope import snapshot_download

DEFAULT_IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state*",
    "events.out.tfevents*",
]


def load_models(config_path: Path) -> list[dict]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    if not models:
        raise ValueError(f"No models found in {config_path}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="Download all files, including default training-state artifacts.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    model_root = Path(args.model_root).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    models = load_models(config_path)
    ignore_patterns = None if args.no_default_ignore else DEFAULT_IGNORE_PATTERNS

    for model in models:
        alias = model["alias"]
        repo_id = model["repo_id"]
        target_dir = model_root / alias
        if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
            print(f"[skip] {alias} already exists at {target_dir}")
            continue
        print(f"[download] {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=args.token,
            ignore_patterns=ignore_patterns,
        )


if __name__ == "__main__":
    main()
