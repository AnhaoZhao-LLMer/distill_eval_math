from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.eval_logic import MathVerifier, build_prompt, ensure_dir, load_eval_records, score_candidates, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.k < 1 or args.n < 1 or args.k > args.n:
        raise ValueError(f"k ({args.k}) must satisfy 1 <= k <= n ({args.n}).")

    output_dir = ensure_dir(args.output_dir)
    records = load_eval_records(args.dataset_path, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, item["question"], args.enable_thinking) for item in records]

    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    verifier = MathVerifier()
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    acc_sum = 0.0
    pass_sum = 0.0
    total_len = 0
    total_candidate_count = 0
    rows: list[dict] = []

    for item, output in zip(records, outputs, strict=True):
        candidates = []
        for candidate in output.outputs:
            candidates.append(
                {
                    "text": candidate.text,
                    "token_ids": list(candidate.token_ids),
                    "token_len": len(candidate.token_ids),
                }
            )

        scored = score_candidates(verifier, item["gold_answer"], candidates, args.k)
        k_used = scored["k_used"]
        considered = candidates[:k_used]
        sample_rows = []
        for idx, candidate in enumerate(candidates):
            sample_rows.append(
                {
                    "sample_idx": idx,
                    "text": candidate["text"],
                    "token_len": candidate["token_len"],
                    "is_correct": bool(scored["correct_flags"][idx]) if idx < len(scored["correct_flags"]) else False,
                }
            )

        acc_sum += scored["avg_at_k"]
        pass_sum += scored["pass_at_k"]
        total_len += sum(candidate["token_len"] for candidate in considered)
        total_candidate_count += k_used

        rows.append(
            {
                "sample_idx": int(item["sample_idx"]),
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "samples": sample_rows,
                "selected_text": scored["selected_text"],
                "selected_len": scored["selected_len"],
                "selected_is_correct": scored["is_correct"],
                "avg_at_n": scored["avg_at_k"],
                "pass_at_n": scored["pass_at_k"],
                "k_used": k_used,
            }
        )

    elapsed_sec = time.time() - start_time
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "model_alias": args.model_alias,
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset_name,
        "n": args.n,
        "k": args.k,
        "avg_at_n": acc_sum / max(1, len(records)),
        "pass_at_n": pass_sum / max(1, len(records)),
        "avg_len": total_len / max(1, total_candidate_count),
        "count": len(records),
        "elapsed_sec": elapsed_sec,
        "predictions_path": str(predictions_path),
    }
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

