from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def extract_last_boxed(text: str) -> str:
    if not text:
        return ""
    pos = 0
    last = ""
    while True:
        idx = text.find(r"\boxed", pos)
        if idx < 0:
            break
        brace_start = text.find("{", idx)
        if brace_start < 0:
            pos = idx + 6
            continue
        depth = 0
        for i in range(brace_start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace_start + 1 : i].strip()
                    break
        pos = idx + 6
    return last


def extract_after_hashes(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    return ""


def extract_candidate_answer(text: str) -> str:
    return extract_last_boxed(text) or extract_after_hashes(text) or (text or "").strip()


def _normalize_interval_text(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).strip().replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _split_top_level_comma(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "{[(":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def parse_interval_bounds(text: str) -> tuple[str, str, str, str] | None:
    normalized = _normalize_interval_text(text)
    if len(normalized) < 3:
        return None
    if normalized[0] not in "[(" or normalized[-1] not in ")]":
        return None
    inner = normalized[1:-1].strip()
    if not inner:
        return None
    parts = _split_top_level_comma(inner)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return normalized[0], parts[0], parts[1], normalized[-1]


def supports_enable_thinking(tokenizer) -> bool:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return False
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return False
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    return "enable_thinking" in signature.parameters or has_var_kwargs


def build_prompt(tokenizer, question: str, enable_thinking: bool | None = False) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if supports_enable_thinking(tokenizer) and enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token and prompt.startswith(bos_token):
        prompt = prompt[len(bos_token) :]
    return prompt


def load_eval_records(dataset_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_parquet(dataset_path)
    required_columns = {"dataset_name", "sample_idx", "question", "gold_answer"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {dataset_path}: {sorted(missing)}")
    if limit is not None:
        df = df.iloc[:limit]
    return df.to_dict(orient="records")


class MathVerifier:
    def __init__(self):
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify

    def _verify_parse_equivalence(self, gold: str, pred: str) -> bool:
        gold_parsed = self._parse(gold)
        pred_parsed = self._parse(pred)
        if len(gold_parsed) == 0 or len(pred_parsed) == 0:
            return False
        return bool(self._verify(gold_parsed, pred_parsed))

    def _intervals_match(self, gold: str, pred: str) -> bool:
        gold_bounds = parse_interval_bounds(gold)
        pred_bounds = parse_interval_bounds(pred)
        if gold_bounds is None or pred_bounds is None:
            return False
        gold_left, gold_start, gold_end, gold_right = gold_bounds
        pred_left, pred_start, pred_end, pred_right = pred_bounds
        if gold_left != pred_left or gold_right != pred_right:
            return False
        return self._verify_parse_equivalence(gold_start, pred_start) and self._verify_parse_equivalence(gold_end, pred_end)

    def is_correct(self, gold: str, pred_text: str) -> bool:
        candidate = extract_candidate_answer(pred_text)
        if self._verify_parse_equivalence(gold, candidate):
            return True
        if self._intervals_match(gold, candidate):
            return True
        return False


def score_candidates(verifier: MathVerifier, gold: str, candidates: list[dict[str, Any]], eval_k: int) -> dict[str, Any]:
    k_used = max(1, min(eval_k, len(candidates))) if candidates else 0
    selected_text = candidates[0]["text"] if candidates else ""
    selected_len = candidates[0]["token_len"] if candidates else 0
    correct_flags: list[bool] = []
    any_correct = False

    for candidate in candidates:
        try:
            is_correct = verifier.is_correct(gold, candidate["text"])
        except Exception:
            is_correct = False
        correct_flags.append(bool(is_correct))
        if is_correct and not any_correct:
            any_correct = True
            selected_text = candidate["text"]
            selected_len = candidate["token_len"]

    correct_in_k = sum(correct_flags[:k_used]) if k_used > 0 else 0
    avg_at_k = (correct_in_k / k_used) if k_used > 0 else 0.0
    pass_at_k = 1.0 if any(correct_flags[:k_used]) else 0.0
    return {
        "selected_text": selected_text,
        "selected_len": selected_len,
        "is_correct": any_correct,
        "correct_flags": correct_flags,
        "avg_at_k": avg_at_k,
        "pass_at_k": pass_at_k,
        "k_used": k_used,
    }


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

