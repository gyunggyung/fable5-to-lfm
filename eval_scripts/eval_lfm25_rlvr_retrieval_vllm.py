#!/usr/bin/env python
"""Evaluate Harness-style retrieval-curation adapters through vLLM HTTP."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--served-model-name", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8133/v1")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--stop", action="append", default=["<|im_end|>"])
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("doc_id") or item.get("docid") or item.get("id")
            pieces.append(str(item))
    else:
        pieces = [str(value)]
    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        doc = str(piece).strip().strip("\"'`[](){}")
        if not doc or doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
    return out


def parse_curated_ids(text: str) -> tuple[list[str], bool]:
    parsed = extract_json_object(text)
    if parsed is None:
        return [], False
    for key in ("curated_doc_ids", "doc_ids", "selected_doc_ids", "evidence_doc_ids"):
        if key in parsed:
            return normalize_ids(parsed.get(key)), True
    return [], False


def extract_candidate_ids_from_text(text: str, candidate_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cid in sorted((str(item) for item in candidate_ids), key=len, reverse=True):
        if cid and cid not in seen and cid in text:
            seen.add(cid)
            out.append(cid)
    return out


def f_beta(recall: float, precision: float, beta: float = 2.0) -> float:
    if recall <= 0.0 and precision <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def score_completion(text: str, row: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = [str(x) for x in row.get("candidate_doc_ids", [])]
    curated_ids, valid_json = parse_curated_ids(text)
    used_fallback = False
    if not valid_json:
        fallback_ids = extract_candidate_ids_from_text(text, candidate_ids)
        if fallback_ids:
            curated_ids = fallback_ids
            used_fallback = True
    gold = set(str(x) for x in row.get("gold_doc_ids", []))
    answer = set(str(x) for x in row.get("answer_doc_ids", [])) or gold
    candidates = set(candidate_ids)
    selected = [doc for doc in curated_ids if doc in candidates]
    selected_set = set(selected)
    invalid_count = max(0, len(curated_ids) - len(selected))

    recall = len(selected_set & gold) / max(len(gold), 1)
    precision = len(selected_set & gold) / max(len(selected_set), 1)
    answer_recall = len(selected_set & answer) / max(len(answer), 1)
    f2 = f_beta(recall, precision, beta=2.0)
    over_select = max(0, len(selected_set) - max(len(gold) + 3, 8))

    if not curated_ids and not used_fallback:
        retrieval_reward = -1.0
    elif not selected_set:
        retrieval_reward = -0.5 - 0.05 * invalid_count
    else:
        retrieval_reward = (
            0.65 * f2
            + 0.75 * recall
            + 0.85 * answer_recall
            - 0.05 * invalid_count
            - 0.025 * over_select
        )
        if not valid_json:
            retrieval_reward -= 0.45
        retrieval_reward = max(-1.0, retrieval_reward)

    strict_json_reward = -0.5
    parsed = extract_json_object(text)
    if parsed is not None:
        strict_json_reward = 0.25
        if isinstance(parsed.get("reasoning"), str) and 8 <= len(parsed["reasoning"]) <= 600:
            strict_json_reward += 0.15
        if curated_ids:
            strict_json_reward += 0.15
        if text.strip().startswith("{") and text.strip().endswith("}"):
            strict_json_reward += 0.10

    return {
        "valid_json": bool(valid_json),
        "used_fallback": bool(used_fallback),
        "selected_count": len(selected_set),
        "invalid_count": invalid_count,
        "gold_count": len(gold),
        "recall": recall,
        "precision": precision,
        "answer_recall": answer_recall,
        "f2": f2,
        "all_gold_found": recall >= 1.0,
        "retrieval_reward": retrieval_reward,
        "strict_json_reward": strict_json_reward,
        "total_reward": retrieval_reward + strict_json_reward,
        "curated_doc_ids": curated_ids,
    }


def apply_chat_template_text(tokenizer: Any, messages: list[dict[str, str]], disable_thinking: bool = False) -> str:
    try:
        kwargs = {"enable_thinking": False} if disable_thinking else {}
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        kwargs = {"enable_thinking": False} if disable_thinking else {}
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **kwargs,
        )
        return tokenizer.decode(ids, skip_special_tokens=False)


def truncate_prompt(tokenizer: Any, prompt: str, max_prompt_length: int) -> tuple[str, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if max_prompt_length > 0 and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return prompt, len(prompt_ids)


def generate(base_url: str, payload: dict[str, Any], timeout: float) -> str:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {body[:2000]}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM response had no choices: {data}")
    return str(choices[0].get("text") or "").strip()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scored_rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for eval_idx, row_idx in enumerate(indices, start=1):
            row = dict(dataset[row_idx])
            prompt, prompt_tokens = truncate_prompt(
                tokenizer,
                apply_chat_template_text(tokenizer, row["prompt"], disable_thinking=args.disable_thinking),
                args.max_prompt_length,
            )
            payload: dict[str, Any] = {
                "model": args.served_model_name,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed + eval_idx,
            }
            if args.min_p > 0:
                payload["min_p"] = args.min_p
            stops = [stop for stop in args.stop if stop]
            if stops:
                payload["stop"] = stops
            completion = generate(args.vllm_base_url, payload, args.request_timeout)
            metrics = score_completion(completion, row)
            record = {
                "eval_idx": eval_idx,
                "row_idx": row_idx,
                "query_id": row.get("query_id"),
                "source": row.get("source"),
                "prompt_tokens": prompt_tokens,
                "completion": completion,
                **metrics,
            }
            scored_rows.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if eval_idx % 10 == 0:
                print(
                    json.dumps(
                        {
                            "eval_idx": eval_idx,
                            "mean_total_reward": mean(scored_rows, "total_reward"),
                            "mean_recall": mean(scored_rows, "recall"),
                            "valid_json_rate": mean(scored_rows, "valid_json"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "dataset_path": args.dataset_path,
        "served_model_name": args.served_model_name,
        "vllm_base_url": args.vllm_base_url,
        "count": len(scored_rows),
        "mean_total_reward": mean(scored_rows, "total_reward"),
        "mean_retrieval_reward": mean(scored_rows, "retrieval_reward"),
        "mean_strict_json_reward": mean(scored_rows, "strict_json_reward"),
        "valid_json_rate": mean(scored_rows, "valid_json"),
        "fallback_rate": mean(scored_rows, "used_fallback"),
        "mean_recall": mean(scored_rows, "recall"),
        "mean_precision": mean(scored_rows, "precision"),
        "mean_answer_recall": mean(scored_rows, "answer_recall"),
        "mean_f2": mean(scored_rows, "f2"),
        "all_gold_found_rate": mean(scored_rows, "all_gold_found"),
        "mean_selected_count": mean(scored_rows, "selected_count"),
        "mean_invalid_count": mean(scored_rows, "invalid_count"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
