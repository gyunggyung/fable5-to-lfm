#!/usr/bin/env python3
"""TB2-lite replay evaluator for OpenAI-compatible chat servers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [{"role": "user", "content": str(row.get("prompt", ""))}]
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            role = "user"
            content = f"Tool result:\n{content}"
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            normalized.append({"role": role, "content": content})
    return normalized or [{"role": "user", "content": str(row.get("prompt", ""))}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--extra-body-json", default="{}")
    args = parser.parse_args()

    rows = load_rows(Path(args.eval_path), args.limit or None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or args.model.rstrip("/").split("/")[-1]
    out_path = output_dir / f"{model_short}.json"
    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    per_step = []
    gen_start = time.time()
    total_completion_tokens = 0
    for idx, row in enumerate(rows, start=1):
        start = time.time()
        response = client.chat.completions.create(
            model=args.model,
            messages=row_messages(row),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body=extra_body or None,
        )
        elapsed = time.time() - start
        choice = response.choices[0] if response.choices else None
        pred_text = choice.message.content if choice and choice.message else ""
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "completion_tokens", None):
            total_completion_tokens += int(usage.completion_tokens)

        pred = parse_prediction(pred_text or "")
        ref = parse_prediction(row["ref_raw"])
        first_exact, precision, recall, f1 = score_commands(pred["command_units"], ref["command_units"])
        pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
        per_step.append(
            {
                "task_id": row["task_id"],
                "sample_idx": row["sample_idx"],
                "step_idx": row["step_idx"],
                "bucket": step_bucket(row["step_idx"]),
                "source_group": row["source_group"],
                "valid_json": pred["valid_json"],
                "has_analysis": pred["has_analysis"],
                "has_plan": pred["has_plan"],
                "ref_task_complete": bool(ref["task_complete"]),
                "pred_task_complete": pred["task_complete"],
                "pred_task_complete_true": pred_complete_true,
                "ref_command_units": ref["command_units"],
                "pred_command_units": pred["command_units"],
                "first_cmd_exact": first_exact,
                "command_precision": round(precision, 4),
                "command_recall": round(recall, 4),
                "command_f1": round(f1, 4),
                "pred_preview": (pred_text or "")[:1200],
                "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                "elapsed_sec": round(elapsed, 3),
            }
        )
        print(json.dumps({"idx": idx, "steps": len(rows), "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False), flush=True)

    gen_time = time.time() - gen_start
    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model,
        "model_short": model_short,
        "eval_path": args.eval_path,
        "timestamp": datetime.utcnow().isoformat(),
        "gen_time_sec": round(gen_time, 2),
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "completion_tokens": total_completion_tokens,
        "completion_tokens_per_sec": round(total_completion_tokens / max(gen_time, 1e-9), 2),
        "sampling": {
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "extra_body_json": args.extra_body_json,
        },
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
