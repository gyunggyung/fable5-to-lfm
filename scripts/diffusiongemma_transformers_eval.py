#!/usr/bin/env python3
"""DiffusionGemma Transformers fallback evaluator.

Use this when the official vLLM Gemma Docker image is unavailable. It is slower
than vLLM serving, but still tests whether the dLLM model itself is useful.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def load_jsonl(path: str, limit: int = 0, shard_index: int = 0, shard_count: int = 1) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if shard_count > 1 and row_idx % shard_count != shard_index:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [{"role": "user", "content": str(row.get("prompt", ""))}]
    clean = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            role = "user"
            content = f"Tool result:\n{content}"
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            clean.append({"role": role, "content": content})
    return clean or [{"role": "user", "content": str(row.get("prompt", ""))}]


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_diagnostics() -> dict[str, Any]:
    info: dict[str, Any] = {
        "event": "cuda_diagnostics",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def normalize_decoded_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def strip_decoded_prompt(text: str, prompt_text: str) -> str:
    for candidate in (prompt_text, prompt_text.strip()):
        if candidate and text.startswith(candidate):
            text = text[len(candidate):]
            break
    if "\nmodel\n" in text:
        text = text.rsplit("\nmodel\n", 1)[-1]
    elif text.startswith("model\n"):
        text = text[len("model\n"):]
    return text.strip()


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    processor: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
) -> tuple[str, int, float]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model_device(model))
    prompt_len = int(inputs["input_ids"].shape[-1])
    start = time.time()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.time() - start
    output_ids = output[0]
    input_ids = inputs["input_ids"][0]
    if output_ids.shape[-1] > prompt_len and torch.equal(output_ids[:prompt_len], input_ids):
        generated = output_ids[prompt_len:]
    else:
        generated = output_ids
    text = normalize_decoded_text(processor.decode(generated, skip_special_tokens=True))
    if not text.strip() and generated.numel():
        text = normalize_decoded_text(processor.decode(generated, skip_special_tokens=False))
    prompt_text = normalize_decoded_text(processor.decode(input_ids, skip_special_tokens=True))
    text = strip_decoded_prompt(text, prompt_text)
    return text.strip(), int(generated.numel()), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--eval-path", default="")
    parser.add_argument("--prompt-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-short", default="diffusiongemma-26b-a4b-it-transformers")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probe-limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps(cuda_diagnostics(), ensure_ascii=False), flush=True)
    device_map: str | dict[str, int]
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = "auto"

    load_start = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model,
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    load_time = time.time() - load_start
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "load_time_sec": round(load_time, 2),
                "device": str(model_device(model)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    result: dict[str, Any] = {
        "model": args.model,
        "model_short": args.model_short,
        "timestamp": datetime.utcnow().isoformat(),
        "backend": "transformers",
        "load_time_sec": round(load_time, 2),
        "settings": vars(args),
    }

    if args.eval_path:
        rows = load_jsonl(args.eval_path, args.limit, args.shard_index, args.shard_count)
        per_step = []
        total_tokens = 0
        start_all = time.time()
        for idx, row in enumerate(rows, start=1):
            pred_text, tokens, elapsed = generate_one(
                model,
                processor,
                normalize_messages(row),
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
            total_tokens += tokens
            pred = parse_prediction(pred_text)
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
                    "pred_preview": pred_text[:1200],
                    "elapsed_sec": round(elapsed, 3),
                    "output_tokens": tokens,
                }
            )
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "steps": len(rows),
                        "elapsed_sec": round(elapsed, 3),
                        "output_tokens": tokens,
                        "blank_output": not bool(pred_text.strip()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        gen_time = time.time() - start_all
        result["tb2"] = {
            "eval_path": args.eval_path,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "gen_time_sec": round(gen_time, 2),
            "output_tokens": total_tokens,
            "output_tokens_per_sec": round(total_tokens / max(gen_time, 1e-9), 2),
            "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
            "aggregate": aggregate_scores(per_step),
            "per_step": per_step,
        }
        result["aggregate"] = result["tb2"]["aggregate"]
        result["per_step"] = per_step

    if args.prompt_jsonl:
        rows = load_jsonl(args.prompt_jsonl, args.probe_limit)
        items = []
        total_tokens = 0
        start_all = time.time()
        for row in rows:
            text, tokens, elapsed = generate_one(
                model,
                processor,
                normalize_messages(row),
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
            total_tokens += tokens
            items.append(
                {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "elapsed_sec": round(elapsed, 3),
                    "output_tokens": tokens,
                    "output_tokens_per_sec": round(tokens / max(elapsed, 1e-9), 2),
                    "preview": text[:1600],
                }
            )
        gen_time = time.time() - start_all
        result["probe"] = {
            "prompt_jsonl": args.prompt_jsonl,
            "gen_time_sec": round(gen_time, 2),
            "output_tokens": total_tokens,
            "output_tokens_per_sec": round(total_tokens / max(gen_time, 1e-9), 2),
            "avg_sec_per_prompt": round(gen_time / max(len(rows), 1), 3),
            "items": items,
        }

    out_path = output_dir / f"{args.model_short}.transformers.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"output_path": str(out_path), "backend": "transformers"}
    if "tb2" in result:
        summary["score"] = result["tb2"]["aggregate"]["next_action_score"]
    if "probe" in result:
        summary["probe_tps"] = result["probe"]["output_tokens_per_sec"]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
