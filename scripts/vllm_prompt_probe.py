#!/usr/bin/env python3
"""Run a lightweight vLLM prompt probe and record speed/shape metrics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]], chat_template_kwargs: dict[str, Any]) -> list[str]:
    prompts = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list):
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            )
        else:
            prompts.append(str(row.get("prompt", "")))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--prompt-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument("--engine-kwargs-json", default="{}")
    parser.add_argument("--sampling-params-json", default="{}")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_jsonl)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
    engine_kwargs = json.loads(args.engine_kwargs_json)
    sampling_kwargs = json.loads(args.sampling_params_json)
    if not isinstance(chat_template_kwargs, dict) or not isinstance(engine_kwargs, dict) or not isinstance(sampling_kwargs, dict):
        raise ValueError("JSON kwargs must decode to objects")

    rows = load_rows(prompt_path, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model, trust_remote_code=True)
    prompts = build_prompts(tokenizer, rows, chat_template_kwargs)

    load_start = time.time()
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer_path or args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        **engine_kwargs,
    )
    load_sec = time.time() - load_start

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        **sampling_kwargs,
    )
    gen_start = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    gen_sec = time.time() - gen_start

    items = []
    total_output_tokens = 0
    for row, prompt, output in zip(rows, prompts, outputs):
        completion = output.outputs[0] if output.outputs else None
        token_ids = list(getattr(completion, "token_ids", []) or []) if completion else []
        text = completion.text if completion else ""
        total_output_tokens += len(token_ids)
        items.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "prompt_chars": len(prompt),
                "output_tokens": len(token_ids),
                "output_chars": len(text),
                "finish_reason": getattr(completion, "finish_reason", None) if completion else None,
                "preview": text[:1600],
            }
        )

    result = {
        "model": args.model,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_jsonl": str(prompt_path),
        "rows": len(rows),
        "load_time_sec": round(load_sec, 2),
        "gen_time_sec": round(gen_sec, 2),
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_sec": round(total_output_tokens / max(gen_sec, 1e-9), 2),
        "avg_sec_per_prompt": round(gen_sec / max(len(rows), 1), 3),
        "settings": vars(args),
        "items": items,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("output_tokens_per_sec", "avg_sec_per_prompt", "output_path") if k in result} | {"output_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
