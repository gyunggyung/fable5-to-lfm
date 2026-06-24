#!/usr/bin/env python3
"""Run long-output prompt probes against an OpenAI-compatible chat server."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--extra-body-json", default="{}")
    args = parser.parse_args()

    rows = load_rows(Path(args.prompt_jsonl), args.limit)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total_completion_tokens = 0
    items = []
    gen_start = time.time()
    for row in rows:
        start = time.time()
        response = client.chat.completions.create(
            model=args.model,
            messages=row["messages"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body=extra_body or None,
        )
        elapsed = time.time() - start
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""
        usage = getattr(response, "usage", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_completion_tokens += completion_tokens
        items.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "elapsed_sec": round(elapsed, 3),
                "completion_tokens": completion_tokens,
                "completion_tokens_per_sec": round(completion_tokens / max(elapsed, 1e-9), 2),
                "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                "preview": (text or "")[:1600],
            }
        )

    gen_time = time.time() - gen_start
    result = {
        "model": args.model,
        "timestamp": datetime.utcnow().isoformat(),
        "rows": len(rows),
        "gen_time_sec": round(gen_time, 2),
        "completion_tokens": total_completion_tokens,
        "completion_tokens_per_sec": round(total_completion_tokens / max(gen_time, 1e-9), 2),
        "avg_sec_per_prompt": round(gen_time / max(len(rows), 1), 3),
        "settings": vars(args),
        "items": items,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "completion_tokens_per_sec": result["completion_tokens_per_sec"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
