#!/usr/bin/env python3
"""Convert the official-agentic mix to ms-swift's standard agent SFT JSONL.

ms-swift accepts a standard `messages` field and can train agent traces with
`tool_call` / `tool_response` roles plus a JSON-string `tools` field. The source
mix is OpenAI-style chat data, so this converter normalizes assistant tool calls
into explicit assistant/tool_call/tool_response turns while preserving ordinary
assistant text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    return {
        "name": fn.get("name") or call.get("name") or "",
        "arguments": parse_arguments(fn.get("arguments", call.get("arguments", {}))),
    }


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
    out_messages: list[dict[str, Any]] = []
    source_messages = row.get("messages") or []
    if not isinstance(source_messages, list):
        return None

    for message in source_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role in {"system", "user"}:
            if content:
                out_messages.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            if content:
                out_messages.append({"role": "assistant", "content": str(content)})
            for call in message.get("tool_calls") or []:
                out_messages.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(normalize_tool_call(call), ensure_ascii=False),
                    }
                )
            continue

        if role == "tool":
            if content is not None:
                out_messages.append({"role": "tool_response", "content": str(content)})
            continue

        if role in {"tool_call", "tool_response"}:
            if content is not None:
                out_messages.append({"role": role, "content": str(content)})

    if not any(msg["role"] == "assistant" for msg in out_messages):
        return None

    converted: dict[str, Any] = {"messages": out_messages}
    tools = row.get("tools")
    if tools:
        converted["tools"] = json.dumps(tools, ensure_ascii=False)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = written = skipped = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            converted = convert_row(row)
            if converted is None:
                skipped += 1
                continue
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            if args.max_rows and written >= args.max_rows:
                break

    meta = {
        "source": str(args.input),
        "output": str(args.output),
        "total_seen": total,
        "written": written,
        "skipped": skipped,
        "format": "ms-swift standard messages/tools JSONL",
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
