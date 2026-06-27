#!/usr/bin/env python
"""Convert Fable/OpenAI-style message JSONL to Axolotl completion JSONL.

Axolotl's chat_template path may not preserve OpenAI tool_calls/tool messages
for this GLM experiment. This keeps every turn as explicit ChatML text so the
model learns the Fable/tool-call style traces without relying on template
support for tool_calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/official_agentic_sft_mix_20260627.jsonl")
    parser.add_argument("--output", default="datasets/official_agentic_sft_mix_20260627.axolotl_chatml.jsonl")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "user")
    content = message.get("content")
    parts: list[str] = []
    if content:
        parts.append(str(content) if isinstance(content, str) else compact_json(content))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        parts.append("<tool_calls>" + compact_json(tool_calls) + "</tool_calls>")
    if role == "tool":
        name = message.get("name")
        tool_call_id = message.get("tool_call_id")
        meta = {k: v for k, v in {"name": name, "tool_call_id": tool_call_id}.items() if v}
        if meta:
            parts.insert(0, "<tool_meta>" + compact_json(meta) + "</tool_meta>")
    return f"<|im_start|>{role}\n" + "\n".join(parts) + "<|im_end|>\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if args.max_rows and rows >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            text = "".join(render_message(message) for message in messages if isinstance(message, dict))
            if text:
                sink.write(compact_json({"text": text}) + "\n")
                rows += 1
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(input_path),
                "output": str(output_path),
                "rows": rows,
                "format": "axolotl completion JSONL with text field, ChatML-style turns",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote rows={rows} output={output_path} meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
