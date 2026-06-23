#!/usr/bin/env python
"""Fable-5-traces (Glint) → LFM ToolBench-style SFT JSONL 변환.

Glint 원본 (context/cot/output/output_type 구조화) → LFM 공식 tool-use 멀티턴 형식.
마지막 assistant turn만 학습 대상 (이전 context는 mask).

출력 포맷: Liquid-CLI에서 쓰던 LFM conversations JSONL
  {"messages": [{"role":"system","content":...}, {"role":"user",...}, ...]}

LFM tool-call 공식 형식:
  <|tool_call_start|>[ToolName(arg1='val1', arg2=val2)]<|tool_call_end|>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are an agentic coding assistant. Read the conversation history and tool results, "
    "think step by step inside <think>...</think>, then either call a tool using "
    "<|tool_call_start|>[ToolName(arg=value)]<|tool_call_end|> or respond with text. "
    "Use available tools (Bash, Edit, Read, Write, Glob, Grep, WebSearch, WebFetch, etc.) "
    "to accomplish the user's task. Be concise but thorough."
)


SLASH_META_PATTERNS = [
    r"<local-command-caveat>.*?</local-command-caveat>",
    r"<command-name>.*?</command-name>",
    r"<command-message>.*?</command-message>",
    r"<command-args>.*?</command-args>",
    r"<local-command-stdout>.*?</local-command-stdout>",
    r"<local-command-stderr>.*?</local-command-stderr>",
]


def clean_meta(text: str) -> str:
    for pat in SLASH_META_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_context(context: str) -> list[dict[str, str]]:
    """Glint context string → list of {role, content}.

    Context 형식:
      USER: {text}
      USER: {text}
      ASSISTANT (message): {text}
      USER: {text}
      ASSISTANT (message): {text}
      ...
    """
    messages: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?m)^(USER|ASSISTANT \(message\)):\s*(.*?)(?=\n(?:USER|ASSISTANT \(message\)):|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(context):
        role_raw = m.group(1)
        content = m.group(2)
        content = clean_meta(content)
        if not content:
            continue
        role = "user" if role_raw == "USER" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def format_arg_value(v: Any) -> str:
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        return f"'{escaped}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "null"
    return json.dumps(v, ensure_ascii=False)


def format_tool_call(output: dict[str, Any]) -> str:
    """{'tool': 'Read', 'input': {...}} → <|tool_call_start|>[Read(file_path='...')]<|tool_call_end|>"""
    name = output.get("tool") or output.get("name") or "unknown"
    args = output.get("input") or output.get("arguments") or {}
    if isinstance(args, dict):
        arg_str = ", ".join(f"{k}={format_arg_value(v)}" for k, v in args.items())
    elif isinstance(args, str) and args:
        arg_str = args
    else:
        arg_str = ""
    return f"<|tool_call_start|>[{name}({arg_str})]<|tool_call_end|>"


def build_assistant_turn(row: dict[str, Any]) -> dict[str, str]:
    cot = (row.get("cot") or "").strip()
    output = row.get("output")
    output_type = row.get("output_type", "text")

    parts = []
    if cot:
        parts.append(f"<think>\n{cot}\n</think>")

    if output_type == "tool_use" and isinstance(output, dict):
        parts.append(format_tool_call(output))
    elif isinstance(output, str):
        parts.append(output)
    elif isinstance(output, dict):
        text_field = output.get("text") or output.get("content")
        if text_field:
            parts.append(text_field)
        else:
            parts.append(json.dumps(output, ensure_ascii=False))

    return {"role": "assistant", "content": "\n\n".join(parts)}


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = parse_context(row.get("context", ""))
        messages.extend(history)
        messages.append(build_assistant_turn(row))
        if len(messages) < 3:
            return None
        return {
            "uid": row.get("uid", ""),
            "session": row.get("session", ""),
            "output_type": row.get("output_type", ""),
            "messages": messages,
        }
    except Exception as e:
        return {"error": str(e), "uid": row.get("uid", "")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="fable_distillation/datasets/Fable-5-traces/fable5_cot_merged.jsonl")
    p.add_argument("--output", default="fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl")
    p.add_argument("--meta", default="fable_distillation/datasets/fable5_lfm_sft_20260623.meta.json")
    p.add_argument("--show-samples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        rows = [json.loads(line) for line in f]

    from collections import Counter
    stats = {
        "input_rows": len(rows),
        "output_rows": 0,
        "skipped_error": 0,
        "skipped_short": 0,
        "output_type_dist": Counter(),
        "tool_dist": Counter(),
        "history_turns": [],
    }

    samples = []
    with out_path.open("w") as f:
        for row in rows:
            result = convert_row(row)
            if result is None:
                stats["skipped_short"] += 1
                continue
            if "error" in result:
                stats["skipped_error"] += 1
                continue
            stats["output_rows"] += 1
            stats["output_type_dist"][result["output_type"]] += 1
            stats["history_turns"].append(len(result["messages"]) - 2)
            if row.get("output_type") == "tool_use":
                out = row.get("output", {})
                if isinstance(out, dict):
                    stats["tool_dist"][out.get("tool", "unknown")] += 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if len(samples) < args.show_samples:
                samples.append(result)

    import numpy as np
    stats_summary = {
        "input_rows": stats["input_rows"],
        "output_rows": stats["output_rows"],
        "skipped_error": stats["skipped_error"],
        "skipped_short": stats["skipped_short"],
        "output_type_dist": dict(stats["output_type_dist"]),
        "tool_dist_top20": dict(stats["tool_dist"].most_common(20)),
        "history_turns_per_row": {
            "median": int(np.median(stats["history_turns"])) if stats["history_turns"] else 0,
            "mean": float(np.mean(stats["history_turns"])) if stats["history_turns"] else 0,
            "max": int(max(stats["history_turns"])) if stats["history_turns"] else 0,
        },
        "system_prompt": SYSTEM_PROMPT,
        "tool_call_format": "<|tool_call_start|>[ToolName(arg='value')]<|tool_call_end|>",
    }
    meta_path.write_text(json.dumps(stats_summary, ensure_ascii=False, indent=2))
    print(json.dumps(stats_summary, ensure_ascii=False, indent=2))

    if samples:
        print("\n=== 첫 샘플 ===")
        s = samples[0]
        print(f"uid: {s['uid']}")
        print(f"output_type: {s['output_type']}")
        print(f"messages ({len(s['messages'])} turns):")
        for i, m in enumerate(s["messages"]):
            content_preview = m["content"][:300].replace("\n", " ")
            print(f"  [{i}] {m['role']}: {content_preview}{'...' if len(m['content']) > 300 else ''}")


if __name__ == "__main__":
    main()
