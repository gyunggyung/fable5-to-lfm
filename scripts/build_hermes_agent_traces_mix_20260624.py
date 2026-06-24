#!/usr/bin/env python
"""Convert Hermes agent reasoning traces into Fabliq chat SFT JSONL.

The upstream dataset is ShareGPT-like parquet. This builder keeps tool-call
trajectories but normalizes roles to system/user/assistant by default so Gemma,
Qwen, and LFM chat templates can tokenize the result without custom tool roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DEFAULT_FILES = {
    "kimi": "data/kimi/train.parquet",
    "glm-5.1": "data/glm-5.1/train.parquet",
}

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lambda/hermes-agent-reasoning-traces")
    parser.add_argument("--output", default="fable_distillation/datasets/hermes_agent_traces_chat_20260624.jsonl")
    parser.add_argument("--meta", default="fable_distillation/datasets/hermes_agent_traces_chat_20260624.meta.json")
    parser.add_argument("--configs", default="kimi,glm-5.1")
    parser.add_argument("--max-rows-per-config", type=int, default=0)
    parser.add_argument("--max-output-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument(
        "--tool-role-mode",
        choices=("user", "tool", "drop"),
        default="user",
        help="How to encode tool execution results in output messages.",
    )
    parser.add_argument(
        "--inject-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject upstream JSON tool definitions into a system message.",
    )
    return parser.parse_args()


def stable_hash(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False).strip()


def add_tool_system(messages: list[dict[str, str]], tools: str) -> list[dict[str, str]]:
    tools = tools.strip()
    if not tools:
        return messages
    tool_text = "Available tools are encoded as JSON. Use them only when the task requires tool execution.\n\n" + tools
    if messages and messages[0]["role"] == "system":
        updated = dict(messages[0])
        updated["content"] = updated["content"].rstrip() + "\n\n" + tool_text
        return [updated] + messages[1:]
    return [{"role": "system", "content": tool_text}] + messages


def convert_conversation(
    row: dict[str, Any],
    *,
    source_name: str,
    tool_role_mode: str,
    inject_tools: bool,
) -> dict[str, Any] | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None

    messages: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = ROLE_MAP.get(str(turn.get("from", "")).strip())
        content = normalize_text(turn.get("value"))
        if not role or not content:
            continue
        if role == "tool":
            if tool_role_mode == "drop":
                continue
            if tool_role_mode == "user":
                messages.append({"role": "user", "content": "Tool result:\n" + content})
            else:
                messages.append({"role": "tool", "content": content})
        else:
            messages.append({"role": role, "content": content})

    if inject_tools:
        messages = add_tool_system(messages, normalize_text(row.get("tools")))

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None

    return {
        "messages": messages,
        "source_dataset": "lambda/hermes-agent-reasoning-traces",
        "trace_source_model": source_name,
        "trace_id": row.get("id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": row.get("task"),
    }


def load_rows(args: argparse.Namespace, config_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filename = DEFAULT_FILES[config_name]
    local_path = hf_hub_download(args.repo_id, filename=filename, repo_type="dataset")
    table = pq.read_table(local_path)
    raw_rows = table.to_pylist()
    if args.max_rows_per_config and len(raw_rows) > args.max_rows_per_config:
        rng = random.Random(args.seed + len(config_name))
        raw_rows = rng.sample(raw_rows, args.max_rows_per_config)

    converted: list[dict[str, Any]] = []
    skipped = Counter()
    for row in raw_rows:
        item = convert_conversation(
            row,
            source_name=config_name,
            tool_role_mode=args.tool_role_mode,
            inject_tools=args.inject_tools,
        )
        if item is None:
            skipped["invalid_or_no_final_assistant"] += 1
            continue
        converted.append(item)

    return converted, {
        "filename": filename,
        "local_path": local_path,
        "input_rows": len(raw_rows),
        "converted_rows": len(converted),
        "skipped": dict(skipped),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    meta_path = Path(args.meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = sorted(set(configs) - set(DEFAULT_FILES))
    if unknown:
        raise ValueError(f"unknown configs: {unknown}; known={sorted(DEFAULT_FILES)}")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    seen: set[str] = set()
    skipped = Counter()

    for config_name in configs:
        source_rows, stats = load_rows(args, config_name)
        kept = 0
        for row in source_rows:
            digest = stable_hash(row["messages"])
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(digest)
            row["mix_uid"] = f"hermes:{config_name}:{row.get('trace_id') or digest[:16]}"
            rows.append(row)
            kept += 1
        stats["kept_after_dedup"] = kept
        source_stats[config_name] = stats

    rng.shuffle(rows)
    if args.max_output_rows and len(rows) > args.max_output_rows:
        rows = rows[: args.max_output_rows]

    role_counter = Counter()
    source_counter = Counter()
    for row in rows:
        source_counter[row["trace_source_model"]] += 1
        role_counter.update(message["role"] for message in row["messages"])

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "output": str(output_path),
        "repo_id": args.repo_id,
        "configs": configs,
        "tool_role_mode": args.tool_role_mode,
        "inject_tools": args.inject_tools,
        "max_rows_per_config": args.max_rows_per_config,
        "max_output_rows": args.max_output_rows,
        "seed": args.seed,
        "total_output_rows": len(rows),
        "rows_by_source": dict(source_counter),
        "roles": dict(role_counter),
        "skipped": dict(skipped),
        "sources": source_stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
