#!/usr/bin/env python
"""Build a Fabliq terminal/tool-call SFT mix for the GLM-5.2 chaser run.

The mix keeps the best TB2-lite ingredients from Fabliq, then adds local
Harness-1 JSON/action traces to strengthen tool-call discipline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = [
    {
        "name": "fable5_terminal",
        "path": "fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl",
        "cap": 0,
    },
    {
        "name": "phase2_reasoning",
        "path": "fable_distillation/datasets/phase2_reasoning_lfm_sft_20260623.jsonl",
        "cap": 0,
    },
    {
        "name": "harness_agentic_highrecall",
        "path": "/home/work/.data/harness1/sft_data/lfm25_agentic_sft_20260619_lfm25_highrecall_direct_refresh_v1.jsonl",
        "cap": 0,
    },
    {
        "name": "harness_mixed_highrecall",
        "path": "/home/work/.data/harness1/sft_data/lfm25_mixed_agentic_sft_20260619_lfm25_highrecall_direct_refresh_v1.jsonl",
        "cap": 4000,
    },
    {
        "name": "harness_direct_json_highrecall",
        "path": "/home/work/.data/harness1/sft_data/lfm25_direct_json_sft_20260619_lfm25_highrecall_direct_refresh_v1.jsonl",
        "cap": 0,
    },
    {
        "name": "harness_local_hidden_search",
        "path": "/home/work/.data/harness1/sft_data/local_hidden_search_agent_20260619_v1.jsonl",
        "cap": 2000,
    },
    {
        "name": "harness_agentic_hardcase",
        "path": "/home/work/.data/harness1/sft_data/lfm25_agentic_hardcase_20260619_v1.jsonl",
        "cap": 0,
    },
]


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    return cyr / len(text)


def row_hash(messages: list[dict[str, Any]]) -> str:
    normalized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def valid_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    clean: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        if not isinstance(content, str) or not content.strip():
            continue
        clean.append({"role": role, "content": content.strip()})
    if len(clean) < 2 or clean[-1]["role"] != "assistant":
        return None
    return clean


def detect_final_kind(content: str) -> str:
    stripped = content.strip()
    if "<|tool_call_start|>" in stripped:
        return "lfm_tool_call"
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_action"
    if stripped.startswith("[") and stripped.endswith("]"):
        return "json_array"
    return "text"


def load_source(spec: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(spec["path"])
    stats: dict[str, Any] = {
        "path": str(path),
        "input": 0,
        "valid": 0,
        "missing": 0,
        "sampled": 0,
        "invalid": 0,
    }
    if not path.exists():
        stats["missing"] = 1
        return [], stats

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["input"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            messages = valid_messages(obj)
            if messages is None:
                stats["invalid"] += 1
                continue
            obj = dict(obj)
            obj["messages"] = messages
            obj["mix_source"] = spec["name"]
            rows.append(obj)
            stats["valid"] += 1

    cap = int(spec.get("cap") or 0)
    if cap > 0 and len(rows) > cap:
        rows = rng.sample(rows, cap)
    stats["sampled"] = len(rows)
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.jsonl")
    parser.add_argument("--meta", default="fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.meta.json")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-cyrillic", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    meta_path = Path(args.meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    mixed_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    final_kind = Counter()
    source_final_kind: dict[str, Counter[str]] = defaultdict(Counter)
    skipped = Counter()

    for spec in DEFAULT_SOURCES:
        rows, stats = load_source(spec, rng)
        kept = 0
        for row in rows:
            messages = row["messages"]
            assistant_text = " ".join(msg["content"] for msg in messages if msg["role"] == "assistant")
            if cyrillic_ratio(assistant_text) > args.max_cyrillic:
                skipped["cyrillic"] += 1
                continue
            digest = row_hash(messages)
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(digest)
            kind = detect_final_kind(messages[-1]["content"])
            final_kind[kind] += 1
            source_final_kind[spec["name"]][kind] += 1
            row["mix_uid"] = f"{spec['name']}:{row.get('uid') or row.get('query_id') or digest[:16]}"
            mixed_rows.append(row)
            kept += 1
        stats["kept_after_dedup_filter"] = kept
        source_stats[spec["name"]] = stats

    rng.shuffle(mixed_rows)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in mixed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "purpose": "GLM-5.2 chaser SFT mix: preserve Fabliq terminal behavior and add JSON/tool-call traces.",
        "output": str(output_path),
        "seed": args.seed,
        "max_cyrillic": args.max_cyrillic,
        "sources": source_stats,
        "total_output_rows": len(mixed_rows),
        "skipped": dict(skipped),
        "final_kind": dict(final_kind),
        "final_kind_by_source": {name: dict(counter) for name, counter in source_final_kind.items()},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
