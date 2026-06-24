#!/usr/bin/env python
"""Build a DiffusionGemma-oriented SFT mix.

The mix keeps Fable terminal behavior, then adds structured JSON/tool-call
repair examples so block-diffusion self-correction is trained on tasks where it
should help.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FABLE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    {
        "name": "fable5_terminal",
        "path": FABLE_DIR / "datasets/fable5_lfm_sft_20260623.jsonl",
        "cap": 3000,
    },
    {
        "name": "glm52_terminal_toolmix",
        "path": FABLE_DIR / "datasets/glm52_chaser_terminal_toolmix_20260624.jsonl",
        "cap": 4000,
    },
    {
        "name": "hermes_agent_function_code",
        "path": FABLE_DIR / "datasets/hermes_agent_traces_chat_20260624.jsonl",
        "cap": 3000,
    },
    {
        "name": "phase2_reasoning_code",
        "path": FABLE_DIR / "datasets/phase2_reasoning_lfm_sft_20260623.jsonl",
        "cap": 300,
    },
]


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def clean_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    clean: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        content = stringify_content(message.get("content")).strip()
        if not content:
            continue
        clean.append({"role": str(role), "content": content})
    if len(clean) < 2 or clean[-1]["role"] != "assistant":
        return None
    return clean


def row_hash(messages: list[dict[str, str]], task: str) -> str:
    payload = {"task": task, "messages": messages}
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if "\u0400" <= char <= "\u04ff") / len(text)


def detect_final_kind(text: str) -> str:
    stripped = text.strip()
    if "<|tool_call_start|>" in stripped:
        return "lfm_tool_call"
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_object"
    if stripped.startswith("[") and stripped.endswith("]"):
        return "json_array"
    if "```" in stripped:
        return "code_fence"
    return "text"


def compact_context(messages: list[dict[str, str]], max_chars: int) -> str:
    context_messages = messages[:-1]
    chunks: list[str] = []
    total = 0
    for message in reversed(context_messages):
        chunk = f"{message['role']}:\n{message['content']}"
        total += len(chunk)
        chunks.append(chunk)
        if total >= max_chars:
            break
    return "\n\n".join(reversed(chunks))[-max_chars:]


def corrupt_structured(text: str, rng: random.Random) -> str:
    options: list[str] = []
    stripped = text.strip()
    if len(stripped) > 2:
        options.append(stripped[:-1])
    if "," in stripped:
        options.append(stripped.replace(",", "", 1))
    if '"' in stripped:
        options.append(stripped.replace('"', "", 1))
    if "<|tool_call_end|>" in stripped:
        options.append(stripped.replace("<|tool_call_end|>", "", 1))
    if not options:
        return stripped + "\n# malformed"
    return rng.choice(options)


def make_repair_row(messages: list[dict[str, str]], rng: random.Random, max_context_chars: int) -> list[dict[str, str]]:
    target = messages[-1]["content"].strip()
    malformed = corrupt_structured(target, rng)
    context = compact_context(messages, max_context_chars)
    return [
        {
            "role": "system",
            "content": (
                "You repair malformed structured assistant outputs. "
                "Return only the corrected JSON or tool-call text, with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Repair the malformed assistant output so it is syntactically valid and "
                "matches the intended action from the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Malformed assistant output:\n```text\n{malformed}\n```"
            ),
        },
        {"role": "assistant", "content": target},
    ]


def load_source(spec: dict[str, Any], rng: random.Random, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(spec["path"])
    stats: dict[str, Any] = {"path": str(path), "input": 0, "valid": 0, "kept": 0, "missing": 0}
    if not path.exists():
        stats["missing"] = 1
        return [], stats

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["input"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = clean_messages(raw)
            if messages is None:
                continue
            final_text = messages[-1]["content"]
            full_text = "\n".join(message["content"] for message in messages)
            if len(final_text) > args.max_final_chars or len(full_text) > args.max_total_chars:
                continue
            if cyrillic_ratio(final_text) > args.max_cyrillic:
                continue
            stats["valid"] += 1
            rows.append(
                {
                    "messages": messages,
                    "source": spec["name"],
                    "kind": detect_final_kind(final_text),
                    "uid": raw.get("mix_uid") or raw.get("uid") or raw.get("query_id") or "",
                }
            )

    cap = int(spec.get("cap") or 0)
    if cap > 0 and len(rows) > cap:
        rows = rng.sample(rows, cap)
    stats["kept"] = len(rows)
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FABLE_DIR / "datasets/diffusiongemma_strength_mix_20260624.jsonl"))
    parser.add_argument("--meta", default=str(FABLE_DIR / "datasets/diffusiongemma_strength_mix_20260624.meta.json"))
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-final-chars", type=int, default=12000)
    parser.add_argument("--max-total-chars", type=int, default=36000)
    parser.add_argument("--repair-ratio", type=float, default=0.80)
    parser.add_argument("--max-repair-context-chars", type=int, default=12000)
    parser.add_argument("--max-cyrillic", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    meta_path = Path(args.meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_stats: dict[str, Any] = {}
    kind_counts = Counter()
    task_counts = Counter()
    task_by_source: dict[str, Counter[str]] = defaultdict(Counter)

    for spec in DEFAULT_SOURCES:
        source_rows, stats = load_source(spec, rng, args)
        source_stats[spec["name"]] = stats
        for item in source_rows:
            original_messages = item["messages"]
            original_task = "original_code_or_tool_trace" if item["kind"] == "code_fence" else "original_terminal_or_tool_trace"
            for task, messages in [(original_task, original_messages)]:
                digest = row_hash(messages, task)
                if digest in seen:
                    continue
                seen.add(digest)
                kind_counts[item["kind"]] += 1
                task_counts[task] += 1
                task_by_source[item["source"]][task] += 1
                rows.append(
                    {
                        "messages": messages,
                        "mix_source": item["source"],
                        "mix_task": task,
                        "final_kind": item["kind"],
                        "source_uid": item["uid"],
                    }
                )

            if item["kind"] not in {"json_object", "json_array", "lfm_tool_call"}:
                continue
            if rng.random() > args.repair_ratio:
                continue
            repair_messages = make_repair_row(original_messages, rng, args.max_repair_context_chars)
            digest = row_hash(repair_messages, "structured_repair")
            if digest in seen:
                continue
            seen.add(digest)
            task_counts["structured_repair"] += 1
            task_by_source[item["source"]]["structured_repair"] += 1
            rows.append(
                {
                    "messages": repair_messages,
                    "mix_source": item["source"],
                    "mix_task": "structured_repair",
                    "final_kind": item["kind"],
                    "source_uid": item["uid"],
                }
            )

    rng.shuffle(rows)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "purpose": (
            "DiffusionGemma strength SFT mix: keep Fable terminal/tool behavior, "
            "add structured JSON/tool-call repair to exploit bidirectional denoising/self-correction."
        ),
        "output": str(output_path),
        "seed": args.seed,
        "repair_ratio": args.repair_ratio,
        "max_final_chars": args.max_final_chars,
        "max_total_chars": args.max_total_chars,
        "sources": source_stats,
        "total_rows": len(rows),
        "final_kind_counts": dict(kind_counts),
        "task_counts": dict(task_counts),
        "task_counts_by_source": {name: dict(counter) for name, counter in task_by_source.items()},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
