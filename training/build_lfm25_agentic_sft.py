#!/usr/bin/env python
"""Build local multi-turn Harness-style agent SFT rows.

This is a bridge from the simple JSON curation task to the paper's stateful
tool-agent setup.  It uses the already-materialized local candidate pools and
creates supervised trajectories with search, review/verify, curate, and
end_search actions.  The rows stay in normal chat-message format so the existing
local LoRA SFT trainer can consume them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


POOL_RE = re.compile(
    r"\[(?P<idx>\d+)\]\s+doc_id:\s+(?P<doc_id>[^\n]+)\n"
    r"title:\s+(?P<title>[^\n]*)\n"
    r"snippet:\s+(?P<snippet>.*?)(?=\n\n\[\d+\]\s+doc_id:|\n\n# Task|\Z)",
    flags=re.DOTALL,
)

USER_POOL_RE = re.compile(r"# Candidate Document Pool\n\n(?P<pool>.*?)(?:\n\n# Task|\Z)", re.DOTALL)

SYSTEM_PROMPT = """You are a Harness-1 retrieval subagent.

Use tool-style JSON actions to search, inspect, verify, curate evidence, and end
the search. You do not answer directly. Your final product is a curated evidence
set.

Available action schemas:
{"tool":"fan_out_search","queries":["short query 1","short query 2"]}
{"tool":"review_docs","doc_ids":["doc_id"]}
{"tool":"verify","doc_ids":["doc_id"],"claim":"checkable claim"}
{"tool":"curate","add_ids":["doc_id"],"remove_ids":[],"importance":{"doc_id":"high"}}
{"tool":"end_search","reasoning":"brief reason"}

Rules:
- Output exactly one JSON action per assistant message.
- Use the Candidate Document Pool as the persistent search memory.
- Search broadly, inspect promising documents, and curate all relevant evidence.
- Do not end after one document unless only one relevant document exists.
- Prefer recall over precision, but do not include unrelated IDs.
- End only after the curated set is sufficient for the query."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--max-search-results", type=int, default=20)
    parser.add_argument("--max-review-docs", type=int, default=5)
    parser.add_argument("--include-verify", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact(text: str, limit: int = 900) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return f"{text[:head]} ... {text[-tail:]}"


def parse_pool(messages: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    pool_match = USER_POOL_RE.search(user)
    pool_text = pool_match.group("pool") if pool_match else user
    docs: dict[str, dict[str, str]] = {}
    for match in POOL_RE.finditer(pool_text):
        doc_id = clean_text(match.group("doc_id"))
        if not doc_id:
            continue
        docs[doc_id] = {
            "title": clean_text(match.group("title")),
            "snippet": compact(match.group("snippet"), 1200),
        }
    return docs


def pick_gold(row: dict[str, Any], docs: dict[str, dict[str, str]]) -> list[str]:
    candidates = set(docs)
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(row.get("gold_doc_ids", [])) + list(row.get("answer_doc_ids", [])):
        doc_id = str(raw).strip()
        if doc_id in candidates and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def make_search_queries(query: str, answer: str) -> list[str]:
    words = [w.strip(".,;:!?()[]{}\"'") for w in query.split()]
    distinctive = [w for w in words if len(w) >= 5][:10]
    q1 = " ".join(distinctive[:8]) or query[:80]
    if answer:
        q2 = answer[:80]
    else:
        q2 = " ".join(distinctive[-8:]) or query[-80:]
    q3 = " ".join(distinctive[3:11]) or q1
    queries = []
    for q in (q1, q2, q3):
        q = clean_text(q)[:100]
        if q and q not in queries:
            queries.append(q)
    return queries[:3]


def render_search_observation(
    docs: dict[str, dict[str, str]],
    ranked_ids: list[str],
    gold_ids: list[str],
    max_results: int,
) -> str:
    wanted: list[str] = []
    seen: set[str] = set()
    for doc_id in gold_ids + ranked_ids:
        if doc_id in docs and doc_id not in seen:
            seen.add(doc_id)
            wanted.append(doc_id)
        if len(wanted) >= max_results:
            break
    lines = ["Search results:"]
    for idx, doc_id in enumerate(wanted, start=1):
        doc = docs[doc_id]
        lines.append(
            f"[{idx}] doc_id: {doc_id}\n"
            f"title: {doc['title']}\n"
            f"snippet: {compact(doc['snippet'], 700)}"
        )
    return "\n\n".join(lines)


def render_review_observation(docs: dict[str, dict[str, str]], doc_ids: list[str]) -> str:
    lines = ["Reviewed documents:"]
    for doc_id in doc_ids:
        doc = docs.get(doc_id)
        if not doc:
            continue
        lines.append(
            f"doc_id: {doc_id}\n"
            f"title: {doc['title']}\n"
            f"text: {compact(doc['snippet'], 1200)}"
        )
    return "\n\n".join(lines)


def assistant_action(payload: dict[str, Any]) -> dict[str, str]:
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}


def user_observation(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def build_trajectory(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    source_messages = list(row["prompt"])
    docs = parse_pool(source_messages)
    gold_ids = pick_gold(row, docs)
    if not docs or not gold_ids:
        return None

    query = clean_text(row.get("query"))
    answer = clean_text(row.get("answer"))
    source_user = next((m.get("content", "") for m in source_messages if m.get("role") == "user"), "")
    if not source_user:
        source_user = f"# Query\n{query}"
    user_content = (
        source_user.rstrip()
        + "\n\n# Agent Instructions\n"
        + "Ignore any earlier final-answer format in this prompt. Start a Harness search trajectory now. "
        + "Use one tool-style JSON action per assistant message. Use the Candidate Document Pool above as "
        + "search memory, curate all relevant doc_ids, then end_search."
    )
    candidate_ids = [str(x) for x in row.get("candidate_doc_ids", []) if str(x) in docs]
    search_queries = make_search_queries(query, answer)
    review_ids = gold_ids[: args.max_review_docs]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        assistant_action({"tool": "fan_out_search", "queries": search_queries}),
        user_observation(render_search_observation(docs, candidate_ids, gold_ids, args.max_search_results)),
        assistant_action({"tool": "review_docs", "doc_ids": review_ids}),
        user_observation(render_review_observation(docs, review_ids)),
        assistant_action(
            {
                "tool": "curate",
                "add_ids": gold_ids,
                "remove_ids": [],
                "importance": {doc_id: "high" for doc_id in gold_ids},
            }
        ),
        user_observation(
            "Curated set updated: "
            + ", ".join(gold_ids)
            + ". Continue verifying before ending."
        ),
    ]
    if args.include_verify:
        claim = f"These documents support the answer or constraints for: {query[:180]}"
        messages.extend(
            [
                assistant_action({"tool": "verify", "doc_ids": review_ids, "claim": claim}),
                user_observation(
                    "Verification: reviewed documents support the curated evidence set "
                    "for the query constraints."
                ),
            ]
        )
    messages.append(
        assistant_action(
            {
                "tool": "end_search",
                "reasoning": (
                    "Curated the relevant evidence documents and verified enough context "
                    "to support the query."
                ),
            }
        )
    )

    return {
        "messages": messages,
        "query_id": row.get("query_id"),
        "query": query,
        "answer": answer,
        "gold_doc_ids": gold_ids,
        "source": row.get("source"),
        "task_type": "agentic_tool_trajectory",
    }


def split_assistant_action_records(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one SFT example per assistant action.

    The local SFT trainer masks everything before the last assistant message in
    a row.  Splitting the trajectory teaches every action instead of only the
    final end_search action.
    """
    messages = list(trajectory["messages"])
    out: list[dict[str, Any]] = []
    action_idx = 0
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        action_idx += 1
        try:
            action = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            action = {}
        out.append(
            {
                **{k: v for k, v in trajectory.items() if k != "messages"},
                "messages": messages[: idx + 1],
                "action_index": action_idx,
                "target_tool": action.get("tool", ""),
                "task_type": "agentic_tool_action",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            trajectory = build_trajectory(dict(dataset[idx]), args)
            if trajectory is None:
                continue
            for record in split_assistant_action_records(trajectory):
                rows.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    Dataset.from_list(rows).save_to_disk(str(output_path.with_suffix("")))
    print(json.dumps({"rows": len(rows), "jsonl": str(output_path), "dataset": str(output_path.with_suffix(""))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
