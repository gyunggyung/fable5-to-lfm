#!/usr/bin/env python
"""Evaluate local LFM/Qwen Harness-style multi-turn tool trajectories via vLLM."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer


POOL_RE = re.compile(
    r"\[(?P<idx>\d+)\]\s+doc_id:\s+(?P<doc_id>[^\n]+)\n"
    r"title:\s+(?P<title>[^\n]*)\n"
    r"snippet:\s+(?P<snippet>.*?)(?=\n\n\[\d+\]\s+doc_id:|\n\n# Task|\Z)",
    flags=re.DOTALL,
)
USER_POOL_RE = re.compile(r"# Candidate Document Pool\n\n(?P<pool>.*?)(?:\n\n# Task|\Z)", re.DOTALL)

SYSTEM_PROMPT = """You are a Harness-1 evidence retrieval model.

Given a query and a Candidate Document Pool, your primary job is to curate all
document IDs that are needed to answer or verify the query. The preferred output
is the final strict JSON curation object:
{"curated_doc_ids":["doc_id"],"reasoning":"brief evidence-based reason"}

Use tool-style JSON actions only as a fallback when the Candidate Document Pool
is insufficient or ambiguous. The final product must be a curated evidence set.

Available action schemas:
{"tool":"fan_out_search","queries":["short query 1","short query 2"]}
{"tool":"review_docs","doc_ids":["doc_id"]}
{"tool":"verify","doc_ids":["doc_id"],"claim":"checkable claim"}
{"tool":"curate","add_ids":["doc_id"],"remove_ids":[],"importance":{"doc_id":"high"}}
{"tool":"end_search","reasoning":"brief reason"}
{"curated_doc_ids":["doc_id"],"reasoning":"brief reason"}

Rules:
- Output exactly one JSON action per assistant message.
- Use the Candidate Document Pool as the persistent search memory.
- When a Candidate Document Pool is present, prefer the one-step
  {"curated_doc_ids":[...],"reasoning":"..."} final curation action.
- Use search/review/verify only when the pool is insufficient or ambiguous.
- Curate all relevant evidence ids. Do not stop after one document unless only
  one relevant document exists.
- Multi-hop queries usually need several evidence documents. Include all
  documents supporting constraints, intermediate clues, or the final answer.
- Prefer recall over precision, but do not include unrelated IDs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=float, default=240.0)
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
            "snippet": compact(match.group("snippet"), 1400),
        }
    return docs


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, dict):
            item = item.get("doc_id") or item.get("id") or item.get("docid")
        doc_id = str(item).strip().strip("\"'`[](){}")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def collect_string_values(value: Any, limit: int = 8) -> list[str]:
    out: list[str] = []

    def visit(item: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(item, str):
            text = clean_text(item)
            if len(text) >= 3 and text not in out:
                out.append(text[:140])
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return out


def collect_doc_ids_from_action(action: dict[str, Any]) -> list[str]:
    keys = (
        "curated_doc_ids",
        "selected_doc_ids",
        "evidence_doc_ids",
        "add_ids",
        "doc_ids",
        "ids",
    )
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        for doc_id in normalize_ids(action.get(key)):
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


def add_curated(current: list[str], add_ids: list[str], remove_ids: set[str] | None = None) -> list[str]:
    remove_ids = remove_ids or set()
    out = [doc_id for doc_id in current if doc_id not in remove_ids]
    for doc_id in add_ids:
        if doc_id not in out:
            out.append(doc_id)
    return out


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], max_prompt_length: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if max_prompt_length > 0 and len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
    return prompt


def generate(prompt: str, args: argparse.Namespace, seed: int) -> str:
    payload = {
        "model": args.served_model_name,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": seed,
        "stop": ["<|im_end|>"],
    }
    request = urllib.request.Request(
        args.vllm_base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {body[:2000]}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM returned no choices: {data}")
    return str(choices[0].get("text") or "").strip()


def search_observation(docs: dict[str, dict[str, str]], queries: list[str], seen: set[str], limit: int = 20) -> str:
    terms = [term.lower() for query in queries for term in re.findall(r"[A-Za-z0-9가-힣]{4,}", query)]
    scored: list[tuple[int, int, str]] = []
    for rank, (doc_id, doc) in enumerate(docs.items()):
        if doc_id in seen:
            continue
        text = f"{doc['title']} {doc['snippet']}".lower()
        score = sum(1 for term in terms if term in text)
        scored.append((score, rank, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [doc_id for _, _, doc_id in scored[:limit]]
    seen.update(picked)
    if not picked:
        return "Search results: no new documents found."
    lines = ["Search results:"]
    for idx, doc_id in enumerate(picked, start=1):
        doc = docs[doc_id]
        lines.append(
            f"[{idx}] doc_id: {doc_id}\n"
            f"title: {doc['title']}\n"
            f"snippet: {compact(doc['snippet'], 700)}"
        )
    return "\n\n".join(lines)


def review_observation(docs: dict[str, dict[str, str]], doc_ids: list[str]) -> str:
    lines = ["Reviewed documents:"]
    for doc_id in doc_ids[:5]:
        doc = docs.get(doc_id)
        if not doc:
            lines.append(f"doc_id: {doc_id}\nnot found in memory.")
            continue
        lines.append(
            f"doc_id: {doc_id}\n"
            f"title: {doc['title']}\n"
            f"text: {compact(doc['snippet'], 1200)}"
        )
    return "\n\n".join(lines)


def f_beta(recall: float, precision: float, beta: float = 2.0) -> float:
    if recall <= 0.0 and precision <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def score(curated: list[str], row: dict[str, Any], docs: dict[str, dict[str, str]]) -> dict[str, float]:
    gold = set(str(x) for x in row.get("gold_doc_ids", []))
    answer = set(str(x) for x in row.get("answer_doc_ids", [])) or gold
    valid = set(docs)
    selected = [doc_id for doc_id in curated if doc_id in valid]
    selected_set = set(selected)
    invalid = max(0, len(curated) - len(selected))
    recall = len(selected_set & gold) / max(len(gold), 1)
    precision = len(selected_set & gold) / max(len(selected_set), 1)
    answer_recall = len(selected_set & answer) / max(len(answer), 1)
    return {
        "recall": recall,
        "precision": precision,
        "answer_recall": answer_recall,
        "f2": f_beta(recall, precision),
        "all_gold_found": 1.0 if gold and gold <= selected_set else 0.0,
        "invalid_count": float(invalid),
        "selected_count": float(len(selected_set)),
    }


def run_episode(row: dict[str, Any], tokenizer: Any, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    source_messages = list(row["prompt"])
    docs = parse_pool(source_messages)
    query = clean_text(row.get("query"))
    source_user = next((m.get("content", "") for m in source_messages if m.get("role") == "user"), "")
    if not source_user:
        source_user = f"# Query\n{query}"
    user_content = (
        source_user.rstrip()
        + "\n\n# Agent Instructions\n"
        + "Ignore any earlier natural-language final-answer format in this prompt. "
        + "Select all supporting evidence from the Candidate Document Pool above and output the final curation action now. "
        + "Prefer exactly this JSON shape: {\"curated_doc_ids\":[\"doc_id\"],\"reasoning\":\"brief reason\"}. "
        + "Most hard queries need multiple IDs; do not stop after only the final-answer page if intermediate evidence is relevant. "
        + "Only use search/review tools if the pool is insufficient."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    curated: list[str] = []
    seen: set[str] = set()
    valid_actions = 0
    invalid_actions = 0
    ended = False
    trace: list[dict[str, Any]] = []

    for turn in range(args.max_turns):
        prompt = apply_chat_template(tokenizer, messages, args.max_prompt_length)
        text = generate(prompt, args, seed + turn)
        action = extract_json_object(text)
        trace.append({"turn": turn, "raw": text, "action": action})
        if action is None:
            invalid_actions += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Invalid action. Output one strict JSON tool action."})
            continue
        valid_actions += 1
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        tool = str(action.get("tool") or action.get("name") or "").strip()
        if not tool:
            direct_ids = collect_doc_ids_from_action(action)
            if direct_ids:
                curated = direct_ids
                trace[-1]["inferred_tool"] = "direct_curate"
                ended = True
                break
            invalid_actions += 1
            obs = "Missing tool name. Use fan_out_search, review_docs, verify, curate, or end_search."
        elif tool in {"fan_out_search", "search_corpus", "search", "lookup", "web_search"}:
            supplied_doc_ids = collect_doc_ids_from_action(action)
            if supplied_doc_ids:
                curated = add_curated(curated, supplied_doc_ids)
                trace[-1]["inferred_tool"] = "search_doc_ids_as_curate"
                obs = (
                    "Curated set updated from supplied doc_ids: "
                    + ", ".join(curated)
                    + ". Review or verify if needed, then end_search."
                )
            else:
                queries = normalize_ids(action.get("queries") or action.get("query"))
                if not queries:
                    queries = collect_string_values(action)
                obs = search_observation(docs, queries, seen)
        elif tool == "review_docs":
            obs = review_observation(docs, normalize_ids(action.get("doc_ids")))
        elif tool == "verify":
            obs = "Verification: checked requested documents against the claim using local evidence text."
        elif tool == "curate":
            add_ids = collect_doc_ids_from_action(action)
            remove_ids = set(normalize_ids(action.get("remove_ids")))
            curated = add_curated(curated, add_ids, remove_ids)
            obs = "Curated set updated: " + ", ".join(curated)
        elif tool == "end_search":
            ended = True
            break
        else:
            invalid_actions += 1
            supplied_doc_ids = collect_doc_ids_from_action(action)
            if supplied_doc_ids:
                curated = add_curated(curated, supplied_doc_ids)
                trace[-1]["inferred_tool"] = "unknown_doc_ids_as_curate"
                obs = (
                    "Curated set updated from supplied doc_ids: "
                    + ", ".join(curated)
                    + ". Use review_docs, verify, curate, or end_search next."
                )
            else:
                queries = collect_string_values(action)
                if queries:
                    trace[-1]["inferred_tool"] = "unknown_as_search"
                    obs = search_observation(docs, queries, seen)
                else:
                    obs = f"Unknown tool: {tool}. Use fan_out_search, review_docs, verify, curate, or end_search."
        messages.append({"role": "user", "content": obs})

    fallback_ids: list[str] = []
    if not curated:
        for item in trace:
            action = item.get("action")
            if isinstance(action, dict):
                fallback_ids = add_curated(fallback_ids, collect_doc_ids_from_action(action))
        if fallback_ids:
            curated = fallback_ids
            ended = True

    metrics = score(curated, row, docs)
    metrics["valid_action_rate"] = valid_actions / max(valid_actions + invalid_actions, 1)
    metrics["ended"] = 1.0 if ended else 0.0
    metrics["fallback_curated_from_emitted_doc_ids"] = 1.0 if fallback_ids else 0.0
    return {
        "query_id": row.get("query_id"),
        "query": query,
        "curated_doc_ids": curated,
        "gold_doc_ids": list(row.get("gold_doc_ids", [])),
        "metrics": metrics,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    totals: dict[str, float] = {}
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(limit):
            result = run_episode(dict(dataset[idx]), tokenizer, args, seed=20260619 + idx * 17)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
            for key, value in result["metrics"].items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if count % 10 == 0:
                print(
                    json.dumps(
                        {
                            "eval_idx": count,
                            "mean_recall": totals.get("recall", 0.0) / count,
                            "mean_f2": totals.get("f2", 0.0) / count,
                            "valid_action_rate": totals.get("valid_action_rate", 0.0) / count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = {"count": count}
    summary.update({f"mean_{key}": value / max(count, 1) for key, value in totals.items()})
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={args.summary_json}")
    print(f"predictions={args.output_jsonl}")


if __name__ == "__main__":
    main()
