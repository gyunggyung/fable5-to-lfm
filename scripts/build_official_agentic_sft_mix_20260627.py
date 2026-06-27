#!/usr/bin/env python
"""Build a Fable-style official-agentic SFT mix.

This dataset is benchmark-aligned, not benchmark-contaminated: it does not read
official Tool-Decathlon, MCP-Atlas, or tau2-bench task files. It repackages our
Fable/Hermes/Harness traces and adds deterministic synthetic support/tool
trajectories that exercise the same skills: tool choice, parameter discipline,
multi-step recovery, user guidance, and final verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FABLE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    {
        "name": "fable5_agentic_traces",
        "path": FABLE_DIR / "datasets/fable5_lfm_sft_20260623.jsonl",
        "cap": 3500,
        "target": "fable_style_agentic_coding",
    },
    {
        "name": "glm52_terminal_toolmix",
        "path": FABLE_DIR / "datasets/glm52_chaser_terminal_toolmix_20260624.jsonl",
        "cap": 6000,
        "target": "tool_decathlon_like",
    },
    {
        "name": "hermes_agent_function_traces",
        "path": FABLE_DIR / "datasets/hermes_agent_traces_chat_20260624.jsonl",
        "cap": 4500,
        "target": "mcp_atlas_like",
    },
    {
        "name": "phase2_reasoning",
        "path": FABLE_DIR / "datasets/phase2_reasoning_lfm_sft_20260623.jsonl",
        "cap": 500,
        "target": "reasoning_style",
    },
]

HARNESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fan_out_search",
            "description": "Search a hidden local corpus with multiple short keyword queries.",
            "parameters": {
                "type": "object",
                "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_docs",
            "description": "Inspect candidate documents by id before using them as evidence.",
            "parameters": {
                "type": "object",
                "properties": {"doc_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["doc_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curate",
            "description": "Add or remove evidence document ids from the working set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "add_ids": {"type": "array", "items": {"type": "string"}},
                    "remove_ids": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "object"},
                },
                "required": ["add_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": "Check whether reviewed documents support a precise claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                    "claim": {"type": "string"},
                },
                "required": ["doc_ids", "claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_search",
            "description": "Finish after enough evidence is curated and verified.",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    },
]

TELECOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": "Look up subscription, device, and current ticket metadata.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_line_diagnostics",
            "description": "Run backend diagnostics for a broadband, fiber, or mobile line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "line_id": {"type": "string"},
                    "scope": {"type": "string", "enum": ["signal", "provisioning", "connectivity", "all"]},
                },
                "required": ["account_id", "line_id", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_network_feature",
            "description": "Enable or disable a backend network feature for a line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "feature": {"type": "string"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["line_id", "feature", "enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Escalate an unresolved or field-service issue with a concise summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "summary": {"type": "string"},
                },
                "required": ["account_id", "severity", "summary"],
            },
        },
    },
]

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "github_search_issues",
            "description": "Search repository issues and pull requests.",
            "parameters": {
                "type": "object",
                "properties": {"repo": {"type": "string"}, "query": {"type": "string"}},
                "required": ["repo", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slack_search",
            "description": "Search team messages for decisions, incidents, or ownership.",
            "parameters": {
                "type": "object",
                "properties": {"channel": {"type": "string"}, "query": {"type": "string"}},
                "required": ["channel", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_find_doc",
            "description": "Find internal documents by title or content query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "owner": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Create a calendar event after the necessary facts are known.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["title", "attendees", "date"],
            },
        },
    },
]

FABLE_STYLE_SYSTEM = (
    "You are a Fable/Mythos-style agentic model. Work in a durable loop: "
    "understand the goal, choose the next tool deliberately, inspect results, "
    "repair mistakes, verify completion, and only then answer. Prefer concrete "
    "actions over vague advice. Keep final answers concise, but preserve the "
    "reasoning discipline that led to the action."
)

OFFICIAL_ALIGNMENT = {
    "tool_decathlon_like": (
        "Training target: official long-horizon tool-use benchmarks such as "
        "Tool-Decathlon/Toolathlon. Maintain state across many tool calls, use "
        "valid parameters, recover from tool errors, and verify before stopping."
    ),
    "mcp_atlas_like": (
        "Training target: MCP-Atlas style multi-server tool orchestration. "
        "Infer which server/tool is needed from the user goal, avoid distractors, "
        "chain 3-6 precise calls when needed, and ground the final answer in tool outputs."
    ),
    "tau2_telecom_style": (
        "Training target: tau2 telecom style dual-control support. Coordinate "
        "backend tools with user-visible troubleshooting steps. Do not assume the "
        "user already performed a physical action; guide and verify."
    ),
    "fable_style_agentic_coding": (
        "Training target: Fable-style coding and terminal agency. Read, edit, run, "
        "observe, fix, and summarize only after verification."
    ),
    "reasoning_style": (
        "Training target: Mythos/Fable reasoning style. Think through constraints, "
        "avoid premature answers, and keep the final response grounded."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FABLE_DIR / "datasets/official_agentic_sft_mix_20260627.jsonl"))
    parser.add_argument("--meta", default=str(FABLE_DIR / "datasets/official_agentic_sft_mix_20260627.meta.json"))
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-total-chars", type=int, default=90000)
    parser.add_argument("--max-final-chars", type=int, default=32000)
    parser.add_argument("--max-cyrillic", type=float, default=0.30)
    parser.add_argument("--tau2-synthetic-count", type=int, default=900)
    parser.add_argument("--mcp-synthetic-count", type=int, default=900)
    return parser.parse_args()


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if "\u0400" <= ch <= "\u04ff") / len(text)


def stable_hash(row: dict[str, Any]) -> str:
    normalized = json.dumps(
        {
            "messages": row.get("messages"),
            "tools": row.get("tools"),
            "benchmark_target": row.get("benchmark_target"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    clean: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        cleaned = dict(message)
        if "content" in cleaned:
            cleaned["content"] = stringify_content(cleaned.get("content")).strip()
        if role != "assistant" and not cleaned.get("content"):
            continue
        clean.append(cleaned)
    if len(clean) < 2 or clean[-1].get("role") != "assistant":
        return None
    return clean


def prepend_style_system(messages: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    prefix = FABLE_STYLE_SYSTEM + "\n\n" + OFFICIAL_ALIGNMENT[target]
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = prefix + "\n\nOriginal task policy:\n" + stringify_content(first.get("content"))
        return [first] + messages[1:]
    return [{"role": "system", "content": prefix}] + messages


def parse_json_action(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        action = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        return None
    return action


def make_tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def convert_final_json_action_to_tool_call(messages: list[dict[str, Any]], row_uid: str) -> list[dict[str, Any]] | None:
    final = messages[-1]
    action = parse_json_action(stringify_content(final.get("content")))
    if action is None:
        return None
    tool_name = action.pop("tool")
    call_id = "call_" + hashlib.sha1(row_uid.encode("utf-8")).hexdigest()[:16]
    converted = [dict(message) for message in messages[:-1]]
    converted.append({"role": "assistant", "content": "", "tool_calls": [make_tool_call(tool_name, action, call_id)]})
    return converted


def normalize_row(raw: dict[str, Any], *, source_name: str, target: str, uid: str) -> dict[str, Any] | None:
    messages = clean_messages(raw)
    if messages is None:
        return None

    final_text = stringify_content(messages[-1].get("content"))
    full_text = "\n".join(stringify_content(message.get("content")) for message in messages)
    if len(final_text) > ARGS.max_final_chars or len(full_text) > ARGS.max_total_chars:
        return None
    if cyrillic_ratio(final_text) > ARGS.max_cyrillic:
        return None

    tools = None
    mix_task = "fable_style_trace"
    if target == "tool_decathlon_like":
        converted = convert_final_json_action_to_tool_call(messages, uid)
        if converted is not None:
            messages = converted
            tools = HARNESS_TOOLS
            mix_task = "json_action_to_native_tool_call"
    elif target == "mcp_atlas_like":
        mix_task = "function_calling_trace"
    elif target == "reasoning_style":
        mix_task = "reasoning_trace"

    return {
        "messages": prepend_style_system(messages, target),
        "tools": tools,
        "benchmark_target": target,
        "mix_task": mix_task,
        "mix_source": source_name,
        "source_uid": uid,
    }


def load_source(spec: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            uid = str(raw.get("mix_uid") or raw.get("uid") or raw.get("query_id") or f"{spec['name']}:{stats['input']}")
            row = normalize_row(raw, source_name=spec["name"], target=spec["target"], uid=uid)
            if row is None:
                continue
            rows.append(row)
            stats["valid"] += 1

    cap = int(spec.get("cap") or 0)
    if cap > 0 and len(rows) > cap:
        rows = rng.sample(rows, cap)
    stats["kept"] = len(rows)
    return rows, stats


def telecom_issue(index: int) -> dict[str, str]:
    issues = [
        {
            "symptom": "my fiber modem goes online for two minutes, then drops again",
            "scope": "connectivity",
            "feature": "stale_dhcp_binding",
            "fix": "clear the stale DHCP binding and ask the user to power-cycle the modem once.",
        },
        {
            "symptom": "my mobile data works in town but fails at home after the SIM swap",
            "scope": "provisioning",
            "feature": "volte_profile",
            "fix": "re-enable the correct VoLTE/data profile and have the user toggle airplane mode.",
        },
        {
            "symptom": "video calls freeze every evening even though speed tests look fine",
            "scope": "signal",
            "feature": "qos_interactive_video",
            "fix": "enable the interactive-video QoS profile and verify latency after the router reboot.",
        },
        {
            "symptom": "the router says connected but every website times out",
            "scope": "all",
            "feature": "dns_proxy",
            "fix": "repair DNS proxy provisioning and ask the user to reconnect one device before closing.",
        },
    ]
    issue = dict(issues[index % len(issues)])
    issue["account_id"] = f"acct-{72000 + index}"
    issue["line_id"] = f"line-{3100 + index}"
    return issue


def make_tau2_rows(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        issue = telecom_issue(idx)
        base_messages = [
            {
                "role": "system",
                "content": (
                    FABLE_STYLE_SYSTEM
                    + "\n\n"
                    + OFFICIAL_ALIGNMENT["tau2_telecom_style"]
                    + "\n\nPolicy: explain only user-safe steps; use backend tools for account state; "
                    "escalate when diagnostics indicate a field issue."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Account {issue['account_id']}: {issue['symptom']}. "
                    "Please troubleshoot with me. I can reboot devices or check lights if you ask."
                ),
            },
        ]
        first_call = make_tool_call(
            "get_customer_profile",
            {"account_id": issue["account_id"]},
            f"call_tau2_{idx}_profile",
        )
        rows.append(
            {
                "messages": base_messages
                + [{"role": "assistant", "content": "", "tool_calls": [first_call]}],
                "tools": TELECOM_TOOLS,
                "benchmark_target": "tau2_telecom_style",
                "mix_task": "telecom_first_tool_selection",
                "mix_source": "synthetic_tau2_style",
                "source_uid": f"tau2-synth-{idx}-first",
            }
        )

        diag_call = make_tool_call(
            "run_line_diagnostics",
            {"account_id": issue["account_id"], "line_id": issue["line_id"], "scope": issue["scope"]},
            f"call_tau2_{idx}_diag",
        )
        rows.append(
            {
                "messages": base_messages
                + [
                    {"role": "assistant", "content": "", "tool_calls": [first_call]},
                    {
                        "role": "tool",
                        "tool_call_id": first_call["id"],
                        "name": "get_customer_profile",
                        "content": json.dumps(
                            {
                                "account_id": issue["account_id"],
                                "primary_line": issue["line_id"],
                                "plan": "fiber_plus" if idx % 2 == 0 else "mobile_unlimited",
                                "open_tickets": [],
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {"role": "assistant", "content": "", "tool_calls": [diag_call]},
                ],
                "tools": TELECOM_TOOLS,
                "benchmark_target": "tau2_telecom_style",
                "mix_task": "telecom_diagnostic_tool_selection",
                "mix_source": "synthetic_tau2_style",
                "source_uid": f"tau2-synth-{idx}-diag",
            }
        )

        final = (
            f"I found a backend issue on `{issue['line_id']}` related to `{issue['feature']}`. "
            f"I fixed the account-side setting: {issue['fix']} After that, tell me whether the "
            "service stays stable for five minutes; if not, I will escalate with the diagnostics already attached."
        )
        rows.append(
            {
                "messages": base_messages
                + [
                    {"role": "assistant", "content": "", "tool_calls": [diag_call]},
                    {
                        "role": "tool",
                        "tool_call_id": diag_call["id"],
                        "name": "run_line_diagnostics",
                        "content": json.dumps(
                            {
                                "line_id": issue["line_id"],
                                "fault": issue["feature"],
                                "field_visit_required": False,
                                "recommended_action": issue["fix"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {"role": "assistant", "content": final},
                ],
                "tools": TELECOM_TOOLS,
                "benchmark_target": "tau2_telecom_style",
                "mix_task": "telecom_guided_resolution",
                "mix_source": "synthetic_tau2_style",
                "source_uid": f"tau2-synth-{idx}-final",
            }
        )
    return rows


def mcp_case(index: int) -> dict[str, Any]:
    repos = ["infra/runner", "web/app", "ml/evals", "tools/agent-shell"]
    channels = ["#incidents", "#evals", "#frontend", "#agent-tools"]
    owners = ["maya@example.com", "ren@example.com", "alex@example.com", "ji@example.com"]
    return {
        "repo": repos[index % len(repos)],
        "channel": channels[index % len(channels)],
        "owner": owners[index % len(owners)],
        "date": f"2026-07-{(index % 20) + 1:02d}",
        "topic": ["flaky eval", "tool parser regression", "deployment rollback", "latency spike"][index % 4],
    }


def make_mcp_rows(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        case = mcp_case(idx)
        base_messages = [
            {
                "role": "system",
                "content": FABLE_STYLE_SYSTEM + "\n\n" + OFFICIAL_ALIGNMENT["mcp_atlas_like"],
            },
            {
                "role": "user",
                "content": (
                    f"Find the latest context for the {case['topic']} in `{case['repo']}`, "
                    f"identify the owner, and schedule a follow-up on {case['date']} if the issue is still open."
                ),
            },
        ]
        issue_call = make_tool_call(
            "github_search_issues",
            {"repo": case["repo"], "query": case["topic"]},
            f"call_mcp_{idx}_issues",
        )
        slack_call = make_tool_call(
            "slack_search",
            {"channel": case["channel"], "query": case["topic"]},
            f"call_mcp_{idx}_slack",
        )
        calendar_call = make_tool_call(
            "calendar_create_event",
            {
                "title": f"Follow up: {case['topic']}",
                "attendees": [case["owner"]],
                "date": case["date"],
                "notes": f"Review current state in {case['repo']} and confirm owner actions.",
            },
            f"call_mcp_{idx}_calendar",
        )
        rows.append(
            {
                "messages": base_messages + [{"role": "assistant", "content": "", "tool_calls": [issue_call]}],
                "tools": MCP_TOOLS,
                "benchmark_target": "mcp_atlas_like",
                "mix_task": "mcp_first_tool_selection",
                "mix_source": "synthetic_mcp_style",
                "source_uid": f"mcp-synth-{idx}-first",
            }
        )
        rows.append(
            {
                "messages": base_messages
                + [
                    {"role": "assistant", "content": "", "tool_calls": [issue_call]},
                    {
                        "role": "tool",
                        "tool_call_id": issue_call["id"],
                        "name": "github_search_issues",
                        "content": json.dumps(
                            {"status": "open", "issue": 1000 + idx, "owner": case["owner"]},
                            ensure_ascii=False,
                        ),
                    },
                    {"role": "assistant", "content": "", "tool_calls": [slack_call]},
                ],
                "tools": MCP_TOOLS,
                "benchmark_target": "mcp_atlas_like",
                "mix_task": "mcp_cross_server_followup",
                "mix_source": "synthetic_mcp_style",
                "source_uid": f"mcp-synth-{idx}-slack",
            }
        )
        rows.append(
            {
                "messages": base_messages
                + [
                    {"role": "assistant", "content": "", "tool_calls": [issue_call]},
                    {
                        "role": "tool",
                        "tool_call_id": issue_call["id"],
                        "name": "github_search_issues",
                        "content": json.dumps({"status": "open", "owner": case["owner"]}, ensure_ascii=False),
                    },
                    {"role": "assistant", "content": "", "tool_calls": [calendar_call]},
                ],
                "tools": MCP_TOOLS,
                "benchmark_target": "mcp_atlas_like",
                "mix_task": "mcp_action_completion",
                "mix_source": "synthetic_mcp_style",
                "source_uid": f"mcp-synth-{idx}-calendar",
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target = Counter(row["benchmark_target"] for row in rows)
    by_task = Counter(row["mix_task"] for row in rows)
    by_source = Counter(row["mix_source"] for row in rows)
    with_tools = sum(1 for row in rows if row.get("tools"))
    tool_call_targets = 0
    for row in rows:
        final = row["messages"][-1]
        if isinstance(final, dict) and final.get("tool_calls"):
            tool_call_targets += 1
    return {
        "total_rows": len(rows),
        "rows_with_tools_field": with_tools,
        "rows_with_final_tool_call": tool_call_targets,
        "by_benchmark_target": dict(by_target),
        "by_mix_task": dict(by_task),
        "by_mix_source": dict(by_source),
    }


def main() -> None:
    rng = random.Random(ARGS.seed)
    output_path = Path(ARGS.output)
    meta_path = Path(ARGS.meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    skipped = Counter()
    seen: set[str] = set()

    for spec in DEFAULT_SOURCES:
        source_rows, stats = load_source(spec, rng)
        source_stats[spec["name"]] = stats
        rows.extend(source_rows)

    rows.extend(make_tau2_rows(ARGS.tau2_synthetic_count))
    rows.extend(make_mcp_rows(ARGS.mcp_synthetic_count))

    deduped: list[dict[str, Any]] = []
    for row in rows:
        digest = stable_hash(row)
        if digest in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(digest)
        row["mix_uid"] = f"{row['mix_source']}:{digest[:16]}"
        deduped.append(row)

    rng.shuffle(deduped)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in deduped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "purpose": (
            "Fable/Mythos-style SFT mix aligned to official Tool-Decathlon, MCP-Atlas, "
            "and tau2 telecom skills without using official benchmark task files."
        ),
        "output": str(output_path),
        "seed": ARGS.seed,
        "source_stats": source_stats,
        "synthetic": {
            "tau2_style_base_cases": ARGS.tau2_synthetic_count,
            "mcp_style_base_cases": ARGS.mcp_synthetic_count,
            "rows_per_tau2_case": 3,
            "rows_per_mcp_case": 3,
        },
        "skipped": dict(skipped),
        "summary": summarize(deduped),
        "leakage_note": (
            "This builder intentionally does not read official benchmark datasets. "
            "Use official public sets only for held-out evaluation."
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ARGS = parse_args()
    main()
