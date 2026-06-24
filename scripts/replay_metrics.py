#!/usr/bin/env python3
"""Shared replay-scoring helpers without the vLLM dependency."""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter, defaultdict


def parse_json_blob(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def flatten_keystrokes(commands: list[dict]) -> list[str]:
    units: list[str] = []
    for command in commands:
        keystrokes = str(command.get("keystrokes", "")).replace("\r\n", "\n").replace("\r", "\n")
        pieces = [piece.strip() for piece in keystrokes.split("\n") if piece.strip()]
        if pieces:
            units.extend(pieces)
        else:
            units.append("<WAIT>")
    return units


def _split_command_text(value: str) -> list[str]:
    pieces = [
        piece.strip()
        for piece in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if piece.strip()
    ]
    return pieces


def _decode_json_string(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value


def _payload_commands(payload: dict) -> list[str]:
    commands: list[str] = []
    raw_commands = payload.get("commands", [])
    if isinstance(raw_commands, list):
        commands.extend(flatten_keystrokes([cmd for cmd in raw_commands if isinstance(cmd, dict)]))

    args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else payload.get("args")
    if not isinstance(args, dict):
        args = {}

    for candidate in (
        payload.get("keystrokes"),
        payload.get("command"),
        payload.get("cmd"),
        args.get("keystrokes"),
        args.get("command"),
        args.get("cmd"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            commands.extend(_split_command_text(candidate))
    return commands


def fallback_commands(text: str) -> list[str]:
    commands: list[str] = []
    for key in ("keystrokes", "command", "cmd"):
        pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)'
        for match in re.findall(pattern, text):
            commands.extend(_split_command_text(_decode_json_string(match)))
    return commands


def tokenize_command(command: str) -> list[str]:
    normalized = command.strip().lower()
    if not normalized:
        return []
    if normalized == "<wait>":
        return ["<wait>"]
    try:
        return shlex.split(normalized)
    except Exception:
        return normalized.split()


def token_f1(left: str, right: str) -> float:
    left_tokens = tokenize_command(left)
    right_tokens = tokenize_command(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    left_counter = Counter(left_tokens)
    right_counter = Counter(right_tokens)
    overlap = sum((left_counter & right_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_counter.values())
    recall = overlap / sum(right_counter.values())
    return 2 * precision * recall / (precision + recall)


def normalize_units(units: list[str]) -> list[str]:
    normalized: list[str] = []
    for unit in units:
        value = re.sub(r"\s+", " ", unit.strip())
        normalized.append(value if value else "<WAIT>")
    return normalized


def score_commands(pred_units: list[str], ref_units: list[str]) -> tuple[float, float, float, float]:
    pred_units = normalize_units(pred_units)
    ref_units = normalize_units(ref_units)
    if not pred_units and not ref_units:
        return 0.0, 0.0, 0.0, 0.0

    first_exact = float(bool(pred_units and ref_units and pred_units[0].lower() == ref_units[0].lower()))
    if not pred_units:
        return first_exact, 0.0, 0.0, 0.0
    if not ref_units:
        return first_exact, 0.0, 0.0, 0.0

    recall = sum(max(token_f1(ref_unit, pred_unit) for pred_unit in pred_units) for ref_unit in ref_units) / len(ref_units)
    precision = sum(max(token_f1(pred_unit, ref_unit) for ref_unit in ref_units) for pred_unit in pred_units) / len(pred_units)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return first_exact, precision, recall, f1


def step_bucket(step_idx: int) -> str:
    if step_idx <= 1:
        return "early"
    if step_idx <= 4:
        return "mid"
    return "late"


def parse_prediction(text: str) -> dict:
    payload = parse_json_blob(text)
    commands: list[str] = []
    task_complete = None
    has_analysis = False
    has_plan = False
    valid_json = payload is not None

    if payload is not None:
        commands = _payload_commands(payload)
        task_complete = payload.get("task_complete") if isinstance(payload.get("task_complete"), bool) else None
        has_analysis = bool(payload.get("analysis"))
        has_plan = bool(payload.get("plan"))

    if not commands:
        commands = fallback_commands(text)

    return {
        "valid_json": valid_json,
        "has_analysis": has_analysis,
        "has_plan": has_plan,
        "task_complete": task_complete,
        "command_units": commands,
    }


def aggregate_scores(per_step: list[dict]) -> dict:
    total = len(per_step)
    if total == 0:
        return {
            "steps": 0,
            "tasks": 0,
            "avg_ref_cmds": 0.0,
            "avg_pred_cmds": 0.0,
            "valid_json_pct": 0.0,
            "has_analysis_pct": 0.0,
            "has_plan_pct": 0.0,
            "first_cmd_exact_pct": 0.0,
            "avg_command_precision": 0.0,
            "avg_command_recall": 0.0,
            "avg_command_f1": 0.0,
            "complete_true_recall_pct": 0.0,
            "premature_complete_rate_pct": 0.0,
            "by_bucket": {},
            "by_source_group": {},
            "next_action_score": 0.0,
        }
    aggregate: dict[str, object] = {
        "steps": total,
        "tasks": len({row["task_id"] for row in per_step}),
        "avg_ref_cmds": round(sum(len(row["ref_command_units"]) for row in per_step) / total, 2),
        "avg_pred_cmds": round(sum(len(row["pred_command_units"]) for row in per_step) / total, 2),
        "valid_json_pct": round(sum(row["valid_json"] for row in per_step) / total * 100, 1),
        "has_analysis_pct": round(sum(row["has_analysis"] for row in per_step) / total * 100, 1),
        "has_plan_pct": round(sum(row["has_plan"] for row in per_step) / total * 100, 1),
        "first_cmd_exact_pct": round(sum(row["first_cmd_exact"] for row in per_step) / total * 100, 1),
        "avg_command_precision": round(sum(row["command_precision"] for row in per_step) / total, 4),
        "avg_command_recall": round(sum(row["command_recall"] for row in per_step) / total, 4),
        "avg_command_f1": round(sum(row["command_f1"] for row in per_step) / total, 4),
    }

    positive_steps = [row for row in per_step if row["ref_task_complete"]]
    negative_steps = [row for row in per_step if not row["ref_task_complete"]]
    aggregate["complete_true_recall_pct"] = round(
        sum(row["pred_task_complete_true"] for row in positive_steps) / max(len(positive_steps), 1) * 100, 1
    )
    aggregate["premature_complete_rate_pct"] = round(
        sum(bool(row["pred_task_complete"]) for row in negative_steps) / max(len(negative_steps), 1) * 100, 1
    )

    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {"steps": 0, "avg_command_f1": 0.0})
    by_source: dict[str, dict[str, float]] = defaultdict(
        lambda: {"steps": 0, "avg_command_f1": 0.0, "first_cmd_exact_pct": 0.0}
    )
    for row in per_step:
        bucket_stats = by_bucket[row["bucket"]]
        bucket_stats["steps"] += 1
        bucket_stats["avg_command_f1"] += row["command_f1"]

        source_stats = by_source[row["source_group"]]
        source_stats["steps"] += 1
        source_stats["avg_command_f1"] += row["command_f1"]
        source_stats["first_cmd_exact_pct"] += row["first_cmd_exact"]

    aggregate["by_bucket"] = {
        key: {
            "steps": int(stats["steps"]),
            "avg_command_f1": round(stats["avg_command_f1"] / stats["steps"], 4),
        }
        for key, stats in sorted(by_bucket.items())
    }
    aggregate["by_source_group"] = {
        key: {
            "steps": int(stats["steps"]),
            "avg_command_f1": round(stats["avg_command_f1"] / stats["steps"], 4),
            "first_cmd_exact_pct": round(stats["first_cmd_exact_pct"] / stats["steps"] * 100, 1),
        }
        for key, stats in sorted(by_source.items())
    }
    aggregate["next_action_score"] = round(
        100.0 * (0.7 * aggregate["avg_command_f1"] + 0.3 * (aggregate["first_cmd_exact_pct"] / 100.0)),
        2,
    )
    return aggregate
