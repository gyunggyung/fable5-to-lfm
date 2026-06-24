#!/usr/bin/env python3
"""Summarize TB2-lite replay evaluator outputs into markdown."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        agg = data["aggregate"]
        prompt_meta = data.get("prompt_template", {})
        rows.append({
            "model": data["model"],
            "model_short": data["model_short"],
            "score": agg["next_action_score"],
            "cmd_f1": agg["avg_command_f1"],
            "first_exact": agg["first_cmd_exact_pct"],
            "valid_json": agg["valid_json_pct"],
            "complete_recall": agg["complete_true_recall_pct"],
            "false_complete": agg["premature_complete_rate_pct"],
            "sec_per_step": data["avg_sec_per_step"],
            "load_time": data["load_time_sec"],
            "steps": agg["steps"],
            "tasks": agg["tasks"],
            "template_status": prompt_meta.get("template_status", "unknown"),
            "rank_eligible": bool(prompt_meta.get("rank_eligible", True)),
        })
    rows.sort(key=lambda row: (not row["rank_eligible"], -row["score"], -row["cmd_f1"], -row["first_exact"]))
    return rows


def build_markdown(rows: list[dict], results_dir: Path) -> str:
    today = date.today().isoformat()
    lines = [
        f"# TB2-lite Replay Results ({today})",
        "",
        "Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.",
        "",
        "| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Template | Sec/Step | Load (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    rank = 0
    for idx, row in enumerate(rows, start=1):
        if row["rank_eligible"]:
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "excluded"
        lines.append(
            "| "
            f"{rank_text} | {row['model_short']} | {row['score']:.2f} | {row['cmd_f1']:.4f} | "
            f"{row['first_exact']:.1f}% | {row['valid_json']:.1f}% | "
            f"{row['template_status']} | "
            f"{row['sec_per_step']:.3f} | {row['load_time']:.1f} |"
        )

    lines.extend([
        "",
        f"Results directory: `{results_dir}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="tb2_lite/results")
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = load_rows(results_dir)
    markdown = build_markdown(rows, results_dir)
    if args.output_path:
        Path(args.output_path).write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
