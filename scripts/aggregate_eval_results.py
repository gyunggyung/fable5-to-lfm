#!/usr/bin/env python
"""Fabliq eval 결과 취합기.

모든 모델의 평가 결과를 모아서 비교 테이블 생성.
입력: /home/work/.data/fabliq_eval/{model_name}/{tb2_lite,mmlu,humaneval}/...
출력: markdown 비교 표 + JSON 요약
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default="/home/work/.data/fabliq_eval")
    p.add_argument("--output-md", default="fable_distillation/EVAL_RESULTS_20260623.md")
    p.add_argument("--output-json", default="fable_distillation/EVAL_RESULTS_20260623.json")
    return p.parse_args()


def load_tb2_lite(model_dir: Path) -> dict:
    """tb2_lite summary.json 찾기."""
    tb2_dir = model_dir / "tb2_lite"
    if not tb2_dir.exists():
        return {}
    # Find summary file
    for p in tb2_dir.rglob("summary*.json"):
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return {}


def load_lm_eval(model_dir: Path, task: str) -> dict:
    """lm-eval-harness 결과 찾기."""
    task_dir = model_dir / task
    if not task_dir.exists():
        return {}
    # lm-eval writes to a subdirectory with timestamp
    for p in sorted(task_dir.rglob("results.json"), reverse=True):
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return {}


def extract_metric(data: dict, task_name: str, metric_name: str = "acc,none") -> float | None:
    """lm-eval 결과에서 특정 metric 추출."""
    if not data:
        return None
    results = data.get("results", {})
    task_data = results.get(task_name, {})
    # Try several metric keys
    for key in [metric_name, "acc", "acc_norm", "exact_match", "pass@1"]:
        if key in task_data:
            val = task_data[key]
            if isinstance(val, dict):
                return val.get(",")
            if isinstance(val, (int, float)):
                return val
    return None


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_md = Path(args.output_md)
    out_json = Path(args.output_json)

    models = sorted([d for d in eval_root.iterdir() if d.is_dir()])
    print(f"Found {len(models)} models in {eval_root}")

    table_rows = []
    for model_dir in models:
        model_name = model_dir.name
        print(f"Processing: {model_name}")

        tb2 = load_tb2_lite(model_dir)
        mmlu_data = load_lm_eval(model_dir, "mmlu")
        humaneval_data = load_lm_eval(model_dir, "humaneval")

        # tb2_lite metrics
        tb2_score = tb2.get("overall_score") or tb2.get("score")
        tb2_acc = tb2.get("accuracy") or tb2.get("acc")
        tb2_total = tb2.get("total_steps") or tb2.get("total")

        # MMLU
        mmlu_acc = extract_metric(mmlu_data, "mmlu", "acc,none")

        # HumanEval
        humaneval_pass1 = extract_metric(humaneval_data, "humaneval", "pass@1,none")

        table_rows.append({
            "model": model_name,
            "tb2_score": tb2_score,
            "tb2_acc": tb2_acc,
            "tb2_total": tb2_total,
            "mmlu_acc": mmlu_acc,
            "humaneval_pass1": humaneval_pass1,
        })

    # Write JSON
    out_json.write_text(json.dumps(table_rows, indent=2))
    print(f"\nWrote JSON: {out_json}")

    # Write Markdown table
    lines = [
        "# Fabliq Eval Results Comparison",
        "",
        f"Generated: 2026-06-23 (eval root: `{eval_root}`)",
        "",
        "| Model | tb2_lite score | tb2_lite acc | MMLU (5-shot) | HumanEval pass@1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['model']} "
            f"| {r['tb2_score'] if r['tb2_score'] is not None else '-'} "
            f"| {r['tb2_acc'] if r['tb2_acc'] is not None else '-'} "
            f"| {r['mmlu_acc']:.4f if r['mmlu_acc'] is not None else '-'} "
            f"| {r['humaneval_pass1']:.4f if r['humaneval_pass1'] is not None else '-'} |"
        )

    out_md.write_text("\n".join(lines))
    print(f"Wrote Markdown: {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
