#!/usr/bin/env python
"""Simple direct Fabliq eval (transformers backend, no vllm dependency).

터미널 에이전트 능력 + 기본 reasoning 평가.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>\[(?P<tool>\w+)\((?P<args>.*)\)\]<\|tool_call_end\|>"
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


# 평가용 터미널 task 프롬프트들 (단순화)
TEST_PROMPTS = [
    {
        "id": "ls_py",
        "task": "List all Python files in the current directory and count lines.",
        "expected_tool": "Bash",
        "check": lambda out: "Bash" in out and ("ls" in out.lower() or "find" in out.lower()),
    },
    {
        "id": "read_file",
        "task": "Read the README.md file and summarize it.",
        "expected_tool": "Read",
        "check": lambda out: "Read" in out,
    },
    {
        "id": "grep_error",
        "task": "Find all 'ERROR' lines in /var/log/syslog.",
        "expected_tool": "Grep",
        "check": lambda out: "Grep" in out or "Bash" in out,
    },
    {
        "id": "write_test",
        "task": "Create a new Python file test.py with a hello world function.",
        "expected_tool": "Write",
        "check": lambda out: "Write" in out,
    },
    {
        "id": "edit_config",
        "task": "Change the database host from localhost to db.example.com in config.yaml.",
        "expected_tool": "Edit",
        "check": lambda out: "Edit" in out,
    },
]

SYSTEM_PROMPT = (
    "You are an agentic coding assistant. Read the conversation history and tool results, "
    "think step by step inside <think>...</think>, then either call a tool using "
    "<|tool_call_start|>[ToolName(arg=value)]<|tool_call_end|> or respond with text. "
    "Use available tools (Bash, Edit, Read, Write, Glob, Grep, WebSearch, WebFetch, etc.) "
    "to accomplish the user's task. Be concise but thorough."
)


def run_one(model, tok, prompt: str, device: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=False)
    return response


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model.split("/")[-1]

    print(f"Loading {args.model}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"  # single GPU for speed
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    device = next(model.parameters()).device
    results = []

    for test in TEST_PROMPTS:
        t1 = time.time()
        response = run_one(model, tok, test["task"], str(device))
        elapsed = time.time() - t1

        # 평가
        has_tool_call = bool(TOOL_CALL_RE.search(response))
        has_think = bool(THINK_RE.search(response))
        tool_correct = test["check"](response)

        result = {
            "id": test["id"],
            "task": test["task"],
            "expected_tool": test["expected_tool"],
            "response_preview": response[:500],
            "has_tool_call": has_tool_call,
            "has_think_block": has_think,
            "tool_correct": tool_correct,
            "elapsed_sec": round(elapsed, 2),
        }
        results.append(result)
        print(f"  [{test['id']}] tool={has_tool_call} think={has_think} correct={tool_correct} ({elapsed:.1f}s)", flush=True)

    # Summary
    summary = {
        "model": args.model,
        "model_name": model_name,
        "total": len(results),
        "tool_call_rate": sum(r["has_tool_call"] for r in results) / len(results),
        "think_rate": sum(r["has_think_block"] for r in results) / len(results),
        "tool_correct_rate": sum(r["tool_correct"] for r in results) / len(results),
        "avg_latency": sum(r["elapsed_sec"] for r in results) / len(results),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
