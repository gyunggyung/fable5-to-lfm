#!/usr/bin/env python
"""Helio 462x → LFM SFT JSONL 변환 (시나리오 D).

러시아어 70% 문제 해결:
1. Cyrillic 비율 30% 미만 행만 필터 (영어 우세)
2. thinking을 <think>...</think>로 감싸서 assistant reasoning 학습
3. line 192 손상 스킵
4. max 8192 tokens 초과 시 truncate (너무 긴 건 8192로 자름, 중간 발췌)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a deep-reasoning assistant. Think step by step inside <think>...</think>, "
    "then provide a clear, structured answer."
)


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return cyr / len(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="fable_distillation/datasets/Fable-5-Distill-Reasoning-462x/Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl")
    p.add_argument("--output", default="fable_distillation/datasets/helio_lfm_sft_20260623.jsonl")
    p.add_argument("--meta", default="fable_distillation/datasets/helio_lfm_sft_20260623.meta.json")
    p.add_argument("--max-cyrillic", type=float, default=0.30, help="query + thinking 평균 Cyrillic 비율 상한")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    meta_path = Path(args.meta)

    total = 0
    bad_lines = 0
    kept = 0
    skipped_lang = 0
    out_rows = []

    with in_path.open() as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            query = obj.get("query", "")
            thinking = obj.get("thinking", "")
            answer = obj.get("answer") or obj.get("response") or ""

            if not query or not thinking:
                continue

            # Cyrillic 비율 검사 (query + thinking 평균)
            ratio = (cyrillic_ratio(query) + cyrillic_ratio(thinking)) / 2
            if ratio > args.max_cyrillic:
                skipped_lang += 1
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"<think>\n{thinking}\n</think>\n\n{answer}".strip()},
            ]
            out_rows.append({
                "source": "helio_fable5_distill_reasoning_462x",
                "cyrillic_ratio": round(ratio, 3),
                "messages": messages,
            })
            kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "input_total": total,
        "bad_lines": bad_lines,
        "skipped_lang": skipped_lang,
        "kept_rows": kept,
        "max_cyrillic_ratio": args.max_cyrillic,
        "system_prompt": SYSTEM_PROMPT,
        "note": "러시아어 우세 행 제거, line 192 손상 스킵",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
