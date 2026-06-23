#!/usr/bin/env python
"""WithinUs 25k → LFM SFT JSONL 변환 (시나리오 B).

카테고리 균등 샘플링 + 템플릿 과적합 방지:
1. 카테고리별 균등 분할 (advanced_coding, agentic_planning, general_qa, mathematical_reasoning, scientific_analysis, cybersecurity)
2. assistant 응답 "Drawing from the autonomous, frontier-level reasoning..." 첫 문장 truncate
3. user 프롬프트 SHA-256 dedup (템플릿 반복 제거)
4. messages 형식 LFM SFT JSONL 출력
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Provide rigorous, well-structured answers "
    "across coding, cybersecurity, mathematics, scientific analysis, agentic planning, "
    "and general expert topics. Be precise and thorough."
)

# "Drawing from the autonomous..." 템플릿 첫 문장 패턴
DRAWING_PATTERN = re.compile(
    r"^Drawing from the autonomous, frontier[- ]level reasoning[^.]*\.\s*",
    re.IGNORECASE,
)


def clean_assistant(content: str) -> str:
    return DRAWING_PATTERN.sub("", content, count=1).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="fable_distillation/datasets/claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl")
    p.add_argument("--output", default="fable_distillation/datasets/withinus_lfm_sft_20260623.jsonl")
    p.add_argument("--meta", default="fable_distillation/datasets/withinus_lfm_sft_20260623.meta.json")
    p.add_argument("--per-category", type=int, default=350, help="카테고리당 샘플 수 (6카테고리 × 350 = 2,100행)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    # 카테고리별 그룹화 + user 프롬프트 dedup
    by_cat: dict[str, list[dict]] = defaultdict(list)
    seen_hashes: set[str] = set()
    skipped_dup = 0
    for r in rows:
        msgs = r.get("messages", [])
        if not msgs or len(msgs) < 2:
            continue
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        if not user_msg:
            continue
        user_text = user_msg.get("content", "")
        # 템플릿 추출 (대부분 "Solve or provide...:" 같은 패턴 반복) - 실제 질문 부분만 hash
        prompt_hash = hashlib.sha256(user_text.strip().encode()).hexdigest()[:16]
        if prompt_hash in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(prompt_hash)
        by_cat[r.get("category", "unknown")].append(r)

    # 카테고리별 균등 샘플링
    import random
    random.seed(42)
    sampled = []
    cat_stats = {}
    for cat, items in by_cat.items():
        random.shuffle(items)
        take = min(args.per_category, len(items))
        sampled.extend(items[:take])
        cat_stats[cat] = {"available": len(items), "sampled": take}

    # LFM messages 형식 변환 + assistant 첫 문장 truncate
    out_rows = []
    for r in sampled:
        msgs = r["messages"]
        new_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in msgs:
            if m["role"] == "assistant":
                cleaned = clean_assistant(m.get("content", ""))
                if not cleaned:
                    continue
                new_msgs.append({"role": "assistant", "content": cleaned})
            else:
                new_msgs.append({"role": m["role"], "content": m.get("content", "")})
        if len(new_msgs) >= 3:
            out_rows.append({
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "source": "withinus_mythos_distilled_25k",
                "messages": new_msgs,
            })

    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "skipped_dup": skipped_dup,
        "per_category_target": args.per_category,
        "category_stats": cat_stats,
        "system_prompt": SYSTEM_PROMPT,
        "transformations": [
            "user prompt SHA-256 dedup",
            f"assistant first-sentence pattern removal ({DRAWING_PATTERN.pattern[:50]}...)",
            f"per-category uniform sampling (max {args.per_category})",
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
