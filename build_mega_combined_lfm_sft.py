#!/usr/bin/env python
"""Mega-Combined LFM SFT 데이터셋 빌더 (밤샘 작업용).

모든 Fable-5 계열 전처리 결과를 통합 + 교차 dedup + 러시아어 필터:
- fable5_lfm_sft_20260623.jsonl (Phase-1, 4,047 rows)
- withinus_lfm_sft_20260623.jsonl (Phase-2 WithinUs, 135 rows)
- helio_lfm_sft_20260623.jsonl (Phase-2 Helio, 146 rows)
- fable5_2m_lfm_sft_20260623.jsonl (Phase-3 2M, 3,866 rows)
- (선택) lordx64/agentic-distill-fable-5-sft → LFM 변환 (4,659 rows)

교차 dedup: SHA-256 on (첫 user message) 으로 서로 다른 소스 간 중복 제거
러시아어 필터: assistant content Cyrillic 비율 30% 초과 시 drop
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return cyr / len(text)


def conversation_hash(messages: list[dict]) -> str:
    """멀티턴 row 구별 위해 (첫 user msg + 마지막 assistant msg) 로 hash.

    Fable-5 trace는 하나의 대화가 여러 assistant turn으로 쪼개져 들어있음.
    단순히 첫 user msg로 dedup하면 같은 대화의 다른 turn들이 다 중복 처리됨.
    마지막 assistant turn 내용까지 포함해야 정확한 dedup.
    """
    first_user = ""
    last_asst = ""
    for m in messages:
        if m.get("role") == "user" and not first_user:
            first_user = m.get("content", "")[:500]
        if m.get("role") == "assistant":
            last_asst = m.get("content", "")[:500]
    return hashlib.sha256(f"{first_user}||{last_asst}".encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="fable_distillation/datasets/mega_combined_lfm_sft_20260623.jsonl")
    p.add_argument("--meta", default="fable_distillation/datasets/mega_combined_lfm_sft_20260623.meta.json")
    p.add_argument("--max-cyrillic", type=float, default=0.30)
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl",
            "fable_distillation/datasets/withinus_lfm_sft_20260623.jsonl",
            "fable_distillation/datasets/helio_lfm_sft_20260623.jsonl",
            "fable_distillation/datasets/fable5_2m_lfm_sft_20260623.jsonl",
        ],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    out_rows = []
    per_source_stats: dict[str, dict[str, int]] = {}

    for input_path in args.inputs:
        p = Path(input_path)
        if not p.exists():
            print(f"SKIP missing: {p}")
            continue
        source_name = p.stem
        per_source_stats[source_name] = {"input": 0, "kept": 0, "dup": 0, "russian": 0}

        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                per_source_stats[source_name]["input"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages", [])
                if len(msgs) < 3:
                    continue

                # dedup (conversation-level)
                h = conversation_hash(msgs)
                if h in seen_hashes:
                    per_source_stats[source_name]["dup"] += 1
                    continue
                seen_hashes.add(h)

                # Russian filter on assistant content
                asst_text = " ".join(
                    m.get("content", "") for m in msgs if m.get("role") == "assistant"
                )
                if cyrillic_ratio(asst_text) > args.max_cyrillic:
                    per_source_stats[source_name]["russian"] += 1
                    continue

                obj["mega_source"] = source_name
                out_rows.append(obj)
                per_source_stats[source_name]["kept"] += 1

    # shuffle
    import random
    random.seed(42)
    random.shuffle(out_rows)

    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "inputs": args.inputs,
        "total_input": sum(s["input"] for s in per_source_stats.values()),
        "total_kept": len(out_rows),
        "total_dup": sum(s["dup"] for s in per_source_stats.values()),
        "total_russian": sum(s["russian"] for s in per_source_stats.values()),
        "max_cyrillic": args.max_cyrillic,
        "per_source": per_source_stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
