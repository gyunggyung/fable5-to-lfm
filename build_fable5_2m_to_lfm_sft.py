#!/usr/bin/env python
"""Complete-FABLE.5-traces-2M parquet → LFM messages JSONL 변환 (Phase-3).

2M raw Claude Code events를 sessionId별로 그룹화하고, Glint가 미리 뽑은
구조화 row (output_type=cot/output 포함) 만 추출하여 LFM tool-use 형식으로 변환.

기존 build_fable5_to_lfm_sft.py의 변환 로직 (context 파싱, tool_call 변환, cot → <think>) 재사용.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# build_fable5_to_lfm_sft.py의 함수 재사용
sys.path.insert(0, str(Path(__file__).parent))
from build_fable5_to_lfm_sft import (
    SYSTEM_PROMPT,
    convert_row,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="fable_distillation/datasets/Complete-FABLE.5-traces-2M/data/train.parquet",
    )
    p.add_argument(
        "--output",
        default="fable_distillation/datasets/fable5_2m_lfm_sft_20260623.jsonl",
    )
    p.add_argument(
        "--meta",
        default="fable_distillation/datasets/fable5_2m_lfm_sft_20260623.meta.json",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="최대 출력 row 수 (너무 많으면 학습 시간 폭발)",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--seen-count-max",
        type=int,
        default=3,
        help="seen_count가 이 값 이하인 row만 사용 (중복 세션 필터)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading parquet: {in_path}", flush=True)
    pf = pq.ParquetFile(in_path)
    total = pf.metadata.num_rows
    print(f"  total rows: {total}", flush=True)

    # 1. 구조화 row (Glint output_type 포함) 만 추출
    # row_json을 파싱해서 'output_type' 키가 있으면 Glint 구조화 row로 간주
    candidates = []
    scanned = 0
    for batch in pf.iter_batches(batch_size=50000):
        rows = batch.to_pylist()
        for r in rows:
            scanned += 1
            if r["seen_count"] > args.seen_count_max:
                continue
            try:
                j = json.loads(r["row_json"])
            except json.JSONDecodeError:
                continue
            # Glint 구조화 row는 output_type 키를 가짐
            if "output_type" not in j:
                continue
            candidates.append(j)
        if scanned % 500_000 == 0:
            print(f"  scanned {scanned}/{total}, candidates {len(candidates)}", flush=True)

    print(f"Total candidates: {len(candidates)}", flush=True)

    # 2. 샘플링 (너무 많으면 max_rows로 제한)
    import random
    random.seed(args.sample_seed)
    if len(candidates) > args.max_rows:
        random.shuffle(candidates)
        candidates = candidates[: args.max_rows]
        print(f"Sampled down to {len(candidates)}", flush=True)

    # 3. LFM messages 변환 (build_fable5_to_lfm_sft.convert_row 재사용)
    converted = []
    skipped_error = 0
    skipped_short = 0
    for row in candidates:
        result = convert_row(row)
        if result is None:
            skipped_short += 1
            continue
        if "error" in result:
            skipped_error += 1
            continue
        converted.append(result)

    print(f"Converted: {len(converted)}, skipped_short: {skipped_short}, skipped_error: {skipped_error}", flush=True)

    # 4. 출력
    with out_path.open("w") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. 메타
    from collections import Counter
    stats = {
        "input_total": total,
        "scanned": scanned,
        "candidates": len(candidates),
        "max_rows_cap": args.max_rows,
        "seen_count_max": args.seen_count_max,
        "output_rows": len(converted),
        "skipped_short": skipped_short,
        "skipped_error": skipped_error,
        "output_type_dist": dict(Counter(r["output_type"] for r in converted)),
        "system_prompt": SYSTEM_PROMPT,
        "source_dataset": "Complete-FABLE.5-traces-2M",
    }
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
