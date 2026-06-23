#!/usr/bin/env python
"""Build strict-JSON SFT rows for Harness-style retrieval curation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def clean_ids(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            row = dict(dataset[idx])
            candidate_set = {str(item) for item in row.get("candidate_doc_ids", [])}
            gold_ids = [doc_id for doc_id in clean_ids(row.get("gold_doc_ids", [])) if doc_id in candidate_set]
            if not gold_ids:
                continue
            answer = str(row.get("answer") or "").strip()
            target = {
                "curated_doc_ids": gold_ids,
                "reasoning": "Selected the gold evidence documents needed to answer the query.",
            }
            if answer:
                target["reasoning"] = f"Selected the evidence documents supporting the answer: {answer[:220]}"
            messages = list(row["prompt"]) + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]
            record = {
                "messages": messages,
                "query_id": row.get("query_id"),
                "source": row.get("source"),
                "gold_doc_ids": gold_ids,
            }
            rows.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    Dataset.from_list(rows).save_to_disk(str(output_path.with_suffix("")))
    print(json.dumps({"rows": len(rows), "jsonl": str(output_path), "dataset": str(output_path.with_suffix(""))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

