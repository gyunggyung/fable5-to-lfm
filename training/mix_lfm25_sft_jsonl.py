#!/usr/bin/env python
"""Mix direct JSON-curation SFT rows with agentic action SFT rows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-jsonl", required=True)
    parser.add_argument("--agentic-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--direct-repeat", type=int, default=5)
    parser.add_argument("--agentic-repeat", type=int, default=1)
    parser.add_argument("--agentic-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    direct_rows = read_jsonl(args.direct_jsonl)
    agentic_rows = read_jsonl(args.agentic_jsonl)
    if args.agentic_limit > 0:
        agentic_rows = agentic_rows[: args.agentic_limit]

    mixed: list[dict[str, Any]] = []
    for _ in range(max(args.direct_repeat, 0)):
        mixed.extend(direct_rows)
    for _ in range(max(args.agentic_repeat, 0)):
        mixed.extend(agentic_rows)

    random.Random(args.seed).shuffle(mixed)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset_path = output_path.with_suffix("")
    Dataset.from_list(mixed).save_to_disk(str(dataset_path))
    print(
        json.dumps(
            {
                "direct_rows": len(direct_rows),
                "agentic_rows": len(agentic_rows),
                "direct_repeat": args.direct_repeat,
                "agentic_repeat": args.agentic_repeat,
                "mixed_rows": len(mixed),
                "jsonl": str(output_path),
                "dataset": str(dataset_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
