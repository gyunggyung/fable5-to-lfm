#!/usr/bin/env python3
"""Merge DiffusionGemma Transformers TB2 shard outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from replay_metrics import aggregate_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-short", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no shard files matched {input_dir / args.glob}")

    per_step = []
    shard_meta = []
    total_tokens = 0
    total_gen_time = 0.0
    max_wall_time = 0.0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        tb2 = data.get("tb2") or data
        steps = tb2.get("per_step") or data.get("per_step") or []
        per_step.extend(steps)
        total_tokens += int(tb2.get("output_tokens", data.get("completion_tokens", 0)) or 0)
        gen_time = float(tb2.get("gen_time_sec", data.get("gen_time_sec", 0.0)) or 0.0)
        total_gen_time += gen_time
        max_wall_time = max(max_wall_time, gen_time)
        shard_meta.append(
            {
                "path": str(path),
                "steps": len(steps),
                "gen_time_sec": gen_time,
                "output_tokens": int(tb2.get("output_tokens", 0) or 0),
                "shard_index": tb2.get("shard_index"),
                "shard_count": tb2.get("shard_count"),
            }
        )

    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_short,
        "model_short": args.model_short,
        "timestamp": datetime.utcnow().isoformat(),
        "backend": "transformers_sharded",
        "aggregate": aggregate,
        "per_step": per_step,
        "shards": shard_meta,
        "shard_files": [str(path) for path in paths],
        "steps": len(per_step),
        "output_tokens": total_tokens,
        "sum_gen_time_sec": round(total_gen_time, 2),
        "wall_time_sec_estimate": round(max_wall_time, 2),
        "output_tokens_per_wall_sec": round(total_tokens / max(max_wall_time, 1e-9), 2),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
