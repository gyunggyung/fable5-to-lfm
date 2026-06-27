#!/usr/bin/env python
"""Return success only when the local GLM-5.2 BF16 snapshot is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2")
    parser.add_argument("--cache-dir", default="/home/work/.data/huggingface/hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            cache_dir=args.cache_dir,
            local_files_only=True,
        )
    )
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    expected = sorted(set(index.get("weight_map", {}).values()))
    missing = [filename for filename in expected if not (snapshot / filename).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"missing {len(missing)} safetensor shards; first missing: {preview}")
    print(f"ready snapshot={snapshot} shards={len(expected)}")


if __name__ == "__main__":
    main()
