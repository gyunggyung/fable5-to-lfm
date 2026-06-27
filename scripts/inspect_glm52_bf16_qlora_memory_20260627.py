#!/usr/bin/env python
"""Inspect why local GLM-5.2 BF16 -> BitsAndBytes QLoRA is memory risky.

This reads safetensors metadata only. It does not materialize model weights.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2/snapshots/"
        "f2263102df303b2faa54a6861a29d1770ce846c0",
    )
    return parser.parse_args()


def tensor_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def main() -> None:
    args = parse_args()
    snapshot = Path(args.snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    files = sorted(set(index["weight_map"].values()))

    bf16_bytes_by_group: dict[str, int] = defaultdict(int)
    raw_expert_bytes = 0
    raw_expert_tensors = 0
    tensor_count = 0
    for file_name in files:
        with safe_open(snapshot / file_name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_count += 1
                nbytes = tensor_numel(handle.get_slice(key).get_shape()) * 2
                match = re.match(r"model\.layers\.(\d+)\.", key)
                if match:
                    group = f"layer_{int(match.group(1)):02d}"
                elif key.startswith("model.embed_tokens."):
                    group = "embed"
                elif key.startswith("lm_head."):
                    group = "lm_head"
                elif key.startswith("model.norm."):
                    group = "norm"
                else:
                    group = "other"
                bf16_bytes_by_group[group] += nbytes

                if ".mlp.experts." in key and key.endswith(
                    (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight", ".gate_up_proj", ".down_proj")
                ):
                    raw_expert_bytes += nbytes
                    raw_expert_tensors += 1

    total_bf16 = sum(bf16_bytes_by_group.values())
    layer_items = sorted(
        (name, value) for name, value in bf16_bytes_by_group.items() if name.startswith("layer_")
    )
    print(f"snapshot={snapshot}")
    print(f"files={len(files)} tensors={tensor_count}")
    print(f"total_bf16_gib={total_bf16 / 1024**3:.2f}")
    print(f"raw_moe_expert_bf16_gib={raw_expert_bytes / 1024**3:.2f} tensors={raw_expert_tensors}")
    print("non_layer_bf16_gib:")
    for name in ("embed", "lm_head", "norm", "other"):
        if name in bf16_bytes_by_group:
            print(f"  {name}: {bf16_bytes_by_group[name] / 1024**3:.2f}")
    print("layer_bf16_gib:")
    for name, value in layer_items:
        print(f"  {name}: {value / 1024**3:.2f}")


if __name__ == "__main__":
    main()
