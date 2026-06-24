#!/usr/bin/env python3
"""Prefetch DiffusionGemma model files into the Hugging Face cache."""

from __future__ import annotations

import argparse
import json

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    path = snapshot_download(
        args.model,
        revision=args.revision,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.model",
            "*.txt",
            "*.jinja",
            "tokenizer*",
            "processor*",
            "generation_config.json",
        ],
    )
    print(json.dumps({"model": args.model, "local_path": path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
