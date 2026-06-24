#!/usr/bin/env python
"""Merge a PEFT LoRA adapter into a vLLM-loadable HF checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-class",
        choices=("causal-lm", "multimodal-lm", "image-text-to-text"),
        default="causal-lm",
    )
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def model_loader(model_class: str) -> Any:
    return {
        "causal-lm": AutoModelForCausalLM,
        "multimodal-lm": AutoModelForMultimodalLM,
        "image-text-to-text": AutoModelForImageTextToText,
    }[model_class]


def copy_optional_files(source_dirs: list[Path], output_dir: Path) -> list[str]:
    copied: list[str] = []
    for filename in (
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        for source_dir in source_dirs:
            source_path = source_dir / filename
            if source_path.exists():
                shutil.copy2(source_path, output_dir / filename)
                copied.append(filename)
                break
    return copied


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing adapter_config.json in {adapter_dir}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(args.dtype)
    tokenizer_source = args.tokenizer_path or str(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = model_loader(args.model_class)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    base = loader.from_pretrained(args.base_model, **load_kwargs)
    base.config.use_cache = True
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))
    copied = copy_optional_files([adapter_dir, Path(args.base_model)], output_dir)

    manifest = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "model_class": args.model_class,
        "tokenizer_source": tokenizer_source,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "max_shard_size": args.max_shard_size,
        "copied_optional_files": copied,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
