#!/usr/bin/env python
"""Generic chat SFT runner for non-LFM Fabliq experiments.

This is intentionally conservative: it supports LoRA-first smoke runs across
Gemma/Qwen-style Transformers models before spending full-SFT GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-class",
        choices=("causal-lm", "multimodal-lm", "image-text-to-text"),
        default="multimodal-lm",
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokenized-cache-dir",
        default="",
        help="Optional shared tokenized dataset cache. Useful under torchrun to avoid N ranks re-tokenizing the same data.",
    )
    parser.add_argument("--sft-adapter-path", default="")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=52)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="{}",
        help='Extra kwargs for apply_chat_template, e.g. {"enable_thinking": false}.',
    )
    parser.add_argument(
        "--chat-serialization",
        choices=("native", "simple-chatml"),
        default="native",
        help="Use the model chat template, or a conservative ChatML serializer for replay/tool traces.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-config-json", default="")
    return parser.parse_args()


def configure_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        torch.cuda.set_device(int(local_rank))


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    if value.startswith("regex:"):
        return value.removeprefix("regex:")
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules if len(modules) > 1 else value


def load_fsdp_config(raw_config: str) -> dict[str, Any]:
    if not raw_config:
        return {}
    parsed = json.loads(raw_config)
    if not isinstance(parsed, dict):
        raise ValueError("--fsdp-config-json must decode to a JSON object")
    return parsed


def load_template_kwargs(raw_config: str) -> dict[str, Any]:
    parsed = json.loads(raw_config or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("--chat-template-kwargs-json must decode to a JSON object")
    return parsed


def process_rank() -> int:
    return int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")


def acquire_file_lock(lock_path: Path) -> int | None:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
    return fd


def wait_for_tokenized_cache(cache_dir: Path, lock_path: Path) -> None:
    success_path = cache_dir / "_SUCCESS"
    while not success_path.exists():
        if not lock_path.exists():
            break
        time.sleep(5)


def load_template_backend(model_path: str) -> tuple[Any, Any]:
    """Return (template_backend, tokenizer).

    Prefer AutoTokenizer for text-only data. Some Gemma/Qwen multimodal cards
    document AutoProcessor first, so fall back to processor.tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        backend = tokenizer
    except Exception:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        backend = processor
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return backend, tokenizer


def apply_chat_template(
    backend: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any],
) -> list[int]:
    result = backend.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
        **template_kwargs,
    )
    if isinstance(result, dict):
        result = result.get("input_ids")
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return list(result)


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def render_simple_chatml(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    chunks: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = stringify_content(message.get("content"))
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def encode_chat_messages(
    backend: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    chat_serialization: str,
    template_kwargs: dict[str, Any],
) -> list[int]:
    if chat_serialization == "simple-chatml":
        rendered = render_simple_chatml(messages, add_generation_prompt=add_generation_prompt)
        return list(tokenizer.encode(rendered, add_special_tokens=False))
    return apply_chat_template(
        backend,
        messages,
        add_generation_prompt=add_generation_prompt,
        template_kwargs=template_kwargs,
    )


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    cls = {
        "causal-lm": AutoModelForCausalLM,
        "multimodal-lm": AutoModelForMultimodalLM,
        "image-text-to-text": AutoModelForImageTextToText,
    }[args.model_class]
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    try:
        return cls.from_pretrained(args.model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return cls.from_pretrained(args.model_path, **kwargs)


def main() -> None:
    configure_cuda_device()
    args = parse_args()
    fsdp_config = load_fsdp_config(args.fsdp_config_json)
    template_kwargs = load_template_kwargs(args.chat_template_kwargs_json)
    use_fsdp_activation_checkpointing = bool(fsdp_config.get("activation_checkpointing"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_backend, tokenizer = load_template_backend(args.model_path)
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    if args.max_train_rows and len(dataset) > args.max_train_rows:
        rng = random.Random(args.sample_seed)
        indices = sorted(rng.sample(range(len(dataset)), args.max_train_rows))
        dataset = dataset.select(indices)

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("row must contain at least two chat messages")
        prompt_messages = messages[:-1]
        full_ids = encode_chat_messages(
            template_backend,
            tokenizer,
            messages,
            add_generation_prompt=False,
            chat_serialization=args.chat_serialization,
            template_kwargs=template_kwargs,
        )
        prompt_ids = encode_chat_messages(
            template_backend,
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            chat_serialization=args.chat_serialization,
            template_kwargs=template_kwargs,
        )
        if len(full_ids) > args.max_seq_length:
            full_ids = full_ids[-args.max_seq_length :]
            prompt_cut = max(0, len(prompt_ids) - args.max_seq_length)
            prompt_len = max(0, len(prompt_ids) - prompt_cut)
        else:
            prompt_len = len(prompt_ids)
        labels = list(full_ids)
        labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    def build_tokenized_dataset():
        return dataset.map(encode, remove_columns=dataset.column_names, num_proc=1)

    if args.tokenized_cache_dir:
        cache_dir = Path(args.tokenized_cache_dir)
        success_path = cache_dir / "_SUCCESS"
        lock_path = cache_dir.parent / f"{cache_dir.name}.lock"
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if success_path.exists():
            tokenized = load_from_disk(str(cache_dir))
        else:
            lock_fd = acquire_file_lock(lock_path)
            if lock_fd is None:
                wait_for_tokenized_cache(cache_dir, lock_path)
                if not success_path.exists():
                    lock_fd = acquire_file_lock(lock_path)
            if lock_fd is not None:
                try:
                    if cache_dir.exists() and not success_path.exists():
                        shutil.rmtree(cache_dir)
                    tokenized = build_tokenized_dataset()
                    tokenized.save_to_disk(str(cache_dir))
                    success_path.write_text("ok\n")
                finally:
                    os.close(lock_fd)
                    lock_path.unlink(missing_ok=True)
            else:
                tokenized = load_from_disk(str(cache_dir))
    else:
        tokenized = build_tokenized_dataset()

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    model = load_model(args)
    model.config.use_cache = False
    if not use_fsdp_activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.finetune_mode == "full":
        if args.sft_adapter_path:
            raise ValueError("--sft-adapter-path is only supported with --finetune-mode lora")
        for param in model.parameters():
            param.requires_grad_(True)
    elif args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=args.lora_rank,
                target_modules=parse_target_modules(args.target_modules),
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
            ),
        )

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": not use_fsdp_activation_checkpointing,
        "ddp_find_unused_parameters": False,
        "optim": args.optim,
    }
    if args.fsdp:
        training_kwargs["fsdp"] = args.fsdp
    if fsdp_config:
        training_kwargs["fsdp_config"] = fsdp_config

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=tokenized,
        data_collator=collate,
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
            )
            if checkpoints:
                resume_from_checkpoint = str(checkpoints[-1])
        else:
            resume_from_checkpoint = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = output_dir / ("final_model" if args.finetune_mode == "full" else "final_lora")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if template_backend is not tokenizer and hasattr(template_backend, "save_pretrained"):
        template_backend.save_pretrained(str(final_dir))
    run_config = vars(args) | {
        "final_artifact_dir": str(final_dir),
        "final_artifact_type": args.finetune_mode,
        "train_rows": len(dataset),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
