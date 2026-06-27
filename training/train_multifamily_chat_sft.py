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
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model load dtype. Use auto for native quantized checkpoints such as FP8.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        choices=("true", "false"),
        default="false",
        help="Set TrainingArguments.ddp_find_unused_parameters. Use true for VLMs trained on text-only data.",
    )
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
    parser.add_argument(
        "--place-model-on-current-device-before-lora",
        action="store_true",
        help="Move the model to the current CUDA device before PEFT setup. Useful for DDP LoRA jobs on large GPUs.",
    )
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-config-json", default="")
    parser.add_argument("--deepspeed-config", default="")
    return parser.parse_args()


def configure_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        torch.cuda.set_device(int(local_rank))


def log_rank(message: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = os.environ.get("LOCAL_RANK", "?")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{timestamp}][rank={rank} local_rank={local_rank}] {message}", flush=True)


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


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    kwargs = dict(template_kwargs)
    if tools:
        kwargs["tools"] = tools
    result = backend.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
        **kwargs,
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
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    if chat_serialization == "simple-chatml":
        rendered = render_simple_chatml(messages, add_generation_prompt=add_generation_prompt)
        return list(tokenizer.encode(rendered, add_special_tokens=False))
    return apply_chat_template(
        backend,
        messages,
        add_generation_prompt=add_generation_prompt,
        template_kwargs=template_kwargs,
        tools=tools,
    )


def resolve_torch_dtype(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    cls = {
        "causal-lm": AutoModelForCausalLM,
        "multimodal-lm": AutoModelForMultimodalLM,
        "image-text-to-text": AutoModelForImageTextToText,
    }[args.model_class]
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": resolve_torch_dtype(args.torch_dtype),
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
    log_rank("starting SFT runner")
    fsdp_config = load_fsdp_config(args.fsdp_config_json)
    template_kwargs = load_template_kwargs(args.chat_template_kwargs_json)
    use_fsdp_activation_checkpointing = bool(fsdp_config.get("activation_checkpointing"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_rank(f"loading tokenizer/template backend: {args.model_path}")
    template_backend, tokenizer = load_template_backend(args.model_path)
    log_rank(f"loading dataset: {args.train_jsonl}")
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    if args.max_train_rows and len(dataset) > args.max_train_rows:
        rng = random.Random(args.sample_seed)
        indices = sorted(rng.sample(range(len(dataset)), args.max_train_rows))
        dataset = dataset.select(indices)

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("row must contain at least two chat messages")
        tools = row.get("tools")
        if not isinstance(tools, list):
            tools = None
        prompt_messages = messages[:-1]
        full_ids = encode_chat_messages(
            template_backend,
            tokenizer,
            messages,
            add_generation_prompt=False,
            chat_serialization=args.chat_serialization,
            template_kwargs=template_kwargs,
            tools=tools,
        )
        prompt_ids = encode_chat_messages(
            template_backend,
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            chat_serialization=args.chat_serialization,
            template_kwargs=template_kwargs,
            tools=tools,
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
            log_rank(f"loading tokenized cache: {cache_dir}")
            tokenized = load_from_disk(str(cache_dir))
        else:
            lock_fd = acquire_file_lock(lock_path)
            if lock_fd is None:
                log_rank(f"waiting for tokenized cache lock: {lock_path}")
                wait_for_tokenized_cache(cache_dir, lock_path)
                if not success_path.exists():
                    lock_fd = acquire_file_lock(lock_path)
            if lock_fd is not None:
                try:
                    if cache_dir.exists() and not success_path.exists():
                        shutil.rmtree(cache_dir)
                    log_rank(f"building tokenized cache: {cache_dir}")
                    tokenized = build_tokenized_dataset()
                    tokenized.save_to_disk(str(cache_dir))
                    success_path.write_text("ok\n")
                    log_rank(f"wrote tokenized cache: {cache_dir}")
                finally:
                    os.close(lock_fd)
                    lock_path.unlink(missing_ok=True)
            else:
                log_rank(f"loading tokenized cache after wait: {cache_dir}")
                tokenized = load_from_disk(str(cache_dir))
    else:
        log_rank("building tokenized dataset without disk cache")
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

    log_rank(f"loading model: {args.model_path}")
    model = load_model(args)
    log_rank("model loaded")
    if args.place_model_on_current_device_before_lora and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        log_rank(f"moving model to {device} before LoRA setup")
        model.to(device)
        log_rank(f"model moved to {device}")
    model.config.use_cache = False
    if not use_fsdp_activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        log_rank("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    if args.finetune_mode == "lora" and hasattr(model, "enable_input_require_grads"):
        log_rank("enabling input require grads")
        model.enable_input_require_grads()

    if args.finetune_mode == "full":
        if args.sft_adapter_path:
            raise ValueError("--sft-adapter-path is only supported with --finetune-mode lora")
        for param in model.parameters():
            param.requires_grad_(True)
    elif args.sft_adapter_path:
        log_rank(f"loading trainable LoRA adapter: {args.sft_adapter_path}")
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    else:
        log_rank(f"creating LoRA adapter: target_modules={args.target_modules} r={args.lora_rank}")
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
    log_rank("LoRA/model trainable setup complete")

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
        "ddp_find_unused_parameters": parse_bool(args.ddp_find_unused_parameters),
        "optim": args.optim,
    }
    if args.fsdp:
        training_kwargs["fsdp"] = args.fsdp
    if fsdp_config:
        training_kwargs["fsdp_config"] = fsdp_config
    if args.deepspeed_config:
        training_kwargs["deepspeed"] = args.deepspeed_config

    log_rank("creating Trainer")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=tokenized,
        data_collator=collate,
    )
    log_rank("Trainer created")

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
    log_rank(f"starting training resume_from_checkpoint={resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    log_rank("training complete")

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
