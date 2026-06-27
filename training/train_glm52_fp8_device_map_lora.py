#!/usr/bin/env python
"""Single-process model-parallel LoRA/QLoRA SFT for GLM-5.2.

The torchrun/ZeRO-3 path is still useful for smaller dense models, but the
GLM-5.2-FP8 checkpoint currently loads as CPU-side FP8 weights before ZeRO can
partition it. This runner uses Transformers' device_map path instead: one
process shards the frozen base over all visible GPUs and trains only LoRA
weights with a small manual loop. For trainable GLM runs, prefer the BF16
checkpoint with --load-in-4bit; the native FP8 checkpoint is inference-oriented
and its fine-grained FP8 matmul path currently has no autograd formula.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multifamily_chat_sft import (  # noqa: E402
    encode_chat_messages,
    load_template_backend,
    load_template_kwargs,
    parse_target_modules,
    patch_config_compat,
    resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenized-cache-dir", default="")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=52)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.02)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-max-memory-gib", type=int, default=132)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=64)
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--bnb-4bit-use-double-quant", choices=("true", "false"), default="true")
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument("--resume-from-checkpoint", default="auto")
    return parser.parse_args()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def visible_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    return torch.cuda.device_count()


def initialize_cuda_devices() -> None:
    """Initialize CUDA contexts on the main thread before HF weight loading.

    Transformers 5.x can materialize safetensors slices through a thread pool.
    On this host that path fails with CUDA initialization errors for GLM FP8, so
    the launcher disables async load and this warmup makes the sync path explicit.
    """
    if not torch.cuda.is_available():
        return
    original_device = torch.cuda.current_device()
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        torch.empty((1,), device=f"cuda:{index}")
    torch.cuda.set_device(original_device)


def build_max_memory(args: argparse.Namespace) -> dict[int | str, str]:
    max_memory: dict[int | str, str] = {}
    for index in range(visible_gpu_count()):
        max_memory[index] = f"{args.gpu_max_memory_gib}GiB"
    if args.cpu_max_memory_gib > 0:
        max_memory["cpu"] = f"{args.cpu_max_memory_gib}GiB"
    return max_memory


def build_quantization_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=resolve_torch_dtype(args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant == "true",
    )


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def read_state(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {"global_step": 0}
    state_path = checkpoint_dir / "manual_trainer_state.json"
    if not state_path.exists():
        return {"global_step": 0}
    return json.loads(state_path.read_text())


def trim_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    for checkpoint in checkpoints[:-save_total_limit]:
        for child in sorted(checkpoint.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        checkpoint.rmdir()


def encode_dataset(args: argparse.Namespace, template_backend: Any, tokenizer: Any):
    template_kwargs = load_template_kwargs(args.chat_template_kwargs_json)
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
            chat_serialization="native",
            template_kwargs=template_kwargs,
            tools=tools,
        )
        prompt_ids = encode_chat_messages(
            template_backend,
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            chat_serialization="native",
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

    if args.tokenized_cache_dir:
        cache_dir = Path(args.tokenized_cache_dir)
        success_path = cache_dir / "_SUCCESS"
        if success_path.exists():
            log(f"loading tokenized cache: {cache_dir}")
            tokenized = load_from_disk(str(cache_dir))
        else:
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            log(f"building tokenized cache: {cache_dir}")
            tokenized = dataset.map(encode, remove_columns=dataset.column_names, num_proc=1)
            tokenized.save_to_disk(str(cache_dir))
            success_path.write_text("ok\n")
    else:
        log("building tokenized dataset without disk cache")
        tokenized = dataset.map(encode, remove_columns=dataset.column_names, num_proc=1)
    return tokenized, len(dataset)


def make_collate(tokenizer: Any):
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

    return collate


def input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    for param in embeddings.parameters():
        return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    train_rows: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    state = {
        "global_step": step,
        "saved_at": utc_now(),
        "train_rows": train_rows,
        "args": vars(args),
    }
    (checkpoint_dir / "manual_trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "last_checkpoint").write_text(str(checkpoint_dir) + "\n")
    log(f"saved checkpoint: {checkpoint_dir}")
    trim_checkpoints(output_dir, args.save_total_limit)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    initialize_cuda_devices()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading tokenizer/template backend: {args.model_path}")
    template_backend, tokenizer = load_template_backend(args.model_path)
    tokenized, train_rows = encode_dataset(args, template_backend, tokenizer)

    resume_checkpoint: Path | None = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_checkpoint = find_latest_checkpoint(output_dir)
        else:
            resume_checkpoint = Path(args.resume_from_checkpoint)
    state = read_state(resume_checkpoint)
    start_step = int(state.get("global_step") or 0)
    if resume_checkpoint:
        log(f"resume checkpoint: {resume_checkpoint} start_step={start_step}")

    quantization_config = build_quantization_config(args)
    quantization_label = "4bit QLoRA" if quantization_config is not None else "native"
    log(f"loading GLM base with device_map={args.device_map} quantization={quantization_label}")
    config = patch_config_compat(AutoConfig.from_pretrained(args.model_path, trust_remote_code=True))
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "config": config,
        "dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
        "max_memory": build_max_memory(args),
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.offload_folder:
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = args.offload_folder
    log(f"max_memory={kwargs['max_memory']}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        log("preparing k-bit model for LoRA training")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        log("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    if not args.load_in_4bit and hasattr(model, "enable_input_require_grads"):
        log("enabling input require grads")
        model.enable_input_require_grads()

    if resume_checkpoint:
        log(f"loading trainable LoRA adapter: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, str(resume_checkpoint), is_trainable=True)
    else:
        log(f"creating LoRA adapter target_modules={args.target_modules} r={args.lora_rank}")
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
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_trainable = sum(param.numel() for param in trainable_params)
    log(f"trainable_params={total_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    data_loader = DataLoader(
        tokenized,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
        drop_last=False,
    )
    first_device = input_device(model)
    log(f"input_device={first_device}")

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_steps:
        for batch in data_loader:
            if global_step >= args.max_steps:
                break
            batch = move_batch(batch, first_device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            update_due = (global_step + 1) % args.gradient_accumulation_steps == 0
            if update_due:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.logging_steps == 0:
                used = [torch.cuda.memory_allocated(i) // (1024**3) for i in range(torch.cuda.device_count())]
                log(f"step={global_step} loss={loss.item() * args.gradient_accumulation_steps:.6f} gpu_alloc_gib={used}")
            if global_step % args.save_steps == 0:
                save_adapter(model, tokenizer, output_dir, global_step, args, train_rows)

    final_dir = output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "final_artifact_dir": str(final_dir),
                "final_artifact_type": "lora",
                "train_rows": train_rows,
                "completed_at": utc_now(),
                "args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    log(f"training complete final_lora={final_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
