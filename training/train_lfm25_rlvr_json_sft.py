#!/usr/bin/env python
"""LoRA SFT for strict-JSON Harness retrieval curation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--sft-adapter-path", default="")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--save-total-limit", type=int, default=0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-config-json", default="")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--finetune-mode",
        choices=("lora", "full"),
        default="lora",
        help="Use LoRA adapters by default. Use full for full-parameter SFT on models that fit in memory.",
    )
    return parser.parse_args()


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    if value.startswith("regex:"):
        return value.removeprefix("regex:")
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules if len(modules) > 1 else value


def configure_cuda_device() -> None:
    """Bind each torchrun worker before model/cuda contexts are created."""
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    torch.cuda.set_device(int(local_rank))


def load_fsdp_config(raw_config: str) -> dict[str, Any]:
    if not raw_config:
        return {}
    parsed = json.loads(raw_config)
    if not isinstance(parsed, dict):
        raise ValueError("--fsdp-config-json must decode to a JSON object")
    return parsed


def main() -> None:
    configure_cuda_device()
    args = parse_args()
    fsdp_config = load_fsdp_config(args.fsdp_config_json)
    use_fsdp_activation_checkpointing = bool(fsdp_config.get("activation_checkpointing"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        prompt_messages = messages[:-1]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(full_ids) > args.max_seq_length:
            full_ids = full_ids[-args.max_seq_length :]
            prompt_cut = max(0, len(prompt_ids) - args.max_seq_length)
            prompt_len = max(0, len(prompt_ids) - prompt_cut)
        else:
            prompt_len = len(prompt_ids)
        labels = list(full_ids)
        labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    tokenized = dataset.map(encode, remove_columns=dataset.column_names, num_proc=1)

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        pad_id = int(tokenizer.pad_token_id)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
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
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": not use_fsdp_activation_checkpointing,
        "ddp_find_unused_parameters": False,
        "optim": args.optim,
    }
    if args.save_total_limit > 0:
        training_kwargs["save_total_limit"] = args.save_total_limit
    if args.fsdp:
        training_kwargs["fsdp"] = args.fsdp
    if fsdp_config:
        training_kwargs["fsdp_config"] = fsdp_config

    training_args = TrainingArguments(
        **training_kwargs,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collate)
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
    run_config = vars(args) | {"final_artifact_dir": str(final_dir), "final_artifact_type": args.finetune_mode}
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
