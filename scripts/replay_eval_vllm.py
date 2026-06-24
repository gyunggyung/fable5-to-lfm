#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import (
    aggregate_scores,
    parse_prediction,
    score_commands,
    step_bucket,
)
from prompt_builder import build_prompts, sanitize_name


def engine_accepts_kwarg(name: str) -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs

        return name in inspect.signature(EngineArgs.__init__).parameters
    except Exception:
        return False


def parse_chat_template_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    mode = str(args.thinking_mode).lower()
    if mode in {"on", "true", "1", "yes", "enabled"}:
        kwargs["enable_thinking"] = True
    elif mode in {"off", "false", "0", "no", "disabled"}:
        kwargs["enable_thinking"] = False
    else:
        identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
        if "gemma-4" in identity:
            # TB2-lite expects the model answer itself to be JSON. The official
            # Gemma 4 template supports thinking, but disabling it keeps the
            # generated JSON in the assistant content instead of a thought channel.
            kwargs["enable_thinking"] = False
    return kwargs


def is_gemma4_model(args: argparse.Namespace, tokenizer: Any) -> bool:
    identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
    return "gemma-4" in identity


def gemma4_has_nonthinking_channel(args: argparse.Namespace, tokenizer: Any) -> bool:
    identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
    return "gemma-4-26b" in identity or "gemma-4-31b" in identity


def parse_prompt_options(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    gemma4 = is_gemma4_model(args, tokenizer)

    strip_mode = str(args.strip_thinking_history).lower()
    if strip_mode in {"on", "true", "1", "yes", "enabled"}:
        strip_thinking_history = True
    elif strip_mode in {"off", "false", "0", "no", "disabled"}:
        strip_thinking_history = False
    else:
        strip_thinking_history = gemma4

    channel_mode = str(args.gemma4_empty_thought_channel).lower()
    if channel_mode in {"on", "true", "1", "yes", "enabled"}:
        empty_thought_channel = True
    elif channel_mode in {"off", "false", "0", "no", "disabled"}:
        empty_thought_channel = False
    else:
        empty_thought_channel = gemma4 and gemma4_has_nonthinking_channel(args, tokenizer)

    return {
        "strip_thinking_history": strip_thinking_history,
        "gemma4_empty_thought_channel": empty_thought_channel,
        "use_gemma4_patched_template": gemma4 and (strip_thinking_history or empty_thought_channel),
    }


def load_rows(
    path: Path,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if shard_count > 1 and row_idx % shard_count != shard_index:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def add_prompt_length_meta(tokenizer: Any, rows: list[dict], prompts: list[str], args: argparse.Namespace, prompt_meta: dict) -> None:
    lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    too_long = [
        {
            "idx": idx,
            "task_id": rows[idx].get("task_id"),
            "step_idx": rows[idx].get("step_idx"),
            "prompt_tokens": length,
        }
        for idx, length in enumerate(lengths)
        if length + args.max_tokens > args.max_model_len
    ]
    if too_long:
        raise RuntimeError(
            "prompt context overflow before generation: "
            f"max_model_len={args.max_model_len} max_tokens={args.max_tokens} "
            f"overflow_count={len(too_long)} examples={too_long[:10]}"
        )
    sorted_lengths = sorted(lengths)
    prompt_meta.update(
        {
            "row_count": len(rows),
            "messages_rows": sum(isinstance(row.get("messages"), list) and bool(row.get("messages")) for row in rows),
            "prompt_tokens_min": min(lengths) if lengths else 0,
            "prompt_tokens_max": max(lengths) if lengths else 0,
            "prompt_tokens_p50": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
            "prompt_tokens_p95": sorted_lengths[int(len(sorted_lengths) * 0.95) - 1] if sorted_lengths else 0,
            "prompt_tokens_p99": sorted_lengths[int(len(sorted_lengths) * 0.99) - 1] if sorted_lengths else 0,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
        }
    )


def load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        if "Step-3.5-Flash-FP8" in tokenizer_path or "Step_hyphen_3_dot_5" in str(exc):
            from huggingface_hub import snapshot_download

            local_path = Path(snapshot_download(tokenizer_path, local_files_only=True))
            return PreTrainedTokenizerFast(
                tokenizer_file=str(local_path / "tokenizer.json"),
                tokenizer_config_file=str(local_path / "tokenizer_config.json"),
            )
        if "deepseek_v4" not in str(exc).lower():
            raise
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def build_llm(args: argparse.Namespace) -> tuple[LLM, dict]:
    tokenizer_path = args.tokenizer_path or args.model
    kwargs: dict = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
    }
    optional_string_kwargs = {
        "model_impl": args.model_impl if args.model_impl and args.model_impl != "auto" else "",
        "tokenizer_mode": args.tokenizer_mode,
        "mamba_cache_dtype": args.mamba_cache_dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "moe_backend": args.moe_backend,
        "attention_backend": args.attention_backend,
        "reasoning_parser": args.reasoning_parser,
        "quantization": args.quantization,
    }
    for key, value in optional_string_kwargs.items():
        if value and engine_accepts_kwarg(key):
            kwargs[key] = value
    optional_int_kwargs = {
        "block_size": args.block_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "data_parallel_size": args.data_parallel_size,
        "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
    }
    for key, value in optional_int_kwargs.items():
        if value is not None and engine_accepts_kwarg(key):
            kwargs[key] = value
    if args.hf_config_path and engine_accepts_kwarg("hf_config_path"):
        kwargs["hf_config_path"] = args.hf_config_path
    if args.hf_overrides_json and engine_accepts_kwarg("hf_overrides"):
        kwargs["hf_overrides"] = json.loads(args.hf_overrides_json)
    if args.speculative_config_json and engine_accepts_kwarg("speculative_config"):
        kwargs["speculative_config"] = json.loads(args.speculative_config_json)
    if args.engine_kwargs_json:
        extra_kwargs = json.loads(args.engine_kwargs_json)
        if not isinstance(extra_kwargs, dict):
            raise ValueError("--engine-kwargs-json must decode to a JSON object")
        kwargs.update(extra_kwargs)
    if args.enable_expert_parallel and engine_accepts_kwarg("enable_expert_parallel"):
        kwargs["enable_expert_parallel"] = True
    if args.disable_custom_all_reduce and engine_accepts_kwarg("disable_custom_all_reduce"):
        kwargs["disable_custom_all_reduce"] = True
    if engine_accepts_kwarg("limit_mm_per_prompt") and args.language_model_only:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    if engine_accepts_kwarg("language_model_only") and args.language_model_only:
        kwargs["language_model_only"] = True
    if engine_accepts_kwarg("skip_mm_profiling") and args.language_model_only:
        kwargs["skip_mm_profiling"] = True
    if engine_accepts_kwarg("disable_chunked_mm_input") and args.language_model_only and not args.keep_chunked_mm_input:
        kwargs["disable_chunked_mm_input"] = True
    if engine_accepts_kwarg("enforce_eager") and args.enforce_eager:
        kwargs["enforce_eager"] = True
    if engine_accepts_kwarg("enable_prefix_caching") and args.enable_prefix_caching:
        kwargs["enable_prefix_caching"] = True
    if args.lora_path:
        if engine_accepts_kwarg("enable_lora"):
            kwargs["enable_lora"] = True
        if engine_accepts_kwarg("max_loras"):
            kwargs["max_loras"] = 1
        if engine_accepts_kwarg("max_lora_rank"):
            kwargs["max_lora_rank"] = args.max_lora_rank
    if engine_accepts_kwarg("enable_chunked_prefill") and args.disable_chunked_prefill:
        kwargs["enable_chunked_prefill"] = False
    if engine_accepts_kwarg("async_scheduling") and args.disable_async_scheduling:
        kwargs["async_scheduling"] = False
    return LLM(**kwargs), kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stop-string", action="append", default=[])
    parser.add_argument("--include-stop-str-in-output", action="store_true")
    parser.add_argument("--no-skip-special-tokens", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--strip-thinking-history", default="auto")
    parser.add_argument("--gemma4-empty-thought-channel", default="auto")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--tokenizer-mode", default="")
    parser.add_argument("--mamba-cache-dtype", default="")
    parser.add_argument("--kv-cache-dtype", default="")
    parser.add_argument("--moe-backend", default="")
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--reasoning-parser", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
    parser.add_argument("--hf-config-path", default="")
    parser.add_argument("--hf-overrides-json", default="")
    parser.add_argument("--speculative-config-json", default="")
    parser.add_argument("--engine-kwargs-json", default="")
    parser.add_argument("--sampling-params-json", default="")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-async-scheduling", action="store_true")
    parser.add_argument("--keep-chunked-mm-input", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-name", default="adapter")
    parser.add_argument("--lora-id", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    rows = load_rows(eval_path, args.limit, args.shard_index, args.shard_count)

    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    chat_template_kwargs = parse_chat_template_kwargs(args, tokenizer)
    prompt_options = parse_prompt_options(args, tokenizer)
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.model,
        chat_template_kwargs=chat_template_kwargs,
        prompt_options=prompt_options,
    )
    if prompt_meta.get("template_status_counts", {}).get("raw_fallback") and not args.allow_raw_fallback:
        raise RuntimeError(f"raw prompt fallback occurred: {prompt_meta}")
    add_prompt_length_meta(tokenizer, rows, prompts, args, prompt_meta)

    load_start = time.time()
    llm, llm_kwargs = build_llm(args)
    load_time = round(time.time() - load_start, 1)

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "skip_special_tokens": not args.no_skip_special_tokens,
    }
    if args.stop_string:
        sampling_kwargs["stop"] = args.stop_string
        sampling_kwargs["include_stop_str_in_output"] = args.include_stop_str_in_output
    if args.min_p > 0:
        sampling_kwargs["min_p"] = args.min_p
    if args.sampling_params_json:
        extra_sampling = json.loads(args.sampling_params_json)
        if not isinstance(extra_sampling, dict):
            raise ValueError("--sampling-params-json must decode to a JSON object")
        sampling_kwargs.update(extra_sampling)
    sampling = SamplingParams(**sampling_kwargs)

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(args.lora_name, args.lora_id, args.lora_path)

    gen_start = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling,
        use_tqdm=True,
        lora_request=lora_request,
    )
    gen_time = round(time.time() - gen_start, 1)

    per_step: list[dict] = []
    for row, output in zip(rows, outputs):
        completion = output.outputs[0] if output.outputs else None
        pred_text = completion.text if completion else ""
        pred = parse_prediction(pred_text)
        ref = parse_prediction(row["ref_raw"])
        first_exact, precision, recall, f1 = score_commands(
            pred["command_units"], ref["command_units"]
        )
        pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
        per_step.append(
            {
                "task_id": row["task_id"],
                "sample_idx": row["sample_idx"],
                "step_idx": row["step_idx"],
                "bucket": step_bucket(row["step_idx"]),
                "source_group": row["source_group"],
                "valid_json": pred["valid_json"],
                "has_analysis": pred["has_analysis"],
                "has_plan": pred["has_plan"],
                "ref_task_complete": bool(ref["task_complete"]),
                "pred_task_complete": pred["task_complete"],
                "pred_task_complete_true": pred_complete_true,
                "ref_command_units": ref["command_units"],
                "pred_command_units": pred["command_units"],
                "first_cmd_exact": first_exact,
                "command_precision": round(precision, 4),
                "command_recall": round(recall, 4),
                "command_f1": round(f1, 4),
                "pred_preview": pred_text[:1200],
                "finish_reason": getattr(completion, "finish_reason", None) if completion else None,
                "stop_reason": getattr(completion, "stop_reason", None) if completion else None,
                "token_ids_preview": list(getattr(completion, "token_ids", [])[:64]) if completion else [],
            }
        )

    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_short or args.model,
        "model_path": args.model,
        "lora_path": args.lora_path or None,
        "lora_name": args.lora_name if args.lora_path else None,
        "model_short": model_short,
        "gpu": str(args.gpu),
        "eval_path": str(eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking_mode": args.thinking_mode,
            "strip_thinking_history": args.strip_thinking_history,
            "gemma4_empty_thought_channel": args.gemma4_empty_thought_channel,
            "dtype": args.dtype,
            "backend": args.backend,
            "model_impl": args.model_impl,
            "tp": args.tp,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
            "language_model_only": args.language_model_only,
            "tokenizer_mode": args.tokenizer_mode,
            "mamba_cache_dtype": args.mamba_cache_dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "moe_backend": args.moe_backend,
            "attention_backend": args.attention_backend,
            "reasoning_parser": args.reasoning_parser,
            "quantization": args.quantization,
            "block_size": args.block_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "data_parallel_size": args.data_parallel_size,
            "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
            "skip_special_tokens": not args.no_skip_special_tokens,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "stop_string": args.stop_string,
            "include_stop_str_in_output": args.include_stop_str_in_output,
            "hf_config_path": args.hf_config_path,
            "hf_overrides_json": args.hf_overrides_json,
            "speculative_config_json": args.speculative_config_json,
            "engine_kwargs_json": args.engine_kwargs_json,
            "sampling_params_json": args.sampling_params_json,
            "enable_expert_parallel": args.enable_expert_parallel,
            "enable_prefix_caching": args.enable_prefix_caching,
            "disable_chunked_prefill": args.disable_chunked_prefill,
            "disable_async_scheduling": args.disable_async_scheduling,
            "keep_chunked_mm_input": args.keep_chunked_mm_input,
            "lora_path": args.lora_path,
            "lora_name": args.lora_name,
            "lora_id": args.lora_id,
            "llm_kwargs": llm_kwargs,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
