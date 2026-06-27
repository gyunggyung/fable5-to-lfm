---
license: mit
base_model: zai-org/GLM-5.2
library_name: peft
tags:
- glm-5.2
- glm-5.2-fp8
- fable
- fable-5
- agentic
- tool-use
- terminal-agent
- lora
- megatron
- ms-swift
- model-parallel
datasets:
- Glint-Research/Fable-5-traces
- lordx64/agentic-distill-fable-5-sft
- armand0e/claude-fable-5-claude-code
pipeline_tag: text-generation
---

# GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA

Adapter-only Fable-style agentic tuning line for [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), using the [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) checkpoint as the local serving/loading target where possible.

This repository is for LoRA adapter checkpoints and model-card metadata only. It does not contain a full copy of the 743B MoE base model.

## Goal

Make GLM-5.2 behave more like the popular Fable/Mythos agentic models:

- terminal-oriented planning and command selection
- structured tool-call formatting
- repair of malformed tool-call JSON
- concise assistant behavior around shell/edit/read/write workflows
- Claude Code / Fable-style multi-step task traces
- lower-friction tool-use behavior without weakening base GLM reasoning

Reference behavior lines:

- [empero-ai/Qwythos-9B-Claude-Mythos-5-1M](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M)
- [empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF)
- [yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
- [yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)

## Current State

No GLM-5.2 Fable adapter weights have been published yet.

The project has already completed data preparation, vLLM GLM-5.2-FP8 serving/probing, Axolotl 8-bit/4-bit attempts, and Megatron-SWIFT setup. The local 8xH200 Megatron-SWIFT runs currently fail before the first training step because the stack instantiates BF16 training parameters for this 743B MoE model and reaches about 140GiB per GPU. The FP8 checkpoint reduces storage/serving memory, but the current training stack does not instantiate trainable FP8 parameters directly.

This card is kept in the target repository so checkpoints can be uploaded incrementally as soon as a runnable adapter path is found.

## Training Setup

- Base: `zai-org/GLM-5.2`
- Local checkpoint target: `zai-org/GLM-5.2-FP8`
- Method: LoRA SFT / adapter-only publishing
- Runner: `ms-swift` Megatron-SWIFT
- Current parallelism attempt: `tensor_model_parallel_size=4`, `pipeline_model_parallel_size=2`, `expert_model_parallel_size=4`, `sequence_parallel=true`
- Hardware target: single node, 8x NVIDIA H200
- Initial sequence length: 2048
- Initial micro/global batch: 1 / 8
- Initial LoRA rank/alpha: 16 / 32
- Initial target modules: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`

The first working run should stay attention-only to establish a checkpoint. If memory and throughput are stable, the next run should expand toward MLP/expert projections and a stronger tool-call repair curriculum.

## Training Data

The active training file is built inside the `fable_distillation` workspace:

```text
datasets/official_agentic_sft_mix_20260627.jsonl
```

Current mix size: 19,536 rows.

The converted Megatron-SWIFT agent-format file is:

```text
datasets/official_agentic_sft_mix_20260627.swift_agent.jsonl
```

That conversion currently keeps 14,374 rows and skips 5,162 rows that do not fit the strict agent/tool-call schema.

The mix is designed around Fable/agentic/tool-use behavior rather than generic chat instruction tuning. It includes Fable terminal/tool traces, structured tool-call repair examples, and agentic command/edit/read/write workflows prepared in the local project.

## Evaluation Plan

Primary public-facing benchmarks should target official agentic/coding/tool-use behavior, not only local toy metrics:

- MCP-Atlas public subset
- Tool-Decathlon official evaluation service
- Terminal-Bench 2.1 / Terminus-style terminal task evaluation
- Terminal-Bench / terminal next-action replay as a local regression signal
- MCP/tool-call JSON validity and recovery tests
- SWE-style coding repair smoke tasks when the adapter is stable

Local TB2-lite is only a regression check. It is not a substitute for official public benchmarks.

## Publishing Plan

- Upload the README/model card first.
- Upload every `adapter_model.safetensors` or `adapter_model.bin` checkpoint produced under the GLM-5.2 Fable output directories.
- Keep this repo adapter-only. Do not upload GLM-5.2 base shards here.
- Record benchmark results in the model card only after a checkpoint has completed evaluation.

## Limitations

- This is an adapter for a very large MoE base model; it is not a standalone model.
- Early checkpoints may improve style/tool-call formatting before they improve broad benchmark scores.
- The Fable-derived data may carry license constraints from upstream datasets. Review upstream dataset licenses before commercial use.
- At the time this README was prepared, no completed GLM-5.2 Fable adapter checkpoint exists from the local 8xH200 experiments.
