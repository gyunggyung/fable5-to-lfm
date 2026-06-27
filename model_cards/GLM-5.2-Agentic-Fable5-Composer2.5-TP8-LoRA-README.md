---
license: mit
base_model: zai-org/GLM-5.2
library_name: peft
tags:
- glm-5.2
- fable
- fable-5
- agentic
- tool-use
- terminal-agent
- lora
- megatron
- ms-swift
- tensor-parallel
datasets:
- Glint-Research/Fable-5-traces
- lordx64/agentic-distill-fable-5-sft
- armand0e/claude-fable-5-claude-code
pipeline_tag: text-generation
---

# GLM-5.2-Agentic-Fable5-Composer2.5-TP8-LoRA

Adapter-only Fable-style agentic tuning run for [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2).

This repo is intended to publish LoRA checkpoints, not a full copy of the GLM-5.2 base model. Load this adapter on top of the original base model according to the checkpoint format produced by Megatron-SWIFT / PEFT.

## Goal

Make GLM-5.2 behave more like the popular Fable/Mythos agentic models:

- terminal-oriented planning and command selection
- structured tool-call formatting
- repair of malformed tool-call JSON
- concise assistant behavior around shell/edit/read/write workflows
- Claude Code / Fable-style multi-step task traces

Reference behavior lines:

- [empero-ai/Qwythos-9B-Claude-Mythos-5-1M](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M)
- [empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF)
- [yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
- [yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)

## Training Setup

- Base: `zai-org/GLM-5.2`
- Method: LoRA SFT
- Runner: `ms-swift` Megatron-SWIFT
- Parallelism: `tensor_model_parallel_size=8`, `sequence_parallel=true`
- Hardware target: single node, 8x NVIDIA H200
- Initial sequence length: 2048
- Initial micro/global batch: 1 / 8
- Initial LoRA rank/alpha: 16 / 32
- Initial target modules: attention projections

The first production run intentionally starts with attention-projection LoRA to establish a working checkpoint on the 743B MoE model. If memory and throughput are stable, the follow-up run should expand target modules toward MLP/expert projections to strengthen Fable-style behavior.

## Training Data

The active training file is built inside the `fable_distillation` workspace:

```text
datasets/official_agentic_sft_mix_20260627.jsonl
```

Current mix size: 19,536 rows.

The mix is designed around Fable/agentic/tool-use behavior rather than generic chat instruction tuning. It includes Fable terminal/tool traces, structured tool-call repair examples, and agentic command/edit/read/write workflows prepared in the local project.

## Evaluation Plan

Primary public-facing benchmarks should target official agentic/coding/tool-use behavior, not only local toy metrics:

- GLM-5.2 model-card task family: long-horizon coding and agentic tool use
- Terminal-Bench / terminal next-action replay as a local regression signal
- MCP/tool-call JSON validity and recovery tests
- SWE-style coding repair smoke tasks when the adapter is stable

Local TB2-lite is only a regression check. It is not a substitute for official public benchmarks.

## Status

This model card is prepared before the first successful GLM-5.2 Fable LoRA checkpoint upload. Check the repository files and commit history for the latest checkpoint state.

## Limitations

- This is an adapter for a very large MoE base model; it is not a standalone model.
- Early checkpoints may improve style/tool-call formatting before they improve broad benchmark scores.
- The Fable-derived data may carry license constraints from upstream datasets. Review upstream dataset licenses before commercial use.
