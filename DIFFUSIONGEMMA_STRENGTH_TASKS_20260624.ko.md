# DiffusionGemma 다음 학습 태스크 (2026-06-24)

## 결론

DiffusionGemma base를 TB2-lite next-action에 바로 넣은 것은 실행 검증으로는 의미가 있었지만, 모델이 잘하는 영역을 찌른 실험은 아니었다. corrected full run은 score `25.12`였고, 이는 “DiffusionGemma가 나쁘다”보다 “태스크 선택이 안 맞았다”에 가깝다.

다음 DiffusionGemma 학습은 **Fable terminal/tool-call 데이터는 유지**하되, dLLM 구조가 유리한 태스크로 재포맷한다.

1. structured JSON/tool-call repair
2. code/tool trace completion
3. constrained puzzle/grid sanity task
4. 이후 LiveCodeBench/self-repair 또는 function-calling benchmark로 확장

## 근거

공식 Google developer guide는 DiffusionGemma의 핵심을 256-token canvas 병렬 denoising, bidirectional context, self-correction으로 설명한다. 특히 Sudoku 같은 strict multivariable constraint 문제에서 fine-tuned adapter가 base 대비 크게 좋아지는 예시를 제시한다.

Google/HF model card는 DiffusionGemma가 25.2B total / 3.8B active MoE, canvas length 256, 256K context, function calling, coding/reasoning, multimodal OCR/document/screen parsing을 지원한다고 밝힌다. 벤치마크도 LiveCodeBench v6 `69.1`, Tau2 `56.2`, MMLU Pro `77.6`, GPQA Diamond `73.2`, MATH-Vision `70.5`가 공개되어 있다.

NVIDIA NeMo AutoModel 문서는 DiffusionGemma SFT/LoRA recipe가 final response turn을 canvas로 두고, supervised canvas positions를 uniform random vocabulary token으로 corrupt한 뒤 clean token을 복원하도록 학습한다고 설명한다. 이건 “깨진 구조화 출력 복구” 태스크와 잘 맞는다.

참고 링크:

- https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- https://huggingface.co/google/diffusiongemma-26B-A4B-it
- https://ai.google.dev/gemma/docs/diffusiongemma/model_card
- https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/dllm/google/diffusiongemma.html
- https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/dllm/diffusiongemma.md
- https://huggingface.co/datasets/livecodebench/code_generation
- https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench
- https://huggingface.co/datasets/beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched

## 학습 방법론

### 유지할 것

- Fable terminal/tool-call 데이터는 반드시 포함한다.
- TB2-lite는 계속 회귀 벤치로 남긴다.
- Docker는 쓰지 않는다.
- 학습은 NeMo AutoModel `DiffusionGemmaSFTRecipe` + LoRA로 간다.
- vLLM 직접 실행은 환경에서 가능한 시점에만 쓰고, 지금은 Transformers dLLM 평가 경로를 유지한다.

### 바꿀 것

기존 TB2-lite prompt는 “다음 command/action 하나”를 맞히는 autoregressive next-token 성격이 강했다. DiffusionGemma 다음 학습은 final response canvas 전체를 보고 고치는 태스크를 늘린다.

새 데이터 빌더:

- `scripts/build_diffusiongemma_strength_mix_20260624.py`

데이터 소스:

- `datasets/fable5_lfm_sft_20260623.jsonl`
- `datasets/glm52_chaser_terminal_toolmix_20260624.jsonl`
- `datasets/hermes_agent_traces_chat_20260624.jsonl`
- `datasets/phase2_reasoning_lfm_sft_20260623.jsonl`

생성되는 태스크:

- `original_terminal_or_tool_trace`: Fable/GLM52 terminal action과 tool trace 원형 유지
- `original_code_or_tool_trace`: code-fence가 있는 trace 유지
- `structured_repair`: JSON/tool-call final output을 일부 깨뜨린 뒤, context를 보고 원래의 valid output으로 복구하도록 학습

학습 config:

- `configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml`
- LoRA target: attention q/k/v/o + dense MLP gate/up/down
- MoE router frozen
- expert parallelism `EP=8`
- canvas length `256`
- sequence length `4096`
- max steps `1200`
- LR `1e-4`, cosine decay, warmup `50`

실행 runner:

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_diffusiongemma_strength_lora_20260624.sh
```

현재는 Docker 없이 NeMo AutoModel 가상환경에서 실행한다. 첫 retry는 step 199까지 정상 학습했지만 step 200 checkpoint에서 PEFT optimizer state가 `safetensors`에 맞지 않아 실패했다. 그래서 `scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py` wrapper를 추가해 PEFT adapter는 계속 저장하고 optimizer checkpoint만 건너뛰게 했다.

재시작 run은 `20260624_diffusiongemma_strength_lora_fable_structured_repair_nooptim_retry2`로 진행한다. 목표는 먼저 step 200 checkpoint를 통과하고, 이후 1200 step 완료 adapter로 structured repair/TB2-lite/code-tool probe를 평가하는 것이다.

## 다음 평가

학습 후 평가는 세 갈래로 본다.

| 평가 | 목적 | 성공 기준 |
| --- | --- | --- |
| structured repair probe | dLLM self-correction이 실제로 좋아졌는지 | valid JSON/tool-call 복구율 상승 |
| TB2-lite replay | Fable terminal 능력이 유지되는지 | base DiffusionGemma `25.12`보다 명확히 상승 |
| code/tool long probe | code/tool trace 생성 품질과 속도 | probe tok/s 유지, blank/format break 감소 |

## 이후 확장 후보

검색한 공개 데이터 기준으로 추가 후보는 아래가 좋다.

- `NousResearch/hermes-function-calling-v1`: function calling, json-mode, agentic JSON structured output
- `Salesforce/xlam-function-calling-60k`: query/tools/answers 구조의 function-calling 데이터
- `livecodebench/code_generation`: code generation, self-repair, test output prediction, code execution 평가 축
- `open-r1/verifiable-coding-problems-python`: unit-test로 검증 가능한 Python coding 문제
- `bluecoconut/pencil-puzzle-bench`: 94종 puzzle, verified unique solutions
- `beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched`: Sudoku 구조/난이도 feature 포함

우선순위는 `Fable + structured repair`가 1순위다. 그 다음에 Sudoku/Pencil Puzzle을 소량 섞어서 dLLM 강점 시연용 모델을 만들고, 마지막으로 LiveCodeBench/self-repair로 외부 코드 벤치 확장을 한다.
