# Fabliq 벤치마크 실행 런북 (2026-06-24)

> 업데이트: 이 문서는 2026-06-24 초반 Transformers 기반 평가 계획/초기 관측을 기록한 런북이다. 실제 기준 결과는 vLLM으로 재실행한 뒤 `fable_distillation/TB2_VLLM_BENCHMARK_20260624.ko.md`에 정리했다. 현재 후속 학습/평가 계획은 `fable_distillation/GLM52_CHASER_EXPERIMENT_20260624.ko.md`를 기준으로 본다.

## 목적

GLM-5.2 모델 카드의 Benchmark 섹션은 Reasoning / Coding / Agentic 축으로 성능을 제시한다. Fabliq는 범용 reasoning 모델이 아니라 Fable-5 terminal-agent 증류 모델이므로, 우선순위는 다음과 같이 둔다.

1. **Terminal / tool-use 계열**: GLM 표의 Terminal Bench 2.1, Tool-Decathlon, MCP-Atlas에 대응.
2. **Coding proxy**: 로컬에서 즉시 돌릴 수 있는 HumanEval / MBPP 계열.
3. **Reasoning / instruction proxy**: GPQA-Diamond, MMLU-Pro, IFEval.

## 1차 실행: TB2-lite full replay

- 실행 루트: `/home/work/.data/fabliq_benchmarks/20260624T002506Z`
- 벤치: `tb2_lite/data/replay_full.jsonl`
- 규모: 303 steps
- 러너: `tb2_lite/scripts/replay_eval_transformers.py`
- 생성 설정: `temperature=0`, `max_model_len=8192`, `max_tokens=256`, `batch_size=4`, `dtype=bfloat16`
- 결과 디렉터리: `/home/work/.data/fabliq_benchmarks/20260624T002506Z/tb2_full`
- 로그 디렉터리: `/home/work/.data/fabliq_benchmarks/20260624T002506Z/tb2_full/logs`
- 관측 속도: 8개 모델 모두 `[20/303]` 기준 약 255-261초, 약 `0.08 samples/s`
- 예상 소요: 303 steps 기준 약 65분 전후

| GPU | 모델 short | 모델 경로 |
| --- | --- | --- |
| 0 | `base` | `LiquidAI/LFM2.5-8B-A1B` |
| 1 | `toolbench` | `/home/work/.data/hf_upload_stage/lfm25_8b_a1b_toolbench_full/epoch1` |
| 2 | `phase1-fabliq` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model` |
| 3 | `phase1b-frombase` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model` |
| 4 | `phase2-reasoning` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model` |
| 5 | `mega-combined` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model` |
| 6 | `mega-lr2e6` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr2e6-FullSFT-20260623/final_model` |
| 7 | `mega-lr5e7` | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr5e7-FullSFT-20260623/final_model` |

## 1차 평가에서 볼 지표

`replay_eval_transformers.py`의 aggregate 결과 중 우선 볼 값:

- `next_action_score`
- `valid_json_pct`
- `first_cmd_exact_pct`
- `avg_command_f1`
- `complete_true_recall_pct`
- `premature_complete_rate_pct`
- source group별 점수
- early/mid/late step bucket별 점수

## 후속 wave 후보

TB2-lite full 결과가 끝난 뒤 GPU가 비는 순서대로 다음을 돌린다.

### Wave 2: GLM 표와 가까운 reasoning / coding proxy

TB2-lite full replay가 끝나면 watcher가 자동 실행한다.

- watcher 로그: `/home/work/.data/fabliq_benchmarks/20260624T002506Z/lm_eval_wave2_watcher.log`
- 런처: `fable_distillation/scripts/run_fabliq_lm_eval_wave2_20260624.sh`
- 대상 모델: TB2-lite 1차와 같은 8B 비교군 8개
- 결과 디렉터리: `/home/work/.data/fabliq_benchmarks/20260624T002506Z/lm_eval_wave2`

후보 태스크:

- `leaderboard_gpqa_diamond`
- `ifeval`
- `mmlu_pro_computer_science`

### Wave 3: 작은 모델 / size scaling

TB2-lite에서 8B 승자가 정해진 뒤 비교한다.

- `Fabliq-1.2B-Agent`
- `Fabliq-1.2B-Agent-Mega`
- `Fabliq-2.6B-Agent`
- 필요시 `Fabliq-1.2B-Thinking-*`

## 판정 기준

- GLM-5.2급 범용 모델을 GPQA/MMLU-Pro에서 이기는 것은 현실적으로 기대하기 어렵다.
- 승부처는 terminal/tool-use이며, TB2-lite에서 base/toolbench 대비 유의미하게 올라가면 Fabliq의 목적에는 성공이다.
- `mega-lr2e6`은 loss 최저였지만 기존 5-task eval에서 실제 성능이 낮았으므로, TB2-lite에서 재확인한다.
- TB2-lite 1차 순위는 속도를 위해 `max_tokens=256`으로 산출한다. 승자 2-3개는 필요하면 `max_tokens=1024`로 재검증한다.
- TB2-lite 결과가 나오기 전에는 HF 모델 카드의 "best model" 문구를 확정하지 않는다.

## 결과 반영 순서

1. TB2-lite full 결과 취합.
2. README / RESULTS_SUMMARY / FINAL_REPORT의 결론 충돌 정리.
3. 모델 카드와 GGUF 체크섬/벤치 표 업데이트.
4. 그 다음 HF 업로드/카드 정리 작업 진행.
