# TB2-lite vLLM 평가 기록 (2026-06-24)

이 문서는 `fable_distillation/` 안에서 실행한 Fabliq 모델 vLLM 평가를 정리한다. GLM-5.2 모델 카드의 공개 벤치마크와 직접 같은 harness는 아니며, Fabliq 개발 중 빠르게 회귀/개선 여부를 보기 위한 로컬 터미널 next-action replay 평가다.

## 결론

- 현재 1위는 `phase2-reasoning` / **Fabliq-8B-Agent-Reasoning** 이다.
- 점수는 `51.59`, ToolBench 기반 baseline `51.46`보다 근소하게 높다.
- raw LFM base에서 바로 학습한 Mega/FromBase 계열은 loss가 낮아도 TB2-lite 행동 점수는 40-41대에 머물렀다.
- 따라서 다음 학습은 raw base가 아니라 **기존 1위 phase2-reasoning에서 이어 학습**해야 한다.

## 벤치마크 정의

- 평가 파일: `tb2_lite/data/replay_full.jsonl`
- 샘플 수: 303
- 평가 방식: 각 prompt에서 다음 assistant action을 생성하고, 정답 command/action과 비교한다.
- 주 지표:
  - `avg_command_f1`: 생성 command와 정답 command의 token/F1 계열 유사도
  - `first_cmd_exact`: 첫 command exact match
  - `valid_json`: JSON action 포맷 유효성
  - `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`

이 평가는 public Terminal-Bench 2.1의 Docker sandbox/task success 평가가 아니다. 대신 빠르게 모델별 터미널 행동 감각을 비교하는 local replay benchmark다.

## 왜 이 평가부터 봤나

GLM-5.2가 공개한 강점은 일반 지식보다 long-horizon coding/agentic 계열이다. 모델 카드 기준으로 GLM-5.2는 Terminal Bench 2.1, MCP-Atlas, Tool-Decathlon 같은 에이전트/툴 사용 지표를 핵심 벤치로 제시한다. Fabliq도 Fable-5 터미널 trace와 ToolBench 기반으로 만든 모델이므로, 가장 먼저 확인해야 할 축은 터미널 next-action과 tool-call 안정성이다.

참고 링크:

- GLM-5.2 benchmark: https://huggingface.co/zai-org/GLM-5.2#benchmark
- Terminal-Bench: https://www.tbench.ai/
- Terminal-Bench GitHub: https://github.com/harbor-framework/terminal-bench
- LiquidAI LFM2.5-8B-A1B: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- LiquidAI vLLM docs: https://docs.liquid.ai/deployment/gpu-inference/vllm

## vLLM 실행 환경

성공한 환경:

```bash
VLLM_ENV=/home/work/.projects/LLM-OS-Models/Terminal/.vllm-lfm-cu12
```

중요한 점:

- `env -u PYTHONPATH PYTHONNOUSERSITE=1`로 사용자 site/package 오염을 끊어야 한다.
- `.vllm-lfm-cu12`는 LFM2.5 tokenizer/model 로딩이 맞는 조합이다.
- `.vllm-step37-clean2`는 CUDA 13 계열 torch라 현재 driver/runtime 조합에서 실패했다.
- `.vllm-env`는 vLLM C extension은 동작하지만 LFM2.5 tokenizer backend 호환 문제가 있었다.

대표 실행:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

bash fable_distillation/scripts/run_fabliq_tb2_vllm_20260624.sh
bash fable_distillation/scripts/run_fabliq_tb2_vllm_wave2_20260624.sh
```

결과 파일:

- `fable_distillation/benchmarks/20260624_tb2_vllm/results/SUMMARY.md`
- `fable_distillation/benchmarks/20260624_tb2_vllm_wave2/results/SUMMARY.md`
- raw JSON: `fable_distillation/benchmarks/*/results/*.json`

## 전체 순위

| Rank | Model | Score | Cmd F1 | First Cmd | Valid JSON | Sec/Step | Load(s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `phase2-reasoning` | 51.59 | 0.5193 | 50.8% | 76.2% | 0.087 | 131.4 |
| 2 | `toolbench` | 51.46 | 0.5230 | 49.5% | 76.9% | 0.087 | 134.9 |
| 3 | `phase1-fabliq` | 49.31 | 0.5022 | 47.2% | 76.6% | 0.087 | 84.9 |
| 4 | `mega-5ep` | 41.47 | 0.4171 | 40.9% | 60.1% | 0.091 | 43.1 |
| 5 | `mega-combined` | 41.44 | 0.4081 | 42.9% | 55.4% | 0.092 | 80.4 |
| 6 | `phase2b-frombase-reasoning` | 41.22 | 0.4080 | 42.2% | 57.8% | 0.093 | 43.8 |
| 7 | `mega-10ep` | 41.06 | 0.4083 | 41.6% | 57.8% | 0.092 | 44.2 |
| 8 | `phase1b-frombase` | 40.88 | 0.3972 | 43.6% | 58.4% | 0.092 | 80.5 |
| 9 | `base` | 40.51 | 0.3992 | 41.9% | 59.1% | 0.093 | 81.1 |
| 10 | `mega-1ep` | 40.43 | 0.4036 | 40.6% | 60.1% | 0.092 | 44.6 |
| 11 | `mega-phase2m-reasoning` | 40.13 | 0.3993 | 40.6% | 58.7% | 0.093 | 43.7 |
| 12 | `mega-lr5e7` | 40.06 | 0.4069 | 38.6% | 57.8% | 0.093 | 80.9 |
| 13 | `mega-lr2e6` | 39.68 | 0.3929 | 40.6% | 59.1% | 0.092 | 80.7 |
| 14 | `fabliq-1.2b-mega` | 16.43 | 0.1782 | 13.2% | 52.8% | 0.062 | 31.5 |
| 15 | `fabliq-1.2b` | 15.95 | 0.1726 | 12.9% | 54.5% | 0.061 | 31.6 |
| 16 | `fabliq-1.2b-thinking` | 5.62 | 0.0605 | 4.6% | 17.5% | 0.073 | 31.7 |

## 해석

ToolBench 기반이 가장 중요했다. `toolbench`, `phase1-fabliq`, `phase2-reasoning`이 49-51점대이고, raw base 기반 모델은 대부분 40-41점대다.

`mega-lr2e6`은 학습 loss가 가장 낮았지만 TB2-lite 점수는 낮았다. 즉 이 프로젝트에서는 loss만으로 agentic terminal 성능을 고르면 안 된다.

1.2B 계열은 같은 평가에서 8B 대비 충분히 경쟁적이지 않다. 소형 모델은 별도 경량화/온디바이스 목적으로 분리해야 한다.

vLLM은 기존 Transformers 직접 평가보다 훨씬 빠르고 안정적이었다. 8B 모델 기준 실제 생성/채점은 약 `0.087-0.093 sec/step` 범위였고, cold load 포함 8개 모델 병렬 평가도 짧은 시간에 끝났다.

## 다음 실험 기준

새 모델은 반드시 `phase2-reasoning`의 `51.59`를 이겨야 한다. 다음 실험의 default baseline은 이 모델이다.

현재 진행 중인 후속 실험:

- 문서: `fable_distillation/GLM52_CHASER_EXPERIMENT_20260624.ko.md`
- 데이터: `fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.jsonl`
- 런처: `fable_distillation/scripts/run_glm52_chaser_mix_sft_20260624.sh`
