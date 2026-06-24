# GLM-5.2 추격 실험 로그 (2026-06-24)

목표는 GLM-5.2 전체 성능을 8B Fabliq로 정면 돌파하는 것이 아니라, Fabliq가 강점을 가질 수 있는 터미널/툴콜/에이전트 영역에서 하나라도 확실히 이기는 지표를 만드는 것이다.

## 왜 이 방향인가

GLM-5.2는 753B급 공개 모델이고, 모델 카드에서 1M context, long-horizon coding, Terminal Bench 2.1, MCP-Atlas, Tool-Decathlon을 핵심 강점으로 제시한다. Fabliq는 8B급 LFM2.5 기반 terminal-agent 모델이므로 HLE/GPQA/AIME 같은 일반 reasoning 지표로 이기는 것은 현실성이 낮다.

승산이 있는 축은 다음이다.

- 터미널 next-action imitation
- LFM tool-call 포맷 안정성
- JSON action/function-call 포맷 안정성
- 짧은 local terminal task에서의 빠른 실행/저비용
- 모델 크기 대비 agentic 행동 효율

참고 링크:

- GLM-5.2 benchmark: https://huggingface.co/zai-org/GLM-5.2#benchmark
- Hugging Face Datasets: https://huggingface.co/datasets
- Hermes Agent Reasoning Traces: https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces
- Hugging Face Agent Traces docs: https://huggingface.co/docs/hub/en/agent-traces
- Terminal-Bench: https://www.tbench.ai/

## 현재 기준선

로컬 TB2-lite vLLM replay 기준 1위:

| Model | Score | Cmd F1 | First Cmd | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| `phase2-reasoning` | 51.59 | 0.5193 | 50.8% | 76.2% |
| `toolbench` | 51.46 | 0.5230 | 49.5% | 76.9% |
| `phase1-fabliq` | 49.31 | 0.5022 | 47.2% | 76.6% |
| `base` | 40.51 | 0.3992 | 41.9% | 59.1% |

따라서 새 실험은 raw base가 아니라 `phase2-reasoning`에서 이어 학습한다.

자세한 기준선 문서:

- `fable_distillation/TB2_VLLM_BENCHMARK_20260624.ko.md`

## 현재 실행 중인 실험

실험명:

```text
20260624_glm52_chaser_mix
```

목표:

- `phase2-reasoning`의 TB2-lite `51.59`를 넘긴다.
- 터미널 action을 유지하면서 JSON/tool-call 안정성을 강화한다.
- GLM-5.2가 공개 벤치에서 강하게 내세운 MCP/Tool-Decathlon류 대비용 데이터 감각을 넣는다.

런처:

```bash
fable_distillation/scripts/run_glm52_chaser_mix_sft_20260624.sh
```

현재 PID:

```text
launcher pid: 2885269
torchrun pid: 2885773
```

로그:

```bash
tail -f fable_distillation/logs/20260624_glm52_chaser_mix/run.log
tail -f fable_distillation/logs/20260624_glm52_chaser_mix/train.log
nvidia-smi
```

출력 모델:

```text
/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624/final_model
```

학습 설정:

| 항목 | 값 |
| --- | --- |
| Base | `phase2-reasoning` final model |
| GPUs | H200 8장 |
| Fine-tune | full SFT |
| Max seq length | 16,384 |
| Epochs | 8 |
| LR | 2e-7 |
| Per-device batch | 2 |
| Grad accum | 4 |
| Global batch | 64 |
| Save steps | 100 |

학습이 끝나면 런처가 자동으로 vLLM TB2-lite 평가를 실행한다.

예상 결과 위치:

```text
fable_distillation/benchmarks/20260624_glm52_chaser_mix_tb2_vllm/results/SUMMARY.md
```

## 데이터 믹스

생성 스크립트:

```bash
fable_distillation/scripts/build_glm52_chaser_mix_20260624.py
```

출력:

```text
fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.jsonl
fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.meta.json
```

요약:

| Source | Kept |
| --- | ---: |
| `fable5_terminal` | 4,047 |
| `phase2_reasoning` | 281 |
| `harness_agentic_highrecall` | 4,140 |
| `harness_mixed_highrecall` | 815 |
| `harness_direct_json_highrecall` | 13 |
| `harness_local_hidden_search` | 2,000 |
| `harness_agentic_hardcase` | 120 |
| Total | 11,416 |

출력 타입:

| Final kind | Rows |
| --- | ---: |
| LFM tool-call | 3,287 |
| JSON action | 7,088 |
| Text/reasoning | 1,041 |

의도:

- Fable terminal trace로 Bash/Edit/Read/Write류 terminal action을 유지한다.
- Phase2 reasoning 281 rows로 기존 1위 모델의 reasoning 보강 방향을 유지한다.
- Harness JSON action trace로 function-call/JSON output discipline을 강화한다.

## HF 데이터 후보

바로 다음 후보는 `lambda/hermes-agent-reasoning-traces`다.

- Kimi-K2.5 trace: 7,646 samples
- GLM-5.1 trace: 7,055 samples
- 실제 multi-turn tool-call trajectory와 tool response가 포함되어 있다.

이 데이터는 GLM 계열이 잘하는 tool-call 행동을 직접 흡수하는 방향이라, 현재 SFT가 끝난 뒤 다음 실험에 넣을 가치가 높다. 단, 기존 TB2-lite terminal format과 다르므로 그대로 섞기보다 LFM tool-call 형식 또는 JSON action 형식으로 변환한 뒤 작은 LR로 추가 학습하는 것이 맞다.

Terminal-Bench public task 자체는 benchmark contamination 위험이 있으므로 학습 데이터에 넣으면 안 된다. Terminal-Bench는 평가용으로만 유지한다.

## RL 후보

강화학습은 가능하지만 바로 시작할 형태는 아니다. 필요한 구성은 다음이다.

| 구성 | 내용 |
| --- | --- |
| Policy init | 현재 1위 `phase2-reasoning` 또는 이번 chaser SFT 결과 |
| Rollout env | Terminal-Bench/로컬 TB2-lite task runner |
| Reward | valid action, command F1, first command exact, task success |
| Penalty | invalid JSON/tool-call, unsafe command, useless repeated command |
| 시작 방식 | offline rejection sampling 또는 DPO/KTO가 PPO/GRPO보다 빠름 |

실행 우선순위:

1. 지금 돌고 있는 full SFT 결과를 TB2-lite로 평가한다.
2. 이긴 checkpoint가 있으면 해당 checkpoint를 기준 모델로 고정한다.
3. 못 이기면 checkpoint-100/200/300을 vLLM으로 따로 평가해서 overfit 전 peak를 찾는다.
4. 그 다음 Hermes/Kimi/GLM tool traces를 변환해 낮은 LR 추가 SFT를 건다.
5. 마지막으로 Terminal-Bench 스타일 reward를 붙인 offline preference/RL을 설계한다.

## 판정 기준

우선 승리 조건:

```text
TB2-lite vLLM next_action_score > 51.59
```

추가로 봐야 할 보조 조건:

- `valid_json`이 76.2%보다 낮아지지 않는지
- `first_cmd_exact`이 50.8%보다 올라가는지
- JSON action 학습 때문에 LFM terminal tool-call 포맷이 무너지지 않는지
- vLLM eval에서 raw fallback 비율이 늘지 않는지

## 현재 상태 메모

- `2026-06-24T01:28:12Z`: full SFT 시작.
- 8개 torchrun worker가 실행 중.
- 초기 단계는 dataset tokenization/map이 진행되고 있어 VRAM이 아직 낮다.
- 학습 본 단계에 들어가면 GPU memory와 utilization이 올라가야 정상이다.
