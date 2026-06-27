# Data Cleanup, Git, and GLM Training Options (2026-06-27)

작성 시각: 2026-06-27 05:31 UTC / 2026-06-27 14:31 KST

## 현재 디스크/GPU

```text
/home/work        2.9T total, 2.3T used, 513G free, 82%
/home/work/.data  24T total, 24T used, 275G free, 99%
/home/work/.projects 24T total, 24T used, 275G free, 99%
```

GPU:

```text
8x NVIDIA H200, each 143771 MiB total, currently ~1 MiB used
```

즉 GPU는 비어 있지만 `.data` 여유가 `275G`라 새 대형 모델/데이터 다운로드나 merge 저장은 매우 위험하다. 먼저 `.data` 정리가 필요하다.

## 삭제 후보

아직 삭제하지 않았다. 아래는 “삭제하면 확보되는 용량”과 “위험도” 기준 후보이다.

| 우선순위 | 경로 | 용량 | 판단 |
| --- | --- | ---: | --- |
| 1 | `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2` | 1.4T | 가장 좋은 삭제 후보. BF16 GLM-5.2 QLoRA가 로컬에서 OOM으로 막혔으므로 당장 학습에 못 쓴다. 다시 필요하면 재다운로드 가능. |
| 2 | `/home/work/.data/huggingface/hub/datasets--nvidia--Nemotron-Cascade-2-SFT-Data` | 555G | 현재 Fable/GLM 실험과 직접 연결이 확인되지 않으면 삭제 후보. |
| 3 | `/home/work/.data/huggingface/hub/LGAI-EXAONE-K-EXAONE-236B-A23B` | 444G | 현재 작업과 무관하면 삭제 후보. |
| 4 | `/home/work/.data/huggingface/hub/datasets--sapientinc--HRM-Text-data-io-cleaned-20260515` | 328G | 현재 작업과 무관하면 삭제 후보. |
| 5 | `/home/work/.data/harness1/models/LFM2.5-8B-A1B__ko_legal_source_bar_full_trial_20260621_v1` | 290G | 예전 legal trial 산출물. 이번 Fable/GLM 작업과 무관하면 삭제 후보. |
| 6 | `/home/work/.data/huggingface/hub/datasets--nvidia--Nemotron-MIND` | 247G | 현재 작업과 무관하면 삭제 후보. |
| 7 | `/home/work/.data/huggingface/hub/LTX-Video` | 238G | 텍스트/agent 실험과 무관하면 삭제 후보. |
| 8 | `/home/work/.data/huggingface/hub/datasets--nvidia--Nemotron-Math-v2` | 192G | 현재 공식 agent benchmark 준비와 무관하면 삭제 후보. |
| 9 | `/home/work/.data/huggingface/hub/datasets--nvidia--Nemotron-Post-Training-Dataset-v1` | 191G | 현재 작업과 무관하면 삭제 후보. |
| 10 | `/home/work/.data/huggingface/hub/datasets--nvidia--Nemotron-Competitive-Programming-v1` | 178G | coding 데이터라 나중에 쓸 수 있지만, 당장 공간이 급하면 후보. |

유지 권장:

| 경로 | 용량 | 이유 |
| --- | ---: | --- |
| `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8` | 707G | GLM-5.2 FP8 vLLM serving/eval/teacher 용도. 현재 로컬에서 성공한 GLM 경로다. |
| `/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-Merged-20260624` | 18G | Qwen GLM chaser 후속 평가/수정 후보. |
| DiffusionGemma LoRA adapter dirs | 작음 | 이미 학습된 adapter 산출물. |

`/home/work/.data/harness1/models` 대형 산출물:

```text
290G LFM2.5-8B-A1B__ko_legal_source_bar_full_trial_20260621_v1
111G LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623
111G LFM2.5-8B-A1B__Terminal-Mega-Phase2M-Reasoning-FullSFT-20260623
111G LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623
111G LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624
111G LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623
111G LFM2.5-8B-A1B__Terminal-Fable5-FromBase-Phase2B-Reasoning-FullSFT-20260623
111G LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623
```

이 111G급 LFM full-SFT 모델들은 결과 비교에 필요하면 보존하고, Hugging Face 업로드가 끝났거나 재현 가능하면 삭제 후보가 된다. 삭제 전에는 각 모델의 score/result JSON과 HF 업로드 여부를 확인해야 한다.

## git 상태

`pull`은 하지 않았다. 먼저 안전하게 백업 브랜치를 만들고 `fetch`만 했다.

백업 브랜치:

```text
backup/before-pull-20260627T053156Z
```

현재 상태:

```text
main...origin/main [ahead 9]
```

즉 원격이 앞서 있어서 pull이 필요한 상태가 아니라, 로컬이 `origin/main`보다 9커밋 앞서 있다. `git pull`이 아니라 `git push`가 필요한 상태다. 다만 push는 GitHub HTTPS credential이 없어서 실패한다.

남은 untracked:

```text
scripts/run_fabliq_lm_eval_wave2_20260624.sh
scripts/run_qwen35_official_agentic_lora_20260627.sh
```

이 두 파일은 기존 미관리 파일이라 건드리지 않았다.

## GLM-5.2 튜닝 판단 수정

이전 시도에서 한 `device_map` 단일 프로세스 layer sharding은 “8GPU를 한 장처럼 쓰는” proper training stack이 아니다. GLM-5.2급 MoE fine-tuning에는 다음 중 하나가 필요하다.

1. Expert Parallel + Pipeline/Tensor/FSDP 계열 training stack
2. Unsloth MoE bf16 LoRA kernel 경로
3. NeMo Automodel/Megatron 계열 GLM-aware MoE training
4. slime Megatron+SGLang post-training
5. Fireworks 같은 hosted GLM-5.2-FP8 LoRA

Unsloth 확인:

- Unsloth MoE 문서는 MoE training을 `torch._grouped_mm`/Triton kernel로 최적화하고 GLM 4.7 Flash, Qwen3 MoE, DeepSeek류를 언급한다.
- 중요한 문구: MoE 4-bit QLoRA는 BitsAndBytes가 지원하지 않아 권장하지 않고, 지금은 bf16 LoRA 또는 full fine-tuning을 쓰라고 한다.
- 따라서 GLM-5.2에 BitsAndBytes QLoRA를 적용한 것은 잘못된 방향이었다.

NVIDIA NeMo Automodel 확인:

- GLM-5.1 fine-tune 예시는 FSDP2 + Expert Parallel + Pipeline Parallel을 쓴다.
- 예시 설정은 EP=64, PP=4, 32 nodes x 8 H100 규모다.
- 이건 “8장 한 노드에서 단순 LoRA”가 아니라 GLM-aware distributed MoE training이다.

slime 확인:

- slime은 LLM post-training framework이고 Megatron training과 SGLang generation을 연결한다.
- GLM 계열 large-scale RL/post-training에 맞는 방향이지만, 바로 현재 repo에서 한 줄로 실행할 수 있는 LoRA runner는 아니다.

## 바로 할 일

1. `.data`에서 최소 1.4T 확보:
   - 1순위: BF16 GLM cache `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2`
   - FP8 cache는 eval/teacher용으로 보존.
2. GLM local tuning은 당장 재시도 금지:
   - `scripts/run_glm52_bf16_qlora_device_map_20260627.sh`는 guard로 막아둠.
3. GPU를 놀리지 않으려면:
   - Qwen3.5/Gemma/DiffusionGemma trainable base를 adapter-only로 돌린다.
   - Unsloth는 Qwen3.5 또는 Qwen3 MoE류부터 smoke한다. GLM-5.2는 Unsloth GLM-5.2 training example이 명확히 확인된 뒤에만 시도한다.
4. git:
   - 로컬 커밋은 백업 브랜치가 있으므로 안전하다.
   - GitHub credential을 설정해야 push 가능하다.

## 참고 링크

- Unsloth MoE training: https://unsloth.ai/docs/basics/faster-moe
- Unsloth Qwen3.5 fine-tune: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth GLM-5.2 local guide: https://unsloth.ai/docs/models/glm-5.2
- NVIDIA NeMo Automodel GLM-5.1 discussion: https://github.com/NVIDIA-NeMo/Automodel/discussions/1719
- slime: https://github.com/THUDM/slime
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
