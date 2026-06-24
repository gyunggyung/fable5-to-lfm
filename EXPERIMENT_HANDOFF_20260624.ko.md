# 실험 중단 및 인계 정리 (2026-06-24)

작성 시각: 2026-06-24 22:59 KST

## 현재 상태

GPU 사용 프로세스는 모두 중단했다. `nvidia-smi --query-compute-apps` 기준 DiffusionGemma 학습 프로세스는 사라졌고, 각 GPU 메모리는 1MiB 수준으로 비었다.

남아 있는 `Liquid-CLI/scripts/sync_echo_adapter_checkpoints_to_hf_model.py` 프로세스는 Hugging Face 동기화 loop이며 GPU compute app이 아니다.

정리용 stop 스크립트를 추가했다.

```bash
# 먼저 dry-run으로 매칭되는 fable_distillation job 확인
bash fable_distillation/scripts/stop_fable_gpu_jobs_20260624.sh

# 실제 중단
RUN_NOW=1 bash fable_distillation/scripts/stop_fable_gpu_jobs_20260624.sh
```

주의: 이 스크립트는 `fable_distillation` 경로가 command line에 들어간 GPU/eval job만 대상으로 한다. Liquid 동기화 loop처럼 GPU를 쓰지 않는 외부 프로세스는 건드리지 않는다.

## 커밋 상태

최근 관련 커밋:

- `03830aa` `Skip PEFT optimizer checkpoint for DiffusionGemma`
- `aadc8dc` `Plan DiffusionGemma strength-task LoRA`
- `7971963` `Add Qwen LoRA merge and eval watcher`
- `4e76bb2` `Ignore local tokenized cache`
- `9522eb6` `Fix Qwen text-only LoRA DDP training`

## 완료된 것

### 벤치마크/평가

- DiffusionGemma 26B-A4B-it Docker-free Transformers dLLM 평가 성공.
  - corrected full TB2-lite: score `25.12`, Cmd F1 `0.2980`, First Cmd `14.2%`, Valid JSON `55.1%`
  - probe speed: `97.88 tok/s`
  - 결론: 실행은 성공, TB2-lite next-action 성능은 실패.
- Qwen3.5-9B base vLLM sharded TB2-lite 평가 완료.
  - score `36.75`, Cmd F1 `0.4358`, First Cmd `20.8%`, Valid JSON `78.9%`
- GLM-5.2 chaser SFT 평가 완료.
  - final: score `51.13`
  - checkpoint-1400: score `50.56`
  - 로컬 best Fabliq `51.59`는 넘지 못함.

### 학습/데이터

- DiffusionGemma strength-task SFT mix 생성 완료.
  - 파일: `datasets/diffusiongemma_strength_mix_20260624.jsonl` (gitignored)
  - rows: `11,352`
  - Fable source rows: `4,970`
  - 주요 구성: Fable terminal/tool traces + structured JSON/tool-call repair + Hermes function/code traces
- Qwen3.5-9B LoRA SFT300 완료.
  - train loss `0.6287`
  - LoRA merge 완료
  - vLLM load는 VLM/text wrapper weight-key mismatch로 실패, 점수 없음.
- DiffusionGemma NeMo PEFT optimizer checkpoint 문제 우회 코드 추가.
  - wrapper: `scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py`
  - runner: `scripts/run_diffusiongemma_strength_lora_20260624.sh`
  - config: `configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml`
- GPU job 정리 스크립트 추가.
  - stop script: `scripts/stop_fable_gpu_jobs_20260624.sh`

## DiffusionGemma 최신 학습 상태

### retry1

- run id: `20260624_diffusiongemma_strength_lora_fable_structured_repair_retry1`
- step `199`까지 정상 학습.
- step `200` checkpoint에서 실패.
- 원인: NeMo PEFT 경로가 optimizer nested dict를 `safetensors`로 저장하려다 `Key optim is invalid` 오류.
- adapter checkpoint는 남아 있음:
  - `/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-20260624/epoch_0_step_199/model/adapter_model.safetensors`

### no-optimizer retry

- 실제 로그: `logs/debug_nooptim_foreground_223859/train.log`
- checkpoint dir: `/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/`
- KST `22:53:11`에 step `0` 시작.
- 사용자가 중단 요청한 뒤 KST `22:59:17`에 graceful stop signal 수신.
- step `112`까지 진행 후 checkpoint 저장.
- wrapper가 정상 동작해서 optimizer 저장은 건너뛰고 adapter 저장은 성공.

저장된 adapter:

```text
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/epoch_0_step_112/model/adapter_model.safetensors
```

중요: 이 checkpoint는 adapter 평가/분석에는 쓸 수 있지만 optimizer state가 없으므로 완전한 optimizer resume 용도는 아니다.

## 잘 되는 것

- Docker 없이 DiffusionGemma 실행, 평가, NeMo LoRA 학습 초기화까지 가능하다.
- DiffusionGemma LoRA 학습 loss는 빠르게 내려간다.
  - no-optimizer retry 기준 초기 `7.1744`에서 step 112 부근 `1.4690`까지 하락.
- PEFT optimizer checkpoint bug는 우회 가능하다.
  - step 112 저장에서 `Skipping PEFT optimizer checkpoint... Adapter model checkpoints are still saved.` 로그 확인.
- Fable 데이터와 structured repair 데이터를 섞은 학습 데이터 생성 경로는 안정적이다.
- vLLM sharded 평가 인프라는 Qwen base, GLM chaser 평가에서 이미 동작 확인됐다.

## 안 된 것 / 리스크

- DiffusionGemma LoRA는 아직 full 1200 step 완료 전이다.
- no-optimizer checkpoint는 optimizer state가 없으므로 그대로 이어 학습하려면 별도 adapter-load 시작 경로가 필요하다.
- `SIGTERM`으로 graceful stop 시 rank0가 adapter checkpoint를 저장한 뒤 torch elastic launcher가 `SignalException` traceback을 남긴다. 실제 adapter 저장은 성공했지만 로그가 실패처럼 보인다.
- DiffusionGemma VRAM 사용량은 H200 기준 낮다.
  - 학습 중 로그 기준 대략 `21-50GiB/GPU` 범위.
  - 이번 run은 사용자가 “지금은 그대로”라고 해서 batch/seq 조정 없이 유지했다.
- Qwen3.5 merged checkpoint는 vLLM load 실패 상태다.
  - 원인 후보: `Qwen3_5ForConditionalGeneration` / VLM wrapper 계열 config와 weight naming mismatch.
- GLM-5.2 chaser는 로컬 best `51.59`를 아직 넘지 못했다.

## 바로 해야 할 것

1. GPU를 다시 쓰기 전 `bash fable_distillation/scripts/stop_fable_gpu_jobs_20260624.sh` dry-run으로 남은 fable job이 없는지 확인한다.
2. DiffusionGemma를 계속할지 먼저 결정한다.
   - 선택 A: step 112 adapter를 평가만 해본다.
   - 선택 B: base부터 1200 step 재시작한다.
   - 선택 C: step 112 adapter를 초기 adapter로 로드하는 재시작 코드를 만든 뒤 이어간다.
3. DiffusionGemma 다음 run은 VRAM 활용을 올린 config를 별도로 만든다.
   - 후보: `global_batch_size` 증가, micro-batch/grad accumulation 조정, `seq_length` 확대, packing 개선.
   - 먼저 짧은 20-50 step smoke로 OOM과 tps를 비교한다.
4. step 200 checkpoint 통과를 다시 확인한다.
   - no-optimizer wrapper는 step 112 signal checkpoint에서 동작 확인됐지만, 정규 `ckpt_every_steps: 200` 지점은 아직 통과 전이다.
5. Qwen3.5 merged checkpoint를 text-only export로 다시 만들고 vLLM load를 재검증한다.
6. 성공/실패 판단은 공식 공개 benchmark 축으로 연결한다.
   - DiffusionGemma 공식 타깃: `Tau2`, `LiveCodeBench v6`
   - GLM-5.2 추격 타깃: `Tool-Decathlon`, `MCP-Atlas`, `Terminal-Bench 2.1`

## 다음 권장 순서

1. DiffusionGemma step 112 adapter를 작은 structured repair probe로 빠르게 평가한다.
2. 평가가 의미 있으면 1200 step 재시작 전에 VRAM 사용량을 올린 smoke config를 만든다.
3. VRAM/tps가 좋아지는 config를 고정한 뒤 1200 step 이상 run을 다시 건다.
4. 동시에 Qwen3.5 vLLM export mismatch를 고쳐 TB2-lite 점수를 회수한다.
5. 이후 공식 benchmark에 붙일 수 있는 형태로 결과표와 모델카드 초안을 정리한다.
