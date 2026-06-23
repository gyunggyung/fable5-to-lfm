# fable5-to-lfm 밤샘 실험 계획 (2026-06-23 → 2026-06-24)

> 12시간+ 밤샘 실험 로드맵. 진행 상황은 이 파일에 실시간 업데이트.

## 현재 진행 상황

### ✅ 완료 (Phase 0: 기본 라인업)
- [x] Phase-1 (Fabliq-8B-Agent): ToolBench → Fable-5 SFT
- [x] Phase-1B (Fabliq-8B-Agent-FromBase): raw LFM2.5-8B-A1B → Fable-5 SFT
- [x] Phase-2 (Fabliq-8B-Agent-Reasoning): Phase-1 + WithinUs+Helio reasoning
- [x] Phase-2B (Fabliq-8B-Agent-FromBase-Reasoning): Phase-1B + reasoning
- [x] Phase-1 GGUF (Q4_K_M, Q5_K_M, Q6_K, Q8_0) → HF 업로드
- [x] Reasoning/FromBase GGUF 변환 (HF 업로드 대기)

### 🔄 진행 중
- [ ] **Mega-Combined**: raw LFM2.5-8B-A1B → 모든 데이터 (4,328 rows) × 3 epoch
- [ ] Reasoning/FromBase GGUF Q4_K_M 양자화 진행 중

## 12시간 로드맵

### Hour 1-2 (22:55-00:30): 기본 eval + Mega
- [ ] Mega 학습 완료 (10 min)
- [ ] 7개 모델 전체 eval (base, ToolBench, Phase-1, Phase-1B, Phase-2, Phase-2B, Mega)
  - tb2_lite terminal (117 rows)
  - MMLU 5-shot
  - HumanEval pass@1
- [ ] Eval 결과 비교 테이블 생성

### Hour 2-3 (00:30-01:30): Mega → reasoning expansion
- [ ] Phase-2M (Mega + WithinUs+Helio reasoning) 학습
- [ ] Phase-2M eval
- [ ] Phase-2M GGUF 변환 + HF 업로드

### Hour 3-5 (01:30-03:30): Epoch ablation
- [ ] Mega 1ep 학습 + eval
- [ ] Mega 5ep 학습 + eval
- [ ] Mega 10ep 학습 + eval
- [ ] 최적 epoch 도출

### Hour 5-7 (03:30-05:30): LR ablation
- [ ] Mega lr5e-7 학습 + eval
- [ ] Mega lr2e-6 학습 + eval
- [ ] 최적 LR 도출

### Hour 7-9 (05:30-07:30): 작은 모델 라인업
- [ ] LFM2.5-1.2B-Instruct → Fable-5 SFT (Fabliq-1.2B-Agent)
- [ ] Fabliq-1.2B-Agent eval
- [ ] Fabliq-1.2B-Agent GGUF 변환

### Hour 9-11 (07:30-09:30): 문서 정리 + 모델 카드 업데이트
- [ ] 모든 모델 카드에 eval 결과 반영
- [ ] 한글/영문 README에 전 라인업 정리
- [ ] 비교 테이블 생성 (GLM-5.2 vs Fabliq variants)

### Hour 11-12 (09:30-10:30): 최종 마무리
- [ ] 최종 commit + GitHub push
- [ ] HF에 모든 GGUF 업로드
- [ ] 리포트 생성 (EVAL_RESULTS_20260623.md)

## 의사결정 포인트

- 3 epoch vs 5 epoch vs 10 epoch: 성능 vs overfitting tradeoff
- LR 5e-7 vs 1e-6 vs 2e-6: 수렴 속도 vs 안정성
- ToolBench foundation 효과: Phase-1 vs Phase-1B 비교로 측정
- Reasoning expansion 효과: Phase-1 vs Phase-2 비교로 측정

## GPU 활용 철학

- 8 H200을 한 번에 하나의 작업에 올인 (FSDP 효율 최대)
- CPU 작업 (GGUF 변환, 전처리)은 GPU 작업과 병렬
- HF 업로드는 네트워크 작업이라 언제든 병렬 가능

## 예상 결과 해석

- **ToolBench 효과 유의미**: Phase-1 > Phase-1B (ToolBench가 터미널 툴 사용에 도움)
- **Reasoning 효과 유의미**: Phase-2 > Phase-1 (WithinUs+Helio가 MMLU/GPQA 향상)
- **Mega 효과**: 한 번에 학습한 게 단계별보다 나은지 확인
- **LR/epoch 최적값**: 데이터 크기 4,328 rows에 맞는 최적값 도출
