# Phase 0.4 — 참고 코드베이스 학습 노트

본 연구에서 차용·참고할 6개 오픈소스 프로젝트 각각에 대해 **무엇을 어디서 어떻게 쓸지**를 매핑. "전부 읽기"가 목표가 아니라 "필요한 부분을 식별해 본 연구 모듈에 옮겨오는 것"이 목표.

각 항목 형식: 우선순위 / 이 연구에서의 역할 / 읽을 파일 / 차용/참고/무시 / 옮겨올 위치 / 위험.

---

## 1. Flamingo (lucidrains 비공식 PyTorch 구현) — 우선순위 1 ★

- 링크: https://github.com/lucidrains/flamingo-pytorch
- **이 연구에서의 역할**: Phase 2 어댑터의 청사진. cross-attention으로 외부 표현을 frozen LLM에 주입하는 핵심 메커니즘.

**먼저 읽을 파일**.
1. `flamingo_pytorch/flamingo_pytorch.py`
   - `PerceiverResampler` — 가변 길이 활성화를 고정 N개 토큰으로 압축. 본 연구의 Go-Net 활성화 (채널, 9, 9) → N개 활성화 토큰 변환에 그대로 차용.
   - `MaskedCrossAttention` — LLM 시퀀스에 cross-attention 주입. **핵심**.
   - `GatedCrossAttentionBlock` — tanh-gate 로 학습 초기 영향을 0에서 시작. Phase 3.5 위험요소 "어댑터가 LLM 망가뜨림" 의 표준 대응책.
2. `flamingo_pytorch/flamingo_palm.py`
   - LLM (PaLM) 에 cross-attention 블록을 끼워 넣는 패턴. 본 연구는 TinyLlama 에 대해 동일 작업을 해야 함.

**차용 (거의 그대로)**.
- `PerceiverResampler` 클래스
- `MaskedCrossAttention` 클래스
- gating 로직 (`gate_attn`, `gate_ff`)

**참고 (구조 모방, 본 연구 맞춤 재작성)**.
- LLM 본체에 cross-attention 블록을 어떻게 끼워 넣는지의 패턴.
- HuggingFace `LlamaModel` 의 `forward` 를 wrap 하는 방식으로 적용 — Flamingo 처럼 모델 클래스를 새로 만들지 않음.

**무시**.
- 비전 인코더 (CLIP) 부분 — 본 연구에서는 Go-Net 이 그 자리.
- 학습 루프 — 본 연구는 자체 합성 데이터.

**옮겨올 위치**.
- `src/split_brain_go/adapter/xattn.py` — `MaskedCrossAttention`, `GatedCrossAttentionBlock`
- `src/split_brain_go/adapter/projection.py` — `PerceiverResampler` 의 변형 (Go-Net 활성화에 맞춤)
- `src/split_brain_go/llm/instrumented.py` — Llama 모델에 어댑터 블록 주입

**위험**.
- lucidrains 구현은 "참고"이지 production-grade가 아님. 일부 디테일은 원논문과 다름. **원논문 (Alayrac et al., 2022) 도 함께 정독**.
- 코드 스타일이 einops + 함수형 위주 — 본 연구 코드 스타일과 통일 필요.

---

## 2. OpenSpiel — 우선순위 1 ★

- 링크: https://github.com/deepmind/open_spiel
- **이 연구에서의 역할**: 9×9 바둑 환경. 자체 구현보다 빠르고 검증된 룰셋.

**먼저 읽을 파일**.
1. `open_spiel/python/games/__init__.py` — Python에서 어떻게 게임을 로드하는지
2. `open_spiel/python/algorithms/mcts.py` — AlphaZero 스타일 MCTS의 reference 구현. Phase 1.3에서 참고.
3. `open_spiel/python/examples/alpha_zero.py` — self-play + neural net 학습 파이프라인 예제.

**차용**.
- `pyspiel.load_game("go(board_size=9)")` API 자체.
- `state.legal_actions()`, `state.apply_action(a)`, `state.is_terminal()`, `state.returns()` 인터페이스.
- 보드 상태를 텐서로 변환하는 helper (`state.observation_tensor`).

**참고**.
- MCTS 알고리즘의 골격 — UCB1, expansion, backprop. 본 연구의 `gonet/mcts.py` 는 이를 PyTorch로 다시 작성.

**무시**.
- 70여 개의 다른 게임 코드.
- C++ 빌드 자체 — 우리는 wheel만 사용.

**옮겨올 위치**.
- `src/split_brain_go/env/go_env.py` — OpenSpiel의 `pyspiel.State` 를 wrap하는 `GoEnv` 클래스.
- `src/split_brain_go/env/encoding.py` — observation tensor 를 본 연구 채널 정의로 재구성.

**위험**.
- Windows 네이티브 빌드 실패 가능 — WSL2 사용으로 우회 (Phase 0.1 참고).
- `observation_tensor` 의 채널 정의가 AlphaGo Zero 논문과 약간 다름. Phase 1.2에서 명시적 매핑 표 작성.

---

## 3. TransformerLens — 우선순위 2

- 링크: https://github.com/TransformerLensOrg/TransformerLens
- **이 연구에서의 역할**: Phase 4 의 activation patching 메트릭 측정에 직접 사용.

**먼저 읽을 파일**.
1. `transformer_lens/HookedTransformer.py` — 모델을 hook 가능한 형태로 wrap하는 인터페이스.
2. `transformer_lens/hook_points.py` — `HookPoint` 메커니즘.
3. 튜토리얼 노트북 `Main_Demo.ipynb` 의 "activation patching" 섹션.

**차용**.
- `HookedTransformer` 의 hook 추가 방식 — `run_with_cache`, `run_with_hooks`.
- `ActivationCache` 객체.

**참고**.
- 단, 본 연구의 activation patching 대상은 **Go-Net** 의 활성화. TransformerLens는 트랜스포머용. 따라서:
  - LLM 쪽에서 패치 효과를 관찰할 때 TransformerLens 그대로 사용.
  - Go-Net 쪽 패치는 PyTorch 표준 forward hook으로 직접 구현.

**무시**.
- TransformerLens가 직접 지원하는 GPT-2/Pythia 등 사전훈련 모델 분석 기능 중 본 연구와 무관한 것.

**옮겨올 위치**.
- `src/split_brain_go/eval/activation_patch.py` 의 LLM-쪽 분석 부분.
- `src/split_brain_go/gonet/hooks.py` 의 forward hook은 PyTorch 표준으로 작성.

**위험**.
- TinyLlama가 TransformerLens에서 native 지원되는지 확인 필요. 미지원이면 fallback으로 transformers의 `output_hidden_states=True` 사용.

---

## 4. Leela Zero — 우선순위 3

- 링크: https://github.com/leela-zero/leela-zero
- **이 연구에서의 역할**: AlphaGo Zero 식 self-play + 신경망 학습 루프의 깔끔한 reference. C++이라 직접 import 불가, **읽고 학습**.

**먼저 읽을 파일**.
1. `src/Network.cpp` — 잔차 블록 구조, policy/value head 설계.
2. `src/Training.cpp` — replay buffer, mini-batch 샘플링, loss 정의.
3. `training/tf/parse.py` — 학습 데이터 포맷.

**차용**.
- 잔차 블록 깊이/채널 선정 (단, 본 연구는 9×9이므로 더 작게: 잔차 블록 3~6, 채널 64~128).
- policy/value loss 비율 (보통 1:1).
- Dirichlet 노이즈를 root에만 더하는 표준 트릭.

**참고**.
- replay buffer 의 크기와 샘플링 정책.

**무시**.
- 분산 학습/투표 부분.
- 19×19 특화 코드.

**옮겨올 위치**.
- `src/split_brain_go/gonet/network.py` — 신경망 구조 영감.
- `src/split_brain_go/training/selfplay.py`, `replay_buffer.py` — 학습 루프 구조.

**위험**.
- 코드가 C++ — Python으로 직접 옮길 때 미묘한 디테일 누락 위험. **PyTorch reimplementation을 따로 검증**.

---

## 5. KataGo — 우선순위 4 (심화 시)

- 링크: https://github.com/lightvector/KataGo
- **이 연구에서의 역할**: 현대 바둑 AI의 모범. 본 연구 단계에선 너무 거대.

**언제 읽나**. Phase 1.4 평가에서 Go-Net이 약하면, KataGo의 추가 trick들을 일부 도입.

**차용 후보**.
- `playouts` cap 으로 학습 효율 올리는 방식.
- score-based loss term — 단순 승패가 아닌 점수차로 학습 (gradient 풍부).

**무시**.
- C++ 본체, MCTS 최적화, opening book 등.

---

## 6. LLaVA — 우선순위 4 (심화 시)

- 링크: https://github.com/haotian-liu/LLaVA
- **이 연구에서의 역할**: 어댑터 학습 전략의 reference (1단계 어댑터만, 2단계 LLM 일부 함께).

**먼저 읽을 파일**.
1. `llava/train/train.py` — 학습 단계 정의.
2. `llava/model/llava_arch.py` — vision feature → LLM hidden 의 projection.

**차용**.
- 2단계 학습 스케줄 — pretrain (어댑터만) → finetune (어댑터 + LLM 일부).
- LR 스케줄 (warmup + cosine).

**참고**.
- LoRA 적용 방식 (`peft` 통합).

**무시**.
- 비전 인코더 (CLIP).
- 채팅 데이터 포맷.

**옮겨올 위치**.
- `src/split_brain_go/training/joint_train.py` 의 2단계 스케줄 패턴.

---

## 7. 추가 참고 자료 (논문 우선)

코드 외에 반드시 정독:

1. **Alayrac et al., 2022 — Flamingo**. cross-attention 설계와 학습 트릭의 원전.
2. **Silver et al., 2017 — AlphaGo Zero**. self-play + MCTS + 신경망의 정석.
3. **Anthropic, "Language Models Don't Always Say What They Think" (2023)**. 본 연구의 충실성 메트릭이 왜 필요한지의 동기.
4. **Anthropic interpretability 시리즈**. activation patching 의 이론과 실제.
5. **Gazzaniga, "The Mind's Past"**. 본 연구의 인지과학적 동기. 1쪽 요약을 README에 박제.

---

## 8. 우선 순위 행동 계획 (Phase 0.4 한정)

각 리포에 대해 다음 산출물:

| 리포 | 산출물 | 시간 |
|------|--------|------|
| Flamingo | `notebooks/flamingo_walkthrough.ipynb` — 핵심 클래스에 주석 달기 | 1일 |
| OpenSpiel | `notebooks/01_openspiel_play.ipynb` — Go 9×9 한 판, observation tensor 분석 | 0.5일 |
| TransformerLens | `notebooks/03_activation_inspection.ipynb` — TinyLlama 의 hook 동작 확인 | 0.5일 |
| Leela Zero | `docs/leela_zero_notes.md` — 학습 루프 구조도 1쪽 | 0.5일 |
| KataGo, LLaVA | Phase 0 에서 skim만, 본격 정독은 필요할 때 | 0.5일 |

총 약 1주. 이 주의 산출물은 모두 `D:\brain\split_brain_go\notebooks/` 와 `docs/` 에 commit 됨.

---

## 9. 라이선스 점검

본 연구가 학회 발표를 목표로 한다면 차용 코드의 라이선스 호환성을 점검.

| 리포 | 라이선스 | 본 연구 호환성 |
|------|----------|----------------|
| Flamingo (lucidrains) | MIT | ◎ 차용 가능, 출처 표기 필수 |
| OpenSpiel | Apache 2.0 | ◎ 의존성으로 사용 |
| TransformerLens | MIT | ◎ |
| Leela Zero | GPLv3 | △ **코드 직접 차용 시 본 연구도 GPLv3로 묶임**. 따라서 "읽고 영감만" 으로 한정. |
| KataGo | MIT (대부분) | ◎ |
| LLaVA | Apache 2.0 | ◎ |

본 연구 코드는 **Apache 2.0 또는 MIT** 로 공개 예정. Leela Zero 코드를 직접 복사하지 말 것.
