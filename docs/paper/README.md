# SPLIT-9

*9×9 Go-Net에 LLM을 후처리로 붙였을 때 정말 "통역"이 되는지 검증한 프로젝트.*

> **TL;DR.** 동결된 9×9 Go-Net에 동결된 TinyLlama-1.1B를 Flamingo
> 스타일 어댑터로 연결하고, Go-Net의 수(手) 선택을 자연어로 설명하도록
> 학습시켰다. 검증 perplexity가 **16.1 → 1.65**로 극적으로 떨어졌다.
> 그러나 세 종류의 독립적 probe로 따져보니, 그 이득의 약 **95%는
> 학습된 도메인 prior**고, **보드별로 다른 신호의 기여는 5%에 불과**했다.
> 정성 샘플 10개 중 **0개**만 실제 착수 좌표를 맞혔고, 출력은
> 데이터셋에 자주 등장하는 좌표 (`(2,1)`, `(8,8)`, `pass`)로 mode-collapse
> 했다. 어댑터는 설명문의 *문법*을 배운 것이지 *내용*을 배운 게 아니다.

---

## 왜 이걸 만들었나

Roger Sperry와 Michael Gazzaniga의 분리뇌(split-brain) 실험에서 흥미로운
현상이 관찰됐다. 좌·우반구를 잇는 corpus callosum(뇌량)을 절단한 환자에게,
오른쪽 시야를 차단한 채 왼쪽 시야로만 어떤 지시를 보여주면 — 즉 우반구에만
정보가 들어가면 — 환자는 시키는 대로 행동한다. 그런데 "왜 그렇게 했냐"고
물으면, 언어 능력을 가진 좌반구는 자기가 보지 못한 결정을 *그럴듯하게*
설명해 낸다. 거짓말이 아니라 진짜로 "내가 이래서 이렇게 했다"고 믿는다.
Gazzaniga는 이 모듈을 좌반구 통역사(left-hemisphere interpreter)라고 불렀다.
유창하고, 일관성 있고, 자주 틀린다.

지금의 AI 시스템은 이 구조를 **의도적으로** 반복해서 만들고 있다.
시각 인코더, 로봇 정책, 과학 모델 같은 비언어 네트워크를 만들어 놓고,
그 옆에 LLM을 붙여서 "지금 무슨 일이 일어나고 있는지" 자연어로 설명하게
한다. RLHF chain-of-thought, 비전-언어 어시스턴트, "설명 가능한" RL 에이전트
— 전부 같은 아키텍처의 변종이다. 이때 LLM이 정말로 상위 신호를 *번역*하고
있는지, 아니면 단지 통계적으로 그럴듯한 텍스트를 뽑아내고 있는지는
대체로 검증되지 않은 채 남아있다.

SPLIT-9는 그 질문을 가장 작고 재현 가능한 형태로 던져보기 위한
testbed다.

- 9×9 **Go-Net** (AlphaGo Zero 스타일, ~5M 파라미터, 완전한 화이트박스) — "우반구"
- 동결된 **TinyLlama-1.1B** — "좌반구"
- 학습 가능한 **Flamingo 스타일 어댑터** — "corpus callosum"
- 좌반구가 진짜로 통역하고 있는지 잡아내기 위한 일련의 faithfulness probe

소비자용 8 GB GPU 한 대로 처음부터 끝까지 하루 안에 학습된다. 모든
구성 요소가 이 저장소 안에 있다.

---

## 무엇을 만들었나

```
SPLIT-9
├── Go-Net                  (Phase 1에서 학습 후 동결)
│       │ 레이어 {3, 4, 5}의 활성화
│       ▼
├── AsymmetricPerceiverResampler        ← 학습 가능
│       │ 36개 latent token (레이어별 8 / 12 / 16)
│       ▼
├── GatedCrossAttention 블록 — TinyLlama 레이어 {10, 16, 18, 20}에 주입
│       │
│       ▼
├── TinyLlama-1.1B          (동결)
│       │ 32k vocab logit
│       ▼
└── 합성된 자연어 reasoning trace
```

**데이터.** 30번의 self-play 게임 × 평균 ~75수 = 2,293개 예제.
각 예제는 `(보드, 착수, MCTS principal variation, 합성된 설명문)`. 6개
설명 템플릿 × 단어 풀 변형(verb pool, noun pool).

**학습.** bf16, AdamW, lr 1e-4, batch 4, 3 epoch. RTX 3070 Ti에서 약 30분.
어댑터(~404M)만 학습 가능. Go-Net과 LLM 본체는 동결.

---

## 무엇을 측정했나

비용 낮은 것부터 무거운 것 순으로 4개의 probe.

### 1. 베이스라인

| | val loss | val ppl |
|---|---|---|
| Output-only (어댑터 우회)              | 2.78 | 16.14 |
| Random-init 어댑터 (체크포인트 미로딩)  | 2.78 | 16.14 |
| **학습된 어댑터**                      | **0.50** | **1.65** |

학습된 어댑터가 Output-only 대비 loss를 **−81.9%** 줄인다.
Random vs Output-only가 똑같은 이유 — Flamingo 게이트가 0으로 초기화되기
때문에 학습 전에는 어댑터가 사실상 no-op이다. (방법론적 짚고 넘어가기:
Random-Adapter를 의미 있는 베이스라인으로 쓰려면 게이트를 비-0으로
초기화해야 한다.)

### 2. Information Ablation Score (IAS)

Go-Net 활성화 텐서가 어댑터에 도달하기 전에, 채널의 비율 *r*만큼 무작위로
0으로 만든 다음 loss가 얼마나 오르는지를 본다.

| r | loss | base 대비 Δloss |
|---|---|---|
| 0.00 | 0.503 | — |
| 0.25 | 0.507 | +0.4 % |
| 0.50 | 0.521 | +3.5 % |
| 0.75 | 0.556 | +10.6 % |
| 1.00 | 0.607 | +20.6 % |

곡선이 *볼록(convex)*하다 — 절반 채널을 날려도 거의 차이가 없다. 이건
신호가 채널들 사이에 중복되게 인코딩됐다는 뜻이고, 그래서 IAS 단독으로는
faithfulness 판정에 인색한 probe가 된다.

### 3. Activation Patching Consistency (APC)

같은 prompt + label에 대해 두 번 forward를 돌린다. **matched** (정상 활성화)
vs **mismatched** (배치 내에서 derangement로 섞은 활성화 — 어댑터가
설명문이 가리키는 보드와 *다른* 게임의 활성화를 보게 됨).

| | loss |
|---|---|
| matched          | 0.503 |
| mismatched       | 0.594 |
| **APC**          | **+0.181** |

3 seed, σ ≈ 0.003 — 매우 안정적이다. mismatched 페널티는 IAS@0.5보다
**5배 크다**. 보드 전체 swap이 채널 중복 문제를 우회하기 때문.
APC = 0.18은 보드별 신호가 인과적으로 흐르고 있음을 확정하지만, 우리
휴리스틱 임계 기준으로는 "FAITHFUL"이 아니라 "PARTIAL"에 해당한다.

### 4. 정성 검사 (펀치라인)

10개 random val 보드에 대해 prompt를 그리디 디코딩한 결과.

```
[00] action=(4,1)
GROUND TRUTH : ...White selected (4, 1)... value moved from +0.57 to +0.98...
TRAINED      : [endgame/111] W(pass) val +1.00; W(8, 8) val +1.00; ...
OUTPUT-ONLY  : 1. The author's use of the word "suddenly"...

[02] action=(0,1)
GROUND TRUTH : ...Chose W(0, 1): highest expected value.
TRAINED      : [endgame/111] W (8, 0) (p=0.02, v=+0.99, Δv=-0.01) selected...
OUTPUT-ONLY  : 1. The author's use of the word "suddenly"...

[05] action=(5,1)
GROUND TRUTH : ...Black selected (5, 1)... value moved from -0.30 to -0.02...
TRAINED      : [middlegame/21] B (2, 1) (p=0.01, v=-0.99, Δv=+0.11) selected...
OUTPUT-ONLY  : 1. The author's use of the word "suddenly"...
```

세 가지가 눈에 띈다.

1. **OUTPUT-ONLY가 퇴화 상태다.** 어떤 입력을 줘도 동일한 일반 문학
   분석문을 뱉는다. 즉 TinyLlama 자체는 Go에 대해 아무것도 모른다.
   이건 R8(템플릿 암기) 가설을 완전히 기각하고, 어댑터가 도메인 작업의
   **100%**를 떠맡고 있음을 확인시켜 준다.

2. **TRAINED는 합성 템플릿의 문법을 정확히 재현한다.** `[middlegame/21]
   B (2, 1) (p=0.01, v=-0.99, Δv=+0.11) selected` — phase 태그, 색,
   좌표 괄호, `p=`, `v=`, `Δv=`, `selected`까지 전부 완벽.

3. **그러나 TRAINED는 실제 착수에 대해 거의 항상 틀린다.** 10 샘플 중
   **0개**가 actor의 정답 좌표와 일치. 출력 좌표는 `(2,1)` (4회), `(8,8)`
   (3회), `pass`, 그리고 몇 개로 집중된다. 즉 어댑터가 데이터셋에서
   자주 보는 수에 mode-collapse한 상태.

---

## Loss 분해 (헤드라인)

세 probe를 raw loss 단위로 합쳐보면.

```
Output-only (어댑터 없음)        loss 2.78    ┐
                                              │ 총 감소량의 ~95%
Trained, 활성화 전부 0 (mask=1)   loss 0.61    ┘   = 학습된 도메인 prior
                                              ┐
Trained, mismatched (APC)        loss 0.59    │ 총 감소량의 ~5%
Trained, matched (정상)          loss 0.50    ┘   = 보드별 신호
```

어댑터의 기여는 **보드 무관 도메인 prior** (절대값으로 78% loss 감소,
전체 개선의 95%) + **보드별 인과 신호** (loss 0.09, 5%)로 분해된다.
후자는 실재하고 통계적으로 견고하지만, 정확한 좌표를 짚어내기엔 부족해서
"어떤 mode로 collapse할지"만 살짝 영향을 준다.

---

## 이게 무슨 의미인가

**Cross-entropy 학습은 합리적으로 prior에 자원을 투자한다.** 30개 토큰짜리
설명문에는 보드별 정보를 담는 토큰(좌표) 약 3개와 구조 토큰(phase 태그,
괄호, `selected`, 구두점) 약 27개가 들어 있다. 옵티마이저는 27개를
완벽하게 맞추고 3개는 mode-collapse하는 쪽으로 loss를 최소화한다.
APC는 그 작은 잔류 보드 민감도를 잡아내지만, cross-entropy 자체는 이
간격을 전혀 드러내지 않는다.

**이건 튜닝 문제가 아니라 구조적 천장이다.** 어댑터를 키우거나, 더
오래 학습하거나, 같은 템플릿 분포 안에서 데이터를 늘리는 건 prior를 더
잘 맞히는 효과만 있고 보드별 내용 정확도는 거의 못 끌어올린다. faithfulness
를 진짜로 끌어올리려면 **데이터 분포 자체를 바꿔야** 한다 — 보드 의존
내용이 더 많이 들어가도록 하거나, 정답 좌표가 출력에 등장하는 걸 명시적으로
강제하는 auxiliary loss를 추가하거나.

**그리고 이건 일반화된다.** 같은 토큰-엔트로피 가중치 논리가 대부분의
post-hoc 어댑터 setup에 적용된다. 합성 캡션 위의 visual question answering,
RL 에이전트의 자기 행동 narration, next-token loss로만 평가되는 chain-of-thought
rationale — 전부 같은 함정이 있다. APC 같은 ablation probe + 정성 검사는
**필요조건**이고, train/val perplexity만으로는 이 실패를 못 잡는다.

---

## 향후 연구 — co-developed twin networks

post-hoc 어댑터는 데이터 분포가 *번역하기 쉽게* 만들어 둔 신호만 번역할
수 있다. 특정 내용을 강제할 수 없다. 생물학적 corpus callosum은 이 문제가
없는데 — 두 반구가 같은 발달 시기에 함께 자라면서 서로 호환되는 표현을
공동 학습하기 때문이다.

자연스러운 다음 실험: 두 네트워크를 **동시에** 학습시키되 학습 가능한
bottleneck을 사이에 두고, 상위 특징(선택된 수, 상대의 직전 수, 잡힌 돌)이
언어 헤드의 출력에 구체적인 토큰으로 등장하도록 강제하는 auxiliary loss를
추가한다. 이건 생물학적 공동 발달을 흉내 내는 셋업이고, 위에서 정리한
구조적 논리에 따르면 mode-collapse 천장을 깰 수 있는 거의 유일한
방향이다.

---

## 재현 방법

```bash
# 0. 환경
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Go-Net 학습 (3070 Ti default config로 ~3시간)
python scripts/train_gonet.py --config configs/gonet/default.yaml

# 2. Phase 3 데이터셋 생성 (MCTS PV self-play)
python scripts/generate_phase3_data.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --n-games 30 --n-simulations 100 \
    --output runs/phase3_data_small.pkl

# 3. 어댑터 학습 (~30분)
python scripts/train_adapter.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --dataset runs/phase3_data_small.pkl \
    --epochs 3 --batch-size 4

# 4. Faithfulness probe
python scripts/baselines.py             --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_faithfulness.py  --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_apc.py           --gonet-ckpt ... --adapter-ckpt ... --dataset ...

# 5. 정성 샘플
python scripts/sample_generations.py \\
    --gonet-ckpt ... --adapter-ckpt ... --dataset ... \\
    --n-samples 10 --out runs/samples.txt
```

테스트: `pytest tests/ -v` (~200개, ~1분).

---

## 스택

Python 3.12 · PyTorch 2.x · HuggingFace Transformers · OpenSpiel (Go env) ·
NumPy · 8 GB 소비자용 GPU 1대 (RTX 3070 Ti / WSL2 Ubuntu 22에서 검증).
TinyLlama-1.1B-Chat 체크포인트는 HuggingFace Hub에서 받아옴.

---

## 참고 문헌

* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Jaegle et al., *Perceiver IO*, ICLR 2022.
* Silver et al., *Mastering the game of Go without human knowledge*, Nature 2017.
* Wu, *Accelerating Self-Play Learning in Go (KataGo)*, 2019.
* Gazzaniga, *The Bisected Brain*, Appleton-Century-Crofts, 1970.
* Gazzaniga, *The Consciousness Instinct*, Farrar Straus Giroux, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023.

---

## 상태

1인 프로젝트. baseline으로서 완결된 상태.
이 negative result 자체가 결과물이다.
PR · issue · 날서로운 비판 환영.

---

*일하다가 한 망상을 실현시켜준 Claude 에게 감사를.* 🦫
