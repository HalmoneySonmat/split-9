# Phase 0.3 — 평가 프로토콜 (구체 측정 절차)

본 문서는 plan에 나열된 메트릭들을 **재현 가능한 측정 절차**로 옮긴 것이다. 각 메트릭은 (정의 → 입력 → 절차 → 출력 → 합격 기준 → 함정) 구조로 기술한다. 코드 인터페이스는 의사코드로 표시 — 실구현은 Phase 1~4에서 수행.

목표는 두 가지를 분리하는 것이다.
- **(A) Go-Net의 결정 성능**: 본 모델이 "그래도 바둑을 둘 줄 아는가"를 검증.
- **(B) LLM 설명의 충실성**: 본 연구의 핵심 주장이 성립하는가를 검증.

(A)는 비교적 표준이고, (B)가 본 연구가 새로 정립해야 하는 부분이다.

---

## A. 바둑 결정 성능

### A.1 Win Rate (승률)

**정의**. 특정 상대 N판 두었을 때 이긴 비율.

**입력**. 평가 대상 모델 `M`, 상대 정책 `P`, 판수 `N`, 컬러 균형 (선/후 무작위).

**절차**.
1. 게임 i ∈ [1, N]:
   1. 코미 (komi) = 7.5 (9×9 표준)
   2. M과 P가 흑/백을 무작위로 배정 (i가 짝수면 M=흑, 홀수면 M=백)
   3. 각 수마다 양 정책이 합법수 분포에서 sampling 또는 argmax (config로 결정)
   4. 종국 시 중국식 점수로 승패 판정
2. M의 승수를 N으로 나눔.

**출력**. (승률, 95% Wilson 신뢰구간).

**합격 기준 (Phase 1.4)**.
- vs Random: 승률 ≥ 95%
- vs Greedy (Go-Net policy head argmax + no MCTS): 승률 ≥ 80%
- vs 자기 자신 이전 체크포인트: 학습 진행도 따라 단조 증가 (잡음 허용)

**함정**.
- N이 작으면 분산이 큼. 최소 N=200 권장 (95% CI 폭이 ±7%p 이내).
- 컬러 균형을 깨면 코미 영향이 비대칭으로 누적됨.
- "Greedy" 베이스라인을 학습 중인 동일 Go-Net에서 만들면 자기 자신을 측정. 별도 고정 체크포인트로 평가.

**의사코드**.
```python
def winrate(model_a, model_b, n_games=200, komi=7.5):
    wins = 0
    for i in range(n_games):
        a_color = "B" if i % 2 == 0 else "W"
        result = play_one_game(model_a, model_b, a_color, komi)
        wins += int(result.winner == a_color)
    rate = wins / n_games
    lo, hi = wilson_ci(wins, n_games, alpha=0.05)
    return rate, (lo, hi)
```

---

### A.2 ELO (상대 실력)

**정의**. 다중 모델 간 페어와이즈 결과를 단일 점수로 환원.

**입력**. 모델 집합 `{M_1, ..., M_K}`, 모든 페어 (i, j)에 대해 `n_pair` 게임.

**절차**.
1. 모든 페어에 대해 winrate 계산.
2. 결과 행렬을 logistic regression으로 ELO에 fit (PyTorch로 직접 구현 또는 `bayeselo` 등 외부 도구 사용).
3. 임의의 모델 하나를 1500으로 고정해 anchoring.

**출력**. 모델별 ELO와 95% bootstrap CI.

**합격 기준**. Phase 1 종료 시 학습 단계별 모델의 ELO가 단조 증가하면 학습이 진행된 것으로 간주.

**함정**.
- 페어 수가 K(K-1)/2 라 K가 크면 비용 폭발. K ≤ 6, n_pair ≥ 100 권장.
- ELO는 transitivity 가정 — gambit 전략(특정 상대에게만 강함)은 평탄하게 보임. 정성 분석 병행.

---

## B. 설명 충실성 (본 연구의 핵심)

"LLM의 설명이 진짜로 Go-Net의 내부 처리를 반영하는가"를 측정. 세 메트릭은 서로 다른 각도에서 충실성을 본다 — 각각 단독으로는 약하나, 셋이 일치하면 강한 증거.

### B.1 Activation Patching Consistency (APC)

**정의**. Go-Net의 특정 활성화를 인위적으로 다른 값으로 바꿨을 때, LLM 설명이 일관되게 변하는 비율.

**입력**.
- 결합 모델 `J = (Go-Net, Adapter, LLM)`
- 평가 데이터: 국면 집합 `{s_k}`, 각 국면에 대한 "기준" 설명 `e_k`
- 패치 대상 레이어 집합 `L`
- 시도 횟수 `N_patch`

**절차**.
1. 국면 `s_k`에 대해 정상 forward → Go-Net 활성화 `a_k = {a_k^l for l in L}`, LLM 설명 `e_k`.
2. 다른 국면 `s_k'`에서 같은 레이어 활성화 `a_{k'}`을 가져옴.
3. **Patch**: forward 도중 레이어 l의 활성화를 `a_k^l → a_{k'}^l` 로 교체.
4. 패치된 forward로 LLM 설명 `e_k^patched` 생성.
5. **변화 측정**:
   - `Δ_decision`: Go-Net 정책 분포의 KL divergence (원본 vs 패치) — "이 패치가 결정에 영향을 주는가"
   - `Δ_explanation`: 원본 e_k 와 패치 e_k^patched 의 의미 거리 (sentence-BERT 코사인 거리)
6. 한 패치 시도가 "유의미하게 영향"이면 두 조건 모두 만족:
   - Δ_decision > τ_d (예: 0.05)
   - Δ_explanation > τ_e (예: 0.2)
7. **APC = (조건 만족 시도 수) / (Δ_decision > τ_d 인 시도 수)**.
   - 즉, "결정이 바뀌었을 때 설명도 바뀌는 비율".

**출력**. 레이어별 APC ∈ [0, 1].

**합격 기준 (Phase 4)**.
- 본 모델: APC ≥ 0.6 (잠정)
- Random adapter 베이스라인: APC < 0.2 예상 (학습 효과 검증)
- Output-only 베이스라인: APC를 정의하기 어렵지만, 활성화에 접근 안 하니 0에 가까울 것

**함정**.
- `s_k'` 를 너무 다른 국면에서 뽑으면 활성화 분포 자체가 벗어나 LLM이 단순 노이즈로 받아들임. 비슷한 단계(초/중반)에서만 샘플링.
- 의미 거리에 sentence-BERT를 쓰면 그 모델 자체의 편향이 들어옴. Phase 4에서 BLEURT나 NLI 기반 지표로 교차검증.
- τ_d, τ_e 는 사전에 고정해서 보고. p-hacking 방지.

**의사코드**.
```python
def apc(joint, dataset, layer, n_patch=200, tau_d=0.05, tau_e=0.2):
    affected, consistent = 0, 0
    for s_k in dataset:
        a_k, pi_k, e_k = joint.forward_full(s_k)
        for _ in range(n_patch // len(dataset)):
            s_alt = sample_similar_state(dataset, s_k)
            a_alt = joint.gonet.activations(s_alt)[layer]
            pi_p, e_p = joint.forward_with_patch(s_k, layer, a_alt)
            d_decision = kl_div(pi_k, pi_p)
            d_expl = semantic_distance(e_k, e_p)
            if d_decision > tau_d:
                affected += 1
                if d_expl > tau_e:
                    consistent += 1
    return consistent / max(affected, 1)
```

---

### B.2 Counterfactual Consistency (CFC)

**정의**. "Go-Net이 다른 수를 뒀다면 어떤 활성화였을까"를 시뮬레이션해, 그 활성화를 LLM에 주입했을 때 적절히 다른 설명이 나오는가.

APC가 무작위 patch라면 CFC는 **의미 있는** counterfactual.

**입력**.
- 결합 모델 `J`, 국면 `s_k`, Go-Net이 실제 둔 수 `m_k`, top-k 대안 수 `{m_k^1, ..., m_k^K}` (예: K=3).

**절차**.
1. 정상 forward로 `(a_k, e_k)` 획득.
2. 각 대안 수 `m_k^j` 에 대해:
   1. 가상 국면 `s_k^j` (만약 m_k^j 를 두었다면) 구성
   2. Go-Net forward로 `s_k^j` 의 활성화 `a_k^j` 획득
   3. 단, 평가하는 것은 "현재 `s_k` 에서 다른 수를 두려고 하는 Go-Net" 이므로 활성화는 현재 국면의 forward 중간에 적절한 시점에서 m_k^j 의 영향을 반영해야 함. **간단화**: m_k^j 만 다르고 보드는 같은 국면 직전에서 활성화 추출.
   4. 그 활성화로 LLM이 설명 `e_k^j` 생성.
3. **측정**:
   - CFC_diff = mean_j semantic_distance(e_k, e_k^j) — "수가 다르면 설명도 달라야"
   - CFC_align = mean_j alignment(e_k^j, m_k^j) — "다른 설명이 그 다른 수를 실제로 가리키는가"
   - alignment 측정: 설명에 m_k^j 의 좌표/특징이 명시적으로 등장하는 비율 (regex + parsing).

**출력**. (CFC_diff, CFC_align). 둘 다 0~1.

**합격 기준**.
- CFC_diff ≥ 0.3, CFC_align ≥ 0.5 (잠정. Phase 3 중간 검증 결과로 조정).

**함정**.
- "가상 국면"의 정의가 모호. plan 단계에서 "다른 수의 활성화를 같은 국면에서 어떻게 얻을지"를 못 박아야 함. 본 연구 권장 방식: Go-Net 정책 분포에서 top-k를 뽑고, **m_k^j 좌표를 마스킹한 추가 채널**을 입력에 더해 "이 수를 두려고 한다고 가정"한 forward 수행.
- alignment 측정의 false positive: 설명이 m_k^j 좌표를 단순 언급만 해도 정렬된 것으로 보일 수 있음. 좌표 + 의미적 정당화 둘 다 요구.

---

### B.3 Information Ablation Score (IAS)

**정의**. 활성화의 일부 차원을 0으로 만들고 (또는 평균값으로 치환) 설명 품질이 얼마나 떨어지는가.

**입력**. `J`, 평가 데이터, 절제할 차원 비율 `r ∈ {0.1, 0.3, 0.5, 0.9}`.

**절차**.
1. 정상 forward에서 설명 생성과 perplexity p_0 측정 (또는 외부 reference 설명에 대한 token-level NLL).
2. 활성화 텐서의 차원을 균등 무작위로 비율 r 만큼 0으로 마스킹.
3. 동일 절차로 perplexity p_r 측정.
4. **IAS(r) = (p_r − p_0) / p_0** — 상대 증가율.

**출력**. r 별 IAS 곡선.

**합격 기준**.
- r=0.5 에서 IAS ≥ 0.3 (즉 perplexity가 30% 이상 악화) — 활성화가 실제로 사용되고 있다는 증거.
- Random adapter 베이스라인: IAS ≈ 0 (활성화를 무시하므로 절제해도 변화 없음).

**함정**.
- 어느 차원을 마스킹하느냐에 분산이 크다. seed 5개 평균 권장.
- perplexity가 reference 설명에 의존 → reference가 합성 데이터면 자기 분포에 유리. **테스트 분할은 합성 파이프라인에서 분리** 보관.
- 차원이 실제로 0이 되어도 cross-attention의 layer norm 등으로 영향이 퍼지지 않을 수 있음. **마스킹은 어댑터 입력 직전에 적용**.

---

## C. 설명 품질 (보조)

### C.1 Perplexity

**정의**. test split 의 reference 설명에 대한 LLM의 평균 NLL의 exp.

**절차**.
1. 학습/평가 분할: self-play 게임 단위로 80/10/10. 같은 게임의 수가 train과 test에 동시에 들어가지 않도록.
2. test 의 (활성화, reference 설명) 쌍에 대해 forward, token NLL 평균 → exp.

**출력**. perplexity ∈ [1, ∞).

**합격 기준**. 학습 시작 시 대비 50% 이상 감소.

**함정**. 합성 데이터의 템플릿 다양성이 낮으면 perplexity가 비현실적으로 낮아짐 (자기 분포 잘 맞히기). 다양성 증가 후 다시 측정.

---

### C.2 Self-consistency

**정의**. 같은 입력에 N번 sampling 한 설명들의 의미적 평균 거리.

**절차**.
1. 같은 (활성화, 프롬프트)로 temperature=0.7, N=5 샘플링.
2. 모든 페어 의미 거리 평균.

**출력**. self_consistency = 1 - 평균거리. 1에 가까울수록 일관됨.

**합격 기준**. ≥ 0.6.

---

### C.3 Human Evaluation

**정의**. 바둑 ≥ 5급 평가자가 다음 5문항에 1~5점 평가:
1. 설명이 문법적으로 자연스러운가
2. 설명이 그 수와 관련이 있는가 (관련성)
3. 설명이 바둑 개념을 정확히 사용하는가 (전문성)
4. 설명이 구체적인가 (모호하지 않은가)
5. 설명이 통찰을 주는가 (단순 기술 묘사 이상)

**절차**.
- 평가자 ≥ 3명, 각자 무작위 100 샘플 (본 모델/베이스라인 라벨링 가림)
- inter-rater agreement (Krippendorff's α) 보고.

**출력**. 항목별 평균 점수, α.

**합격 기준**. 항목 2~5 평균 ≥ 3.0, α ≥ 0.4.

**함정**. 평가자에게 모델 정체를 가려야 (blind).

---

## D. 베이스라인 정의 (정확한 구현)

본 모델의 우월성 주장은 베이스라인이 강할 때만 성립한다. 각 베이스라인을 가능한 한 fair하게 구성.

### D.1 Baseline 1 — Output-only

LLM이 활성화 없이 Go-Net 출력만 보고 설명. **무엇이 출력인가** 를 명확히:
- 선택된 수 좌표, 정책 분포 top-3, value 추정.
이를 텍스트로 변환해 LLM 입력.

```
"Go-Net selected (3,4). Top-3: (3,4)=0.45, (2,3)=0.20, (4,5)=0.10.
 Value estimate: +0.25. Generate an explanation:"
```

LLM은 본 모델과 같은 LLM (TinyLlama 1.1B), 어댑터 없음. LoRA 파인튜닝은 동일 합성 데이터로 동등 epoch 학습 — fair 비교를 위해.

### D.2 Baseline 2 — Vanilla CoT

사전학습 LLM에 보드 상태(아스키 표현)와 "이 수를 둔 이유를 단계별로 설명" 프롬프트. 추가 학습 없음. 본 모델이 이보다 도메인 정확도가 높아야.

```
. . . . . . . . .
. . . . . . . . .
. . X . . . . . .
. . . X . . . . .
. . . . . . . . .
...
"Black just played at (3,3). Explain step by step why this move makes sense in 9x9 Go."
```

### D.3 Baseline 3 — Random Adapter

본 모델과 동일 구조이나 어댑터 가중치를 학습 안 한 (정규분포 초기화 후 동결) 상태. 활성화 접근의 학습 효과를 격리.

### D.4 Baseline 4 (선택) — Frozen-LLM with text-converted activations

활성화 텐서를 사람이 해석하기 쉬운 형태(top-k 채널의 max 위치 등)로 텍스트화해 LLM에 주입. 본 모델의 cross-attention vs 텍스트 인터페이스의 차이를 본다.

---

## E. 보고 표준

모든 평가 결과는 다음 표 형식으로 보고:

| 메트릭 | Ours | B1 Output-only | B2 Vanilla CoT | B3 Random Adapter | 차이 (Ours − Best B) |
|--------|-----:|---------------:|---------------:|------------------:|---------------------:|
| APC (block-3) | 0.62 | n/a | n/a | 0.18 | +0.44 |
| APC (block-5) | ... | ... | ... | ... | ... |
| CFC_diff | ... | ... | ... | ... | ... |
| CFC_align | ... | ... | ... | ... | ... |
| IAS (r=0.5) | ... | ... | ... | ... | ... |
| Perplexity (test) | ... | ... | ... | ... | ... |
| Self-consistency | ... | ... | ... | ... | ... |
| Human (관련성) | ... | ... | ... | ... | ... |

각 셀: 평균 ± 표준오차 (seed ≥ 3).

---

## F. 통계 보고 규칙

- 모든 평균은 seed ≥ 3 의 평균.
- 차이의 유의성은 paired bootstrap (n=10000).
- 다중 비교 보정: Holm-Bonferroni 또는 Benjamini-Hochberg.
- p-value 만 보고하지 않음. 효과 크기와 신뢰구간 동반.

---

## G. Phase 별 평가 일정

| Phase | 적용 메트릭 |
|-------|-------------|
| 1.4 (Go-Net 평가) | A.1, A.2 |
| 2.4 (forward 통과) | 메모리/속도만 |
| 3.4 (중간 검증) | C.1 perplexity, B.3 IAS 약식 |
| 4.1 | B.1 APC |
| 4.2 | B.2 CFC |
| 4.3 | B.3 IAS 정식 |
| 4.4 | 모든 메트릭 × 모든 베이스라인 |
| 4.5 | C.3 human eval |

---

## H. Decision Point 와의 연결

원본 plan의 Decision Point 1~3에 대응하는 정량 기준:

- **DP1 (Phase 1 종료)**: A.1 의 vs Random ≥ 95%, vs Greedy ≥ 80%. 미달성 시 모델 축소/보드 축소.
- **DP2 (Phase 3 중간)**: B.3 IAS(0.5) ≥ 0.1 + C.1 perplexity 학습 시작 대비 30% 감소. 미달성 시 어댑터 구조 수정.
- **DP3 (Phase 3 종료)**: B.1 APC ≥ 0.4 + B.3 IAS(0.5) ≥ 0.3. 미달성 시 데이터 제약 완화.

이 임계치는 **사전 등록 (pre-registered)** 으로 둔다. Phase 별 평가 직후 임계치를 변경하지 않음.
