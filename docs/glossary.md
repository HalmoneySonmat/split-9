# 용어집 — 본 연구에서 사용하는 단어들

본 연구의 다른 문서에 등장하는 용어를 그룹별로 설명. 각 용어는 (영어 원문, 한국어, 한 줄 설명, 비유/예시) 구조.

---

## 0. 사용법

- 다른 문서를 읽다가 모르는 단어가 나오면 여기서 찾기.
- 한 번 이해하면 됨. 외울 필요 없음.
- 아래 그룹 중 1, 2, 3, 5 번을 먼저 읽으면 다른 문서가 훨씬 쉬워짐.

---

## 1. 운영체제와 개발 환경

본 연구는 Windows에서 시작하지만, 머신러닝 개발은 Linux가 표준이라 Linux를 함께 사용한다.

**OS (Operating System) — 운영체제**
컴퓨터의 가장 기본 소프트웨어. Windows, macOS, Linux 가 대표적. 우리 PC가 Windows 라면, 그 위에 Linux를 추가로 켜는 셈.

**Linux — 리눅스**
오픈소스 운영체제. 서버와 머신러닝에서 표준. 무료. Ubuntu 는 Linux 의 한 종류 (배포판). "Linux 를 쓴다" = 보통 "Ubuntu 같은 배포판을 쓴다" 와 같은 뜻.

**Ubuntu — 우분투**
가장 많이 쓰이는 Linux 배포판. 본 연구는 22.04 버전 사용 (장기 지원, 안정).

**Terminal / Shell — 터미널 / 셸**
글자로 명령을 입력하는 창. Windows의 cmd 또는 PowerShell, Mac/Linux의 Terminal. **Bash** 는 Linux 에서 기본으로 쓰이는 셸 종류. 클릭 대신 글자로 컴퓨터에 명령하는 방식.

**CLI (Command Line Interface) — 명령행 인터페이스**
터미널에 글자를 입력해 프로그램을 쓰는 방식. 반대말은 GUI (마우스로 클릭).

**WSL (Windows Subsystem for Linux)**
Microsoft가 만든 "Windows 안에서 Linux 돌리기" 공식 기능. Windows 사용자가 Linux 환경을 쓸 때의 표준 방법. **WSL2** 가 최신 버전 (가상머신 방식이라 더 빠르고 호환성 좋음).

**드라이버 (Driver)**
하드웨어 (그래픽카드 등) 와 OS 사이의 통역기 소프트웨어. NVIDIA 드라이버 = NVIDIA 그래픽카드를 OS가 쓸 수 있게 해주는 프로그램.

**GPU (Graphics Processing Unit)**
원래는 그래픽 처리 장치였으나, 행렬 연산을 동시에 수많이 할 수 있어 딥러닝 학습에 필수. 본 연구는 NVIDIA RTX 30/40 시리즈 사용.

**VRAM (Video RAM)**
GPU 안의 메모리. 모델과 데이터가 여기 들어가야 GPU가 처리 가능. RTX 4090 = 24GB, RTX 4060 = 8GB. 클수록 큰 모델을 쓸 수 있다.

**CUDA**
NVIDIA가 만든 GPU 프로그래밍 플랫폼. PyTorch 등 딥러닝 라이브러리가 GPU를 쓰려면 CUDA 가 깔려 있어야 함. 버전이 중요 (CUDA 12.1 등).

**nvidia-smi**
GPU 상태 (모델명, VRAM 사용량, 드라이버 버전) 를 보는 명령어. 터미널에서 `nvidia-smi` 입력.

---

## 2. Python 생태계

**Python — 파이썬**
프로그래밍 언어. 머신러닝 표준. 본 연구는 3.10 버전 사용 (3.10 = 메이저.마이너 버전).

**pip**
Python의 패키지 설치 도구. `pip install <패키지>` 로 외부 라이브러리 설치. (Node.js의 npm, Java의 maven 같은 것)

**가상환경 (Virtual Environment, venv)**
프로젝트마다 별도의 Python 작업 공간을 만드는 기능. 한 프로젝트가 PyTorch 2.3 을 쓰고, 다른 프로젝트가 PyTorch 1.x 를 쓸 때 충돌 없이 공존 가능. **본 연구의 모든 패키지는 `.venv/` 안에만 설치됨**.

**requirements.txt**
프로젝트가 필요로 하는 패키지 목록 (버전 포함) 을 담은 파일. 다른 컴퓨터에서 같은 환경을 재현 가능.

**의존성 (Dependency)**
"이 패키지 A 가 동작하려면 패키지 B가 필요하다" 의 관계. 의존성 충돌 = 두 패키지가 서로 다른 버전의 같은 의존성을 요구해서 깔리지 않는 상황.

**패키지 (Package) / 라이브러리 (Library)**
거의 같은 뜻. 누군가 만들어 둔 코드 묶음. `pip install` 로 받아 import 해서 사용.

---

## 3. 딥러닝 기초

**신경망 (Neural Network)**
입력을 받아 여러 층을 거쳐 출력을 내는 함수. 각 층은 행렬 곱 + 비선형 함수의 반복. 학습을 통해 입출력 매핑을 배움.

**가중치 (Weight) / 파라미터 (Parameter)**
신경망이 학습하는 숫자들. 행렬과 편향들. "1B 파라미터 모델" = 가중치 숫자가 10억 개.

**학습 (Training) vs 추론 (Inference)**
- 학습: 데이터를 보고 가중치를 조정하는 과정. 시간과 GPU가 많이 든다.
- 추론: 학습된 모델로 답을 내는 과정. 학습보다 훨씬 가벼움.

**Forward pass — 순전파**
입력 → 신경망 → 출력 까지 가는 한 번의 계산.

**Backward pass — 역전파**
출력의 오차를 가지고 거꾸로 가중치를 어떻게 바꿀지 계산하는 과정. 학습 때만 함.

**Loss (손실 함수)**
"답이 얼마나 틀렸는가" 를 숫자로 만든 것. 학습은 이 숫자를 줄이는 방향으로 가중치를 조정.

**Gradient — 그래디언트, 기울기**
"가중치를 어느 방향으로 바꾸면 loss가 줄어드는가" 의 정보. backward pass 의 결과물.

**Learning rate (학습률, LR)**
가중치를 한 번에 얼마나 크게 바꿀지의 비율. 너무 크면 발산, 너무 작으면 안 배움.

**Batch (배치)**
여러 데이터를 한 번에 묶어서 처리하는 단위. batch size = 32 → 한 번에 32개씩 처리.

**Epoch (에폭)**
전체 데이터를 한 번 다 본 단위. "10 epoch 학습" = 데이터 전체를 10번 반복해 학습.

**Tensor (텐서)**
숫자들의 다차원 배열. 1차원 = 벡터, 2차원 = 행렬, 3+차원 = 텐서. PyTorch 의 기본 자료형. 예: 9×9 바둑판 + 채널 17개 = (17, 9, 9) 텐서.

**Checkpoint (체크포인트)**
학습 중간 또는 끝의 가중치를 파일로 저장한 것. `.pt`, `.ckpt`, `.safetensors` 확장자. 나중에 로딩해서 이어 학습하거나 추론.

**OOM (Out Of Memory)**
GPU 메모리 부족 에러. 배치 크기 줄이기 / 모델 줄이기 / fp16 사용 등으로 대응.

**bf16 / fp16 / fp32**
숫자의 정밀도. fp32 = 32비트 부동소수점 (정확하나 느리고 메모리 많이 씀). fp16 = 16비트 (절반 메모리, 빠름). bf16 = 16비트인데 범위가 fp32에 가깝게 설계. **본 연구는 bf16 권장** (RTX 30/40 지원).

**Gradient checkpointing**
GPU 메모리를 줄이는 트릭. 일부 활성화를 저장 안 하고 backward 때 다시 계산. 메모리는 덜 쓰지만 학습 시간은 약간 증가.

**Quantization — 양자화**
모델 가중치를 더 적은 비트로 줄임 (8비트, 4비트). 메모리/속도 이득, 약간의 품질 손해. **bitsandbytes** 라이브러리가 표준.

**LoRA (Low-Rank Adaptation)**
큰 모델의 가중치를 직접 학습하지 않고, 작은 보조 행렬만 학습하는 기법. 메모리/시간 이득. **peft** 라이브러리가 표준.

---

## 4. 본 연구 핵심 개념

**LLM (Large Language Model) — 대형 언어 모델**
글자 시퀀스를 받아 다음 글자를 예측하도록 학습된 신경망. ChatGPT, Claude 같은 것의 본체. 본 연구는 작은 LLM (TinyLlama 1.1B) 사용.

**Transformer — 트랜스포머**
2017년 발표된 신경망 구조. attention 메커니즘이 핵심. 현재 LLM의 표준 골격. "Llama, GPT, BERT 등" 은 모두 transformer 의 변형.

**Attention — 어텐션, 주의**
시퀀스 안에서 어떤 위치가 어떤 위치에 주의를 기울이는지의 메커니즘. "이 단어를 이해할 때 어느 다른 단어들을 참고해야 하는가" 를 학습.

**Self-attention — 자기 어텐션**
같은 시퀀스 내부에서의 attention. "I went to the bank" 에서 bank 를 이해할 때 to, went 를 참고하는 식.

**Cross-attention — 교차 어텐션**
두 다른 시퀀스 사이의 attention. **본 연구의 핵심**. LLM이 자기 텍스트를 처리하면서 동시에 Go-Net 활성화에 attention 을 줌.

**Residual block (잔차 블록)**
"입력 + 함수(입력)" 형태의 신경망 블록. 깊은 신경망이 학습 가능하게 해준 핵심 발명. ResNet, Transformer 모두 사용.

**Policy head / Value head**
바둑 신경망의 두 출력 머리.
- Policy head: 다음 수의 확률 분포 (어디에 둘까)
- Value head: 현재 국면에서의 승률 (-1~+1)

**Self-play — 자기 대국**
모델이 자기 자신과 게임을 두면서 데이터를 만들어 학습하는 방식. AlphaGo Zero 의 핵심.

**MCTS (Monte Carlo Tree Search) — 몬테카를로 트리 탐색**
게임의 가능한 수들을 트리 형태로 탐색하는 알고리즘. AlphaGo 가 신경망 + MCTS 조합으로 강해짐. "한 수마다 100번 시뮬레이션" 같은 식.

**Adapter — 어댑터**
큰 모델 본체는 그대로 두고, 작은 모듈만 추가 학습하는 부분. 본 연구의 어댑터는 Go-Net 과 LLM 사이의 번역기 역할.

**Gating (게이팅)**
"이 입력을 얼마나 받아들일지" 를 0~1 사이 값으로 조절하는 메커니즘. **tanh-gate** = tanh 함수로 구현된 gate. 학습 초기에 0 으로 시작해 어댑터 영향을 점진 도입.

**Activation (활성화)**
신경망 중간 층의 출력값. forward pass 도중의 텐서. **본 연구의 주요 관찰 대상**: Go-Net 의 중간 층 활성화를 LLM 에 전달.

**Hidden state — 은닉 상태**
Transformer 의 각 층 출력. 활성화의 한 종류.

---

## 5. 라이브러리 / 도구

**PyTorch**
신경망 프레임워크. 본 연구의 모든 모델/학습 코드의 기반. import torch 로 시작.

**HuggingFace**
사전학습 모델과 데이터셋의 마켓플레이스. 회사이자 라이브러리. **transformers** 라이브러리가 가장 유명. 원클릭으로 모델 로딩.

**transformers (라이브러리)**
HuggingFace의 라이브러리. `AutoTokenizer`, `AutoModelForCausalLM` 등으로 사전학습 모델 사용.

**TinyLlama**
1.1B 파라미터의 작은 LLM. Llama 구조를 줄인 것. RTX 4060 정도에서도 fine-tune 가능.

**Llama**
Meta가 만든 오픈 LLM 시리즈. 본 연구 LLM 의 구조적 기반.

**TransformerLens**
Transformer 의 내부를 들여다보는 (mechanistic interpretability) 라이브러리. Phase 4 의 충실성 평가에 사용.

**OpenSpiel / pyspiel**
DeepMind가 만든 게임 환경 라이브러리. 70여 개 게임 (바둑 포함) 지원. `pyspiel` 은 그 Python 인터페이스.

**wandb (Weights & Biases)**
실험 결과를 자동으로 기록·시각화해주는 클라우드 서비스. "어제 학습이 어떻게 됐지" 를 그래프로 봄. 무료 plan 충분.

**TensorBoard**
구글이 만든 학습 시각화 도구. wandb 와 비슷하나 로컬에서 작동.

**accelerate**
HuggingFace 라이브러리. 여러 GPU/메모리 절약 트릭을 자동화. 본 연구에서는 단일 GPU지만 device 관리에 유용.

**bitsandbytes**
양자화 라이브러리. 8/4비트 모델 로딩을 가능하게 함.

**peft (Parameter-Efficient Fine-Tuning)**
LoRA 등 효율적 파인튜닝 기법 라이브러리.

**Hydra**
설정 (config) 관리 도구. YAML 파일로 하이퍼파라미터 정리. 같은 코드를 다른 설정으로 실험하기 쉬움.

**einops**
텐서 모양 변환을 직관적인 문자열로 표현하는 라이브러리. `rearrange(x, "b h w c -> b c h w")` 같은 식.

---

## 6. 평가와 통계

**ELO**
체스/바둑의 상대 실력 점수. 본 연구에서는 학습 단계별 모델끼리 게임시켜 ELO 산출.

**Win rate (승률)**
N판 중 이긴 비율. 단순하지만 강력한 지표.

**Wilson confidence interval — 윌슨 신뢰구간**
승률 같은 비율의 신뢰구간을 계산하는 표준 방법. "승률 65% ± 7%p" 같은 결과의 ± 부분.

**KL divergence — KL 발산**
두 확률분포의 차이를 재는 표준 척도. 0이면 같음, 클수록 다름. 본 연구에서는 정책 분포의 변화 측정에 사용.

**Bootstrap — 부트스트랩**
데이터를 무작위로 재샘플링해서 통계량의 분포를 추정하는 방법. 표본이 작을 때의 신뢰구간 계산.

**Paired bootstrap**
두 모델의 차이가 유의한지 보는 부트스트랩 변형. 같은 입력에 대한 두 모델의 결과를 페어로 비교.

**P-value, p-hacking**
- p-value: "이 차이가 우연일 확률" 의 추정치.
- p-hacking: 결과가 좋아 보일 때까지 분석 방법을 바꾸는 부정행위. 사전 등록으로 방지.

**Pre-registered (사전 등록)**
실험 전에 분석 방법과 임계치를 박제하는 관행. 결과 본 후 임계치를 바꿀 수 없음. 본 연구의 평가 임계치는 모두 사전 등록.

**Multiple comparison correction (다중 비교 보정)**
지표 100개를 비교하면 우연히 5개는 "유의해 보임" — 이를 보정하는 통계 기법. **Holm-Bonferroni**, **Benjamini-Hochberg** 가 표준.

**Semantic distance — 의미 거리**
두 문장이 얼마나 다른 의미인가를 숫자로. **sentence-BERT** 등의 모델이 문장을 벡터로 만들면, 그 벡터들 사이 거리.

**Sentence-BERT**
문장을 의미 벡터로 변환하는 모델. 두 벡터 사이의 코사인 거리로 의미 유사도 측정.

**BLEURT, NLI**
문장 의미 비교의 다른 방법들. NLI = Natural Language Inference (문장 A 가 B 를 함의하는가).

**Perplexity (혼란도)**
LLM이 reference 텍스트를 얼마나 잘 예측하는지의 척도. 낮을수록 좋음. 1에 가까울수록 완벽 예측, 1000+ 면 거의 무작위.

**NLL (Negative Log-Likelihood)**
로그 확률에 음수 부호. perplexity = exp(평균 NLL).

**Temperature sampling — 온도 샘플링**
LLM이 다음 단어를 뽑을 때의 무작위성 정도. T=0 이면 항상 최고 확률만 (deterministic), T=1 이면 표준, T 클수록 창의적/혼란.

**Self-consistency — 자기 일관성**
같은 입력으로 여러 번 샘플링했을 때 결과가 얼마나 일관되는지.

**Inter-rater agreement — 평가자 간 일치도**
여러 사람이 같은 것을 평가할 때 얼마나 의견이 일치하는지. **Krippendorff's α** 가 표준 지표 (0~1, 0.4+ 면 acceptable).

---

## 7. 해석가능성 (Mechanistic Interpretability)

본 연구의 학문적 정체성이 자리잡은 분야. Anthropic 등이 활발히 연구.

**Mechanistic interpretability**
"신경망 내부에서 실제로 무슨 일이 일어나는가" 를 연구하는 분야. 가중치, 활성화의 의미를 해독.

**Hook (훅)**
PyTorch 의 forward pass 도중에 활성화를 가로채거나 바꾸는 기능. 모델 코드 수정 없이 외부에서 활성화를 관찰/수정 가능.

**Activation patching — 활성화 패칭**
신경망의 특정 층 활성화를 다른 값으로 바꿔치기 한 후, 출력이 어떻게 변하는지 관찰. "이 층이 정말 결과에 영향을 주는가" 의 인과 검증. 본 연구의 충실성 평가의 한 축.

**Counterfactual — 반사실, 가상**
"이게 X 였다면 어땠을까" 의 시뮬레이션. 본 연구에서는 "Go-Net 이 다른 수를 뒀다면 어떤 활성화 + 설명이 나왔을까".

**Ablation — 절제, 제거**
모델의 일부를 무력화하고 결과가 얼마나 나빠지는지 관찰. 그 부분의 기여도 측정.

**Faithfulness — 충실성**
설명이 모델의 실제 처리 과정을 반영하는 정도. 본 연구의 핵심 개념.

**Patching consistency, Counterfactual consistency, Information ablation score**
본 연구의 충실성 메트릭 3종. evaluation_protocol.md 참고.

---

## 8. 소프트웨어 도구

**Git, Repository (저장소), Commit (커밋), Tag (태그)**
- Git: 코드 변경 이력을 관리하는 도구.
- Repository: 한 프로젝트의 git 단위 (저장소).
- Commit: "여기까지의 변경을 한 묶음으로 저장" 의 단위.
- Tag: 특정 시점에 라벨 (예: v0.1-phase0).

**GitHub**
git 저장소의 클라우드 호스팅 서비스. 본 연구는 GitHub 사용 권장.

**pyproject.toml**
Python 프로젝트의 메타데이터 + 도구 설정 파일. requirements 와 별개.

**Linter — 린터**
코드의 스타일/실수를 자동 검사하는 도구. **ruff** 가 빠르고 표준화됨.

**Type checker — 타입 검사기**
"이 함수에 넘긴 인자의 타입이 맞는가" 를 검사. **mypy** 가 표준.

**pytest**
Python 테스트 도구. `tests/` 폴더에 `test_*.py` 만들면 자동 실행.

**ADR (Architecture Decision Record)**
"우리가 이렇게 결정한 이유" 를 1~2쪽으로 박제하는 문서 형식. 6개월 후 "왜 9×9 였더라?" 같은 질문에 답이 됨.

**License (라이선스)**
오픈소스 코드 사용 조건.
- **MIT**: 거의 자유, 출처만 표기.
- **Apache 2.0**: MIT 와 비슷, 특허 조항 추가.
- **GPLv3**: 차용하면 자기 코드도 GPLv3 로 공개해야 함 (전염성). **본 연구는 GPLv3 코드 직접 차용 금지**.

---

## 9. 본 연구 특수 용어

**분리뇌 (Split-brain)**
원래는 좌뇌와 우뇌를 잇는 뇌량을 절단한 인간의 인지과학 모델. Gazzaniga 의 실험으로 유명. 본 연구는 이 모티프를 인공적으로 재현.

**좌반구 해석자 (Left-hemisphere interpreter)**
Gazzaniga 의 가설. 인간의 좌뇌가 자신의 행동을 사후에 일관된 서사로 짜맞추는 모듈. 본 연구의 LLM 이 이 역할.

**충실성 vs 합리화**
- 충실성 (faithfulness): 설명이 진짜 내부 처리를 반영함.
- 합리화 (rationalization): 그럴듯한데 실제 처리와 무관한 설명.
- 본 연구는 "완벽한 충실성 X, 인과적으로 연결된 일관된 서사 O" 를 목표.

**단방향 분리뇌 vs 뇌량 모델**
- 단방향: Go-Net → LLM 만. 본 연구.
- 뇌량 모델: LLM → Go-Net 도 가능. 향후 연구.

**Go-Net**
본 연구의 바둑 신경망 (AlphaGo Zero 스타일 축소판) 의 호칭. 결정자 역할.

**합성 설명 (Synthetic explanation)**
인간이 쓴 해설 데이터 없이, 게임 내부 신호 (정책 엔트로피, 영역 변화 등) 로부터 자동 생성된 설명 문장. 본 연구의 학습 타깃.

---

## 10. 단어 찾기 빠른 인덱스

알파벳 순.

- ablation → 7
- activation → 4
- adapter → 4
- attention → 4
- batch → 3
- bf16/fp16 → 3
- bitsandbytes → 5
- bootstrap → 6
- checkpoint → 3
- counterfactual → 7
- cross-attention → 4
- CUDA → 1
- ELO → 6
- epoch → 3
- forward / backward → 3
- gating → 4
- gradient → 3
- HuggingFace → 5
- Hydra → 5
- KL divergence → 6
- learning rate → 3
- LLM → 4
- LoRA → 3
- MCTS → 4
- nvidia-smi → 1
- OpenSpiel → 5
- patching → 7
- perplexity → 6
- policy/value head → 4
- pre-registered → 6
- pyproject.toml → 8
- PyTorch → 5
- quantization → 3
- requirements.txt → 2
- residual block → 4
- self-attention → 4
- self-play → 4
- sentence-BERT → 6
- tensor → 3
- TinyLlama → 5
- transformers (lib) → 5
- TransformerLens → 5
- ubuntu → 1
- venv → 2
- VRAM → 1
- wandb → 5
- WSL2 → 1
