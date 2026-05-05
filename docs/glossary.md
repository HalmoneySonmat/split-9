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

이 그룹은 본 연구뿐 아니라 어떤 딥러닝 책에도 등장하는 공통 어휘. 한 번 잡아두면 평생 씀.

먼저 *데이터 형태의 기본 어휘* 를 잡고 신경망으로 넘어간다. 이 단어들은 아래 다른 항목 정의에 계속 등장.

**시퀀스 (Sequence)**
순서가 있는 원소들의 모음. **"순서" 의미 + "배열" 의미를 동시에** 가짐. ML 문맥에서는 거의 항상 "1차원 배열로 표현되는 순서 있는 데이터". 토큰 시퀀스 = 토큰 정수들이 정해진 순서로 늘어선 1차원 배열. "시퀀스 길이 = 100" = 토큰 100개가 차례로 있는 것.

**배열 (Array)**
숫자나 객체를 직사각형(또는 다차원) 으로 늘어놓은 자료. NumPy 의 `ndarray`, PyTorch 의 `Tensor` 가 모두 배열의 일종. 모양(shape) 을 가짐.

**리스트 (List)**
Python 의 일반 자료형. 가변 길이, 다양한 타입 혼합 가능 (`[1, "a", 3.14]` 도 OK). 배열과 비슷해 보이지만 메모리/속도는 배열이 훨씬 우월. ML 코드에서는 "Python list 로 묶인 텐서들" 식으로 자주 등장.

**벡터 (Vector)**
1차원 배열. 길이 하나만 가짐. shape = `(D,)`. 예: `[1.0, 2.0, 3.0]` 은 길이 3의 벡터.

**행렬 (Matrix)**
2차원 배열. 행과 열의 표. shape = `(H, W)`. 신경망의 가중치는 대부분 행렬.

**텐서 (Tensor)**
다차원 배열의 일반화. 1차원 = 벡터, 2차원 = 행렬, 3차원 이상 = 텐서. PyTorch 의 기본 자료형. (아래 항목에서 더 상세.)

**차원 (Dimension)**
한국어/영어 모두에서 두 의미로 쓰여 혼동 주의:
- 의미 1: 배열의 *한 축* (axis). shape `(B, T, D)` 의 텐서는 "3차원" (축 3개).
- 의미 2: 한 축의 *길이*. "hidden dimension 2048" = D = 2048.
- 문맥으로 구분. "텐서가 3차원" → 의미 1. "차원이 2048" → 의미 2.

**Shape (모양)**
배열의 각 축의 길이를 묶은 튜플. `(B, C, H, W)` = 4축, 각각 B, C, H, W 길이. 본 연구 Go-Net 입력 = `(B, 4, 9, 9)`. 모양 불일치가 ML 디버그의 1순위 원인.

**인덱스 (Index)**
배열에서 특정 원소를 가리키는 정수 (또는 정수 튜플). `arr[3]`, `tensor[0, 5, 2]`. Python/NumPy/PyTorch 모두 0부터 시작.

**모델 (Model)**
신경망 인스턴스 1개. `model = TinyLlamaForCausalLM(...)` 처럼 객체로 만든 것. 가중치 + forward 정의를 묶은 단위. "TinyLlama" 같은 이름이 *구조* 를 가리키면, model 은 *그 구조의 한 인스턴스*.

**모듈 (Module)**
PyTorch 의 신경망 단위. `nn.Module` 클래스의 인스턴스. 한 모듈 안에 자식 모듈들이 트리처럼 들어 있고, 가장 바깥 모듈이 model. attention, FFN, 잔차 블록이 모두 모듈.

**임베딩 (Embedding)**
이산 객체 (단어, 토큰, 카테고리) 를 연속 벡터로 변환한 것. "단어 → 2048 차원 벡터". LLM 은 vocab_size × hidden_size 의 거대한 임베딩 테이블을 가지고, 토큰 ID 로 인덱싱해 시작 벡터를 만듦. TinyLlama: 32k × 2048 = 65M 파라미터가 임베딩 테이블 하나에.

**로짓 (Logit)**
softmax 직전의 raw 출력값. **확률이 아님** (음수 가능, 합이 1 아님).
- 예: 분류기 출력 `[3.2, -1.4, 0.5]` — 이게 로짓.
- softmax 를 씌우면 확률 분포 → `[0.85, 0.01, 0.14]`.

**Softmax — 소프트맥스**
임의의 실수 벡터를 "양수이고 합이 1" 인 분포로 변환. 공식: `softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. LLM 의 다음 토큰 확률, Go-Net 의 정책 분포 모두 softmax 출력.

---

**신경망 (Neural Network) — 뉴럴 네트워크**
입력을 받아 여러 "층 (layer)" 을 거쳐 출력을 내는 함수. 각 층이 하는 일은 단순: (1) 입력에 행렬을 곱해 다른 모양의 벡터로 바꾸고, (2) 비선형 함수 (ReLU, GELU 등) 를 적용. 이것을 N번 반복하면 입력→출력의 매우 복잡한 매핑을 표현 가능. 학습은 "원하는 매핑을 만들도록 행렬 안 숫자를 조정" 하는 과정.
- 비유: 11억 개의 다이얼이 달린 거대한 기계. 다이얼을 잘 맞추면 "고양이 사진 → 고양이"라고 답함.
- 본 연구: Go-Net (작은 신경망) 과 LLM (큰 신경망) 두 개 결합.

**가중치 (Weight) / 파라미터 (Parameter)**
신경망이 학습하는 숫자들. 행렬의 원소와 편향(bias) 을 다 합친 것. "1B 파라미터 모델" = 가중치가 10억 개. TinyLlama 1.1B = 11억 개 숫자. 학습이란 결국 이 11억 개 숫자를 데이터에 맞게 조정하는 과정.
- 가중치 1개의 크기: 보통 4바이트 (fp32) 또는 2바이트 (bf16). 1.1B × 2바이트 = 약 2.2GB → VRAM 차지.

**층 (Layer)**
신경망의 한 단위 처리 (행렬 곱 + 비선형). Transformer 의 "22층 모델" = 22개의 attention+FFN 블록을 쌓음. 본 연구의 Go-Net = 잔차 블록 3~6층.

**학습 (Training) vs 추론 (Inference)**
- **학습**: 데이터를 보고 가중치를 조정. forward + backward + optimizer step 의 반복. 시간과 GPU 자원이 많이 듦. 여러 epoch.
- **추론**: 학습된 가중치로 답을 냄. forward 만. 학습보다 3~5배 가볍고 빠름.
- 본 연구는 Phase 1, 3에서 학습, Phase 4에서 추론 위주.

**Forward pass — 순전파**
입력 → 1층 → 2층 → ... → N층 → 출력 까지의 한 번 계산. 학습 중이든 추론 중이든 매 순간 한 번씩 일어남. 학습 중에는 backward 를 위해 모든 중간 결과(활성화) 를 메모리에 임시 보관 — 이게 GPU 메모리 사용량의 큰 부분.
- 비유: 입력을 컨베이어 벨트에 올리면 N개의 가공기계를 차례로 통과해서 끝에서 출력이 나옴.

**Backward pass — 역전파 (Backpropagation)**
forward 의 반대 방향. 출력의 오차(loss) 부터 시작해 각 가중치가 그 오차에 얼마나 기여했는지 계산. 미분의 chain rule (연쇄 법칙) 을 계산 그래프 위에서 거꾸로 풀어내는 것. 결과: 각 가중치에 대한 gradient.
- forward 보다 보통 1.5~2배 비쌈.
- 학습 시에만 필요. 추론은 forward 만.

**Loss (손실, 손실 함수)**
"답이 얼마나 틀렸는가" 를 한 숫자로 만든 것. 학습은 이 숫자를 줄이는 방향으로 가중치를 조정하는 게임. 종류:
- **MSE (Mean Squared Error)**: 회귀에서. 본 연구의 Go-Net value head.
- **Cross-entropy**: 분류에서. 본 연구의 Go-Net policy head 와 LLM 의 next-token prediction.

**Gradient — 그래디언트, 기울기**
"각 가중치를 조금 바꾸면 loss 가 어느 방향으로 얼마나 변하는지" 의 정보. 수학적으로 ∂L/∂w. backward pass 의 결과물이며, optimizer 의 입력.
- 가중치 업데이트 공식 (가장 단순한 SGD): `w_new = w_old - learning_rate × gradient`.
- 실전에서는 Adam, AdamW 같은 더 똑똑한 optimizer 가 표준.

**Optimizer — 옵티마이저**
gradient 를 받아 가중치를 어떻게 업데이트할지 정하는 알고리즘. SGD, Adam, AdamW 등. 본 연구는 AdamW 표준. 옵티마이저 자체도 메모리를 씀 (Adam 은 가중치당 약 2배의 추가 상태).

**Learning rate (학습률, LR)**
가중치를 한 번에 얼마나 크게 바꿀지의 비율. 너무 크면 loss 가 발산해 학습 실패, 너무 작으면 수렴까지 시간 폭발.
- LLM 처음부터 학습: 1e-4 ~ 3e-4
- LLM 파인튜닝: 1e-5 ~ 5e-5
- 어댑터/LoRA 학습: 1e-4 ~ 1e-3
- **LR scheduler**: 학습 중간에 LR 을 점진 감소시키는 기법. warmup (시작 천천히) + cosine decay (점진 감소) 가 표준.

**Batch (배치)**
여러 데이터 샘플을 한 번에 묶어서 처리하는 단위. batch size = 32 → 한 번 forward 에 32개 샘플 동시 처리.
- 큰 batch: GPU 활용 효율 ↑, gradient noise ↓ (안정적), 메모리 ↑.
- 작은 batch: 메모리 절약, gradient noise (정규화 효과).
- 본 연구의 어댑터 학습: 8~32 권장 (VRAM 따라).

**Epoch (에폭)**
전체 데이터셋을 한 번 다 본 단위. "10 epoch" = 데이터를 10번 반복해 학습. 큰 LLM 사전학습은 1 epoch 정도, 파인튜닝은 3~10 epoch, 어댑터 학습은 데이터 양에 따라 다양.

**Tensor (텐서)**
숫자들의 다차원 배열. 1차원 = 벡터, 2차원 = 행렬, 3차원 이상 = 텐서. PyTorch 의 기본 자료형 `torch.Tensor`. 모든 신경망 입출력은 텐서.
- shape 표기: `(B, C, H, W)` = (배치 B, 채널 C, 높이 H, 너비 W).
- 본 연구의 Go-Net 입력: `(B, 4, 9, 9)` — 배치 B개, 채널 4개, 9×9 보드.
- 본 연구의 Go-Net 활성화: `(B, 64, 9, 9)` 또는 `(B, 128, 9, 9)`.
- LLM 입력: `(B, T)` — 배치 B개, 토큰 T개의 정수 시퀀스.
- LLM hidden state: `(B, T, D)` — D 는 hidden size (TinyLlama = 2048).

**Broadcasting**
shape 가 다른 텐서끼리 자동으로 맞춰 연산. 예: `(B, 1, D) + (B, T, D)` 은 첫 번째를 T 번 복제해 `(B, T, D)` 처럼 더함. 메모리는 실제로 복제하지 않음 (가상). 익숙해지면 코드가 짧아지나, 모르고 쓰면 버그 원인 1위.

**Checkpoint (체크포인트)**
학습 중간 또는 끝의 가중치(+옵티마이저 상태) 를 파일로 저장한 것.
- 확장자: `.pt`, `.ckpt`, `.safetensors`. **safetensors 권장** (보안 — 일반 .pt 는 임의 코드 실행 가능).
- 가중치만 저장: `state_dict`. 옵티마이저 상태도 저장하면 학습 재개 가능.
- 본 연구는 매 100 epoch 또는 1000 step 단위 체크포인트.

**OOM (Out Of Memory)**
GPU 메모리 부족 에러. 학습 중 가장 흔한 막힘.
- 원인: 배치 크기, 모델 크기, 시퀀스 길이, 활성화 누적.
- 대응: (1) batch size 절반, (2) gradient checkpointing 켜기, (3) bf16/fp16 사용, (4) 모델 일부 CPU 로 offload, (5) 더 작은 모델.

**부동소수점 정밀도 — fp32 / fp16 / bf16**
- **fp32**: 32비트. 표준. 정확하나 메모리 많이 먹고 느림.
- **fp16**: 16비트. 절반 메모리, 2배 빠름. 단점: 표현 범위가 좁아 학습 중 underflow (작은 gradient 가 0이 됨) 발생 가능.
- **bf16 (bfloat16)**: 16비트인데 지수부 범위는 fp32 와 같음. 즉 메모리는 fp16 처럼, 안정성은 fp32 에 가깝게. 모던 GPU(RTX 30/40, A100, H100) 에서 하드웨어 가속.
- **본 연구는 bf16 권장**.

**Mixed precision (혼합 정밀도)**
forward 는 bf16/fp16 으로 빠르게, optimizer state 는 fp32 로 정확하게. PyTorch 의 `torch.amp` 가 이를 자동화. 학습 속도 1.5~2배, 메모리 30~50% 절감.

**Gradient checkpointing**
GPU 메모리를 줄이는 트릭. forward 도중의 일부 활성화를 메모리에 저장하지 않고, backward 때 다시 forward 를 부분적으로 재계산해 활성화 복원. 메모리 30~50% 절감, 학습 시간 약 20~30% 증가. 큰 LLM 학습에 표준.

**Quantization — 양자화**
모델 가중치를 더 적은 비트로 압축. 메모리/속도 이득, 약간의 품질 손해.
- **int8**: 8비트 정수. 추론 표준. 거의 무손실.
- **int4 / NF4**: 4비트. 메모리 1/8. 약간의 품질 손해.
- 학습 중 양자화는 까다로움 (gradient 흐름 어려움). QLoRA 같은 기법으로 가능.
- **bitsandbytes** 라이브러리가 표준.

**LoRA (Low-Rank Adaptation)**
큰 모델의 가중치 W (예: 2048×2048 행렬) 를 직접 학습하지 않고, 작은 보조 행렬 두 개 (A: 2048×r, B: r×2048, r=8~64) 를 학습해 W' = W + AB 처럼 사용. r 이 매우 작아 학습할 파라미터가 원본의 0.5~5%만. 결과: 메모리/시간 큰 이득, 성능은 거의 full 파인튜닝 수준.
- **peft** 라이브러리가 표준.
- 본 연구는 어댑터 자체가 LoRA 와 별개로 cross-attention 형태이지만, LLM 일부에 추가 LoRA 를 얹는 옵션도 가능 (Phase 3.3 에서 검토).

**Activation function — 활성화 함수**
신경망 층 사이의 비선형 함수. ReLU, GELU, SiLU, tanh 등. **신경망 중간 출력 (activation) 이라는 단어와 헷갈리기 쉬움** — "activation function" (함수) 와 "activation" (값) 은 다른 것. 함수가 값을 만드는 관계.

---

## 4. 본 연구 핵심 개념

이 그룹이 본 연구의 *연구 자체* 에서 가장 자주 등장. 여기 단어들은 본 연구의 모든 문서에 박혀 있음.

**Token (토큰)**
LLM 이 다루는 텍스트의 최소 단위. 단어, 부분 단어, 한 글자가 모두 가능. "Hello world" → 토큰 시퀀스 [`Hello`, ` world`] (공백 포함). LLM 의 vocabulary = 사용 가능한 토큰 종류 (TinyLlama 는 32k 종류).

**Tokenizer — 토크나이저**
텍스트를 토큰 ID 시퀀스로 변환하는 모듈. 모델마다 자기 토크나이저를 가짐. `AutoTokenizer.from_pretrained(...)` 로 로딩.

**LLM (Large Language Model) — 대형 언어 모델**
"다음 토큰을 예측" 작업으로 학습된 큰 신경망. 학습 데이터: 수십 TB 규모 텍스트 (책, 웹, 코드 등). 자기회귀 (autoregressive) 방식으로 한 토큰씩 생성.
- ChatGPT, Claude, Gemini 등이 이 카테고리.
- 본 연구는 작은 LLM (TinyLlama 1.1B) 사용 — "큰 모델" 이라기엔 작지만 같은 구조.
- 학습 목표 (loss): cross-entropy on next-token. 학습이 끝나면 같은 모델로 텍스트 생성 가능.

**Autoregressive — 자기회귀**
시퀀스를 한 단계씩 생성. 토큰 1을 만들면 그것을 입력에 추가해 토큰 2 예측, 또 추가해 토큰 3 예측, ... 의 반복. 본 연구의 LLM 출력 (설명 문장) 은 자기회귀 생성.

**Transformer — 트랜스포머**
2017년 Google 의 "Attention Is All You Need" 논문으로 등장한 신경망 구조. attention 메커니즘이 핵심. 거의 모든 현대 LLM 의 기본 골격. Llama, GPT, BERT, T5, Mistral 모두 변형.
- 한 layer 의 구성: self-attention → 잔차 → layer norm → FFN → 잔차 → layer norm.
- TinyLlama: 22 layer, hidden size 2048, attention head 32.

**Attention — 어텐션**
시퀀스 안에서 어떤 위치가 어떤 위치에 주의를 기울이는지의 메커니즘. 수학적으로:
- Query, Key, Value 세 행렬 (모두 입력에서 학습된 선형 변환으로 계산).
- `attention(Q, K, V) = softmax(QK^T / √d) · V`.
- 의미: Q 와 K 의 유사도로 V 를 가중평균. 즉 "이 단어가 다른 단어들 중 어디를 얼마나 참조해야 하는가" 를 학습된 방식으로 결정.

**Self-attention — 자기 어텐션**
Q, K, V 모두 같은 시퀀스에서 나옴. "I went to the bank" 에서 `bank` 토큰의 Q 가 다른 모든 토큰의 K 와 비교되어, `to`/`went` 가 강하게 가중되면 "강가의 둑" 보다 "은행" 이 적합한 의미라고 판단되는 식. Transformer 의 매 layer 에 self-attention 이 들어 있음.

**Cross-attention — 교차 어텐션 (★ 본 연구의 핵심)**
Q 는 한 시퀀스(예: LLM 텍스트), K/V 는 다른 시퀀스(예: Go-Net 활성화) 에서 나옴. 즉 두 다른 정보 흐름이 한 attention 으로 연결. **본 연구의 어댑터가 정