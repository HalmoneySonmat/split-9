# SPLIT-9

*9×9 Go-Net に対する事後 LLM アダプタの忠実性監査。*

> **TL;DR.** 凍結された 9×9 Go-Net に、凍結された TinyLlama-1.1B を
> Flamingo 風アダプタで接続し、Go-Net の着手を自然言語で「説明」する
> ように訓練した。検証 perplexity は **16.1 → 1.65** と劇的に下がった。
> ただし三つの独立な probe で見ると、その改善のおよそ **95% は
> 学習されたドメイン事前分布**であって、盤面ごとに変わる忠実な信号
> ではない。サンプル 10 件のうち、actor の実際の着手座標を出力できた
> のは **0 / 10** 件。出力はデータセット中で頻度の高い座標
> (`(2,1)`、`(8,8)`、`pass`)に mode collapse する。アダプタが
> 学んだのは説明文の *文法* であって、*中身* ではない。

---

## なぜ作ったのか

Roger Sperry と Michael Gazzaniga による 60〜70 年代の分離脳実験で
気持ちの悪い現象が観察された。脳梁(corpus callosum)を切断された
患者に対し、左視野だけに(つまり右半球だけに)指示を見せると、患者は
言語的な自覚なしにその指示を実行する。「なぜそうしたのか」と尋ねると、
言語を司る左半球は、自分が見ていない決定について *もっともらしい*
理由をでっち上げる。意識的に嘘をついているのではなく、本人は
「自分はこう考えてこうした」と本当に信じている。Gazzaniga はこの
モジュールを **左半球通訳者(left-hemisphere interpreter)** と呼んだ。
流暢で、もっともらしくて、しかし頻繁に間違える。

現代の AI システムは、この構造を **意図的に** 同じ形で繰り返している。
非言語のネットワーク(視覚エンコーダ、ロボットポリシー、科学モデル)
の隣に LLM を置き、前者がしていることを自然言語で語らせる。
RLHF chain-of-thought、視覚-言語アシスタント、「説明可能な」RL
エージェント――どれも同じアーキテクチャの変種だ。LLM が本当に上流の
信号を *翻訳* しているのか、単に表面統計に沿った流暢なテキストを
出しているだけなのかは、ほとんどの場合まだ実証的に確かめられていない。

SPLIT-9 は、その問いを最小限の再現可能な形で叩いてみるための
testbed である。

- 9×9 **Go-Net**(AlphaGo Zero 風、~5M パラメータ、完全 white-box)を「右半球」
- 凍結された **TinyLlama-1.1B** を「左半球」
- 学習可能な **Flamingo 風アダプタ** を「corpus callosum」
- 通訳者が confabulation している現場を押さえるための一連の faithfulness probe

8 GB の民生用 GPU 1 枚で 1 日以内に最初から最後まで回せる。すべての
コンポーネントがこのリポジトリに入っている。

---

## 構成物

```
SPLIT-9
├── Go-Net                  (Phase 1 で学習後、凍結)
│       │ レイヤ {3, 4, 5} の活性化
│       ▼
├── AsymmetricPerceiverResampler        ← 学習可能
│       │ 36 個の latent token (層あたり 8 / 12 / 16)
│       ▼
├── GatedCrossAttention ブロック — TinyLlama 層 {10, 16, 18, 20} に注入
│       │
│       ▼
├── TinyLlama-1.1B          (凍結)
│       │ 32k 語彙 logit
│       ▼
└── 合成された自然言語 reasoning trace
```

**データ。** self-play 30 局 × 平均 ~75 手 = 2,293 サンプル。各サンプル
は `(盤面, 着手, MCTS principal variation, 合成説明文)`。説明テンプレート
6 種 × 単語プールによる変化(verb pool、noun pool)。

**学習。** bf16、AdamW、lr 1e-4、batch 4、3 epoch。RTX 3070 Ti で
約 30 分。アダプタ(~404M パラメータ)のみ更新、両方の土台ネットワーク
は凍結。

---

## 何を測ったか

コストと情報量の順に 4 つの probe。

### 1. ベースライン

| | val loss | val ppl |
|---|---|---|
| Output-only(アダプタを bypass)        | 2.78 | 16.14 |
| Random-init アダプタ(checkpoint なし)  | 2.78 | 16.14 |
| **学習済みアダプタ**                   | **0.50** | **1.65** |

学習済みアダプタは Output-only に対して loss を **−81.9%** 削る。
Random vs Output-only が同じ値になるのは、Flamingo の gate がゼロ
初期化されており、初期状態でアダプタが事実上 no-op だから。
(方法論的な留意点:Random-Adapter は gate を非ゼロで初期化しない
限り意味のあるベースラインにならない。)

### 2. Information Ablation Score (IAS)

Go-Net の活性化テンソルがアダプタに届く前に、チャネルの一定割合 *r* を
ランダムにゼロ化し、loss がどれだけ上がるかを見る。

| r | loss | base からの Δloss |
|---|---|---|
| 0.00 | 0.503 | — |
| 0.25 | 0.507 | +0.4 % |
| 0.50 | 0.521 | +3.5 % |
| 0.75 | 0.556 | +10.6 % |
| 1.00 | 0.607 | +20.6 % |

曲線は *凸(convex)*。チャネルを半分落としてもほぼ変化がない。これは
信号が冗長に符号化されている典型的な特徴で、IAS 単体では本質的に
甘い probe になる。

### 3. Activation Patching Consistency (APC)

同じ prompt + label に対して 2 回 forward を回す。**matched**(正しい
活性化)vs **mismatched**(batch 内 derangement で別ゲームの活性化に
入れ替えたもの――アダプタは説明文が指している盤面とは別の盤面の活性化
を見る)。

| | loss |
|---|---|
| matched          | 0.503 |
| mismatched       | 0.594 |
| **APC**          | **+0.181** |

3 seed、σ ≈ 0.003 ―― 非常に安定。mismatched のペナルティは
**IAS@0.5 の 5 倍**。盤面ごと swap するとチャネル冗長性の問題を回避
できるからだ。APC = 0.18 は盤面固有信号が因果的に流れていることを
確認するが、我々のヒューリスティック閾値では「FAITHFUL」ではなく
「PARTIAL」止まり。

### 4. 定性検査(オチ)

10 個のランダム val 盤面に対して prompt を greedy 復号した結果。

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

3 つ目立つ点。

1. **OUTPUT-ONLY は退化している。** どんな入力にも同じ一般的な文学
   分析テキストを返す。つまり TinyLlama 単体は囲碁について何も知らない。
   これにより「テンプレート暗記」仮説は完全に却下され、ドメインの
   仕事の **100%** がアダプタによるものだと確定する。

2. **TRAINED は合成テンプレートの文法を完璧に再現する:**
   `[middlegame/21] B (2, 1) (p=0.01, v=-0.99, Δv=+0.11) selected`。
   phase タグ、色、座標の括弧、`p=`、`v=`、`Δv=`、`selected` まで
   すべて完全に一致。

3. **しかし TRAINED は実際の着手についてはほぼ常に間違える。** サンプル
   10 個のうち、actor が選んだ座標と一致したのは **0 個**。出力座標は
   `(2,1)`(4 回)、`(8,8)`(3 回)、`pass` などごく少数の高頻度
   座標に集中する ―― 典型的な、データセットでよく出る着手への
   mode collapse だ。

---

## Loss 分解(見出し)

3 つの probe を raw loss の単位で並べると次のようになる:

```
Output-only(アダプタなし)          loss 2.78    ┐
                                                 │ 総削減量の ~95%
Trained, 活性化を全ゼロ化(mask=1)   loss 0.61    ┘   = 学習されたドメイン事前
                                                 ┐
Trained, mismatched (APC)            loss 0.59   │ 総削減量の ~5%
Trained, matched(正常)              loss 0.50   ┘   = 盤面固有信号
```

アダプタの寄与は、**盤面非依存のドメイン事前分布**(絶対値で 78% の
loss 削減、改善総量の 95%)と、もっと小さい **盤面固有の因果信号**
(loss で 0.09、5%)に分解される。後者は実在しかつ統計的に頑健だが、
モデルが *どの mode に collapse するか* を少しずらす程度で、正しい
マスを選ぶほどの精度はない。

---

## これが意味するもの

**Cross-entropy 学習は合理的に事前分布へリソースを投下する。** 30 token
の reasoning trace のうち、盤面固有の token(座標)は約 3 個、構造
token(phase タグ、括弧、`selected`、句読点)は約 27 個。最適化は loss
を最小化するために 27 個を完璧に当て、3 個を mode-collapse させる
方が効率的だと判断する。APC はその残った小さな盤面感度を捉えるが、
cross-entropy 単体ではこのギャップは何も見えない。

**これはチューニングではなく構造的な天井である。** アダプタを大きく
しても、学習を長くしても、同じテンプレート分布の中でデータを増やしても、
prior の当て込みが上手くなるだけで、盤面固有内容の精度はほとんど
動かない。本気で faithfulness を上げるには **データ分布そのものを
変える** 必要がある ―― 各サンプルに盤面依存の内容をもっと多く入れる、
または「選ばれた座標が出力に現れる」ことを明示的に報酬とする
auxiliary loss を加える、などだ。

**そして、これは一般化する。** 同じ token-エントロピー重みづけの議論は
ほとんどの事後アダプタ setup に当てはまる。合成 caption の上での
visual question answering、自分の行動を語る RL agent、next-token loss
だけで評価される chain-of-thought rationale ―― すべて同じ落とし穴。
APC のような ablation probe + 定性検査が **必要条件** であり、
train/val perplexity だけではこの失敗は見えない。

---

## 今後 ―― 共同発達する双子ネットワーク

事後アダプタは、データ分布が「翻訳しやすく」用意してくれた信号しか
翻訳できない。特定の内容を強制することはできない。生物学上の corpus
callosum にこの問題はない ―― 両半球が同じ発達期に共に育ち、最初から
互いに整合する表現を学んでいるからだ。

自然な次のステップ:2 つのネットワークを **同時に** 学習させ、間に
学習可能な bottleneck を置き、上流の特定の特徴(選ばれた手、相手の
直前の手、取られた石)が言語ヘッドの出力に具体的な token として現れる
ように強制する auxiliary loss を加える。これは生物学的共同発達の
模倣であり、上の構造的議論からすると、mode collapse の天井を破れる
ほぼ唯一の方向だ。

---

## 再現方法

```bash
# 0. 環境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Go-Net を学習(3070 Ti、default config で約 3 時間)
python scripts/train_gonet.py --config configs/gonet/default.yaml

# 2. Phase 3 データセット生成(MCTS PV つき self-play)
python scripts/generate_phase3_data.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --n-games 30 --n-simulations 100 \
    --output runs/phase3_data_small.pkl

# 3. アダプタ学習(~30 分)
python scripts/train_adapter.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --dataset runs/phase3_data_small.pkl \
    --epochs 3 --batch-size 4

# 4. Faithfulness probe
python scripts/baselines.py             --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_faithfulness.py  --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_apc.py           --gonet-ckpt ... --adapter-ckpt ... --dataset ...

# 5. 定性サンプル
python scripts/sample_generations.py \
    --gonet-ckpt ... --adapter-ckpt ... --dataset ... \
    --n-samples 10 --out runs/samples.txt
```

テスト: `pytest tests/ -v` (~200 個、~1 分)。

---

## スタック

Python 3.12 · PyTorch 2.x · HuggingFace Transformers · OpenSpiel (Go env) ·
NumPy · 8 GB 民生用 GPU 1 枚(RTX 3070 Ti / WSL2 Ubuntu 22 で検証済み)。
TinyLlama-1.1B-Chat checkpoint は HuggingFace Hub から取得。

---

## 参考文献

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

## ステータス

ソロプロジェクト。ベースラインとしては完成。
この negative result そのものが結果である。
PR・issue・鋭い批判、歓迎します。

---

*仕事中の妄想を、動くコードに変えてくれた Claude に感謝を。* 🦫
