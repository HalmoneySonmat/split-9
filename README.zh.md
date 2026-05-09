# SPLIT-9

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · 🇨🇳 中文 · [🇯🇵 日本語](README.ja.md)

*9×9 Go-Net 上的事后 LLM 适配器忠实性审计。*

> **TL;DR.** 我们用 Flamingo 风格的适配器把冻结的 TinyLlama-1.1B 接到
> 一个冻结的 9×9 Go-Net 上,训练它用自然语言"解释"Go-Net 的着手。
> 验证集 perplexity 从 **16.1 一路降到 1.65**——但三个独立的探针
> 显示,这些收益的约 **95% 来自学到的领域先验**,而不是真正按棋盘
> 传递的信号。在采样的 10 个输出中,模型 **0 / 10** 次给出 actor
> 实际下的坐标,且会向数据集中高频的坐标 (`(2,1)`、`(8,8)`、`pass`)
> 发生 mode collapse。适配器学会的是解释文本的 *语法*,不是 *内容*。

---

## 为什么会做这个

Roger Sperry 和 Michael Gazzaniga 在 60–70 年代的裂脑实验里发现了
一个让人不太舒服的现象。当病人的胼胝体(corpus callosum)被切断后,
如果只把指令显示在他的左视野——也就是只让右脑看到——他会按指令行动,
但语言上完全没有这一动作的意识。当被问到"你为什么这么做"时,掌握语言
的左脑会编出一套连贯的理由,来解释一个它从未见过的决定。它不是在
有意撒谎,而是真心相信自己说的话。Gazzaniga 把这个模块叫做
**左脑通译者(left-hemisphere interpreter)**:流畅、可信、却经常出错。

今天的 AI 系统几乎是在 **故意** 重复这种结构。一个非语言的网络
(视觉编码器、机器人策略、科学模型)旁边接一个 LLM,让 LLM 把前者
正在做的事用自然语言讲出来。RLHF chain-of-thought、视觉-语言助手、
"可解释" 的 RL agent——全都是同一架构的变种。但 LLM 是真的在 *翻译*
上游信号,还是只是在生成统计上像样的文本,这个问题大都没有得到
经验上的回答。

SPLIT-9 是我能想到的、为这个问题搭一个最小可复现 testbed 的尝试:

- 一个 9×9 **Go-Net**(AlphaGo Zero 风格,~5M 参数,完全 white-box)——"右脑"
- 冻结的 **TinyLlama-1.1B**——"左脑"
- 可训练的 **Flamingo 风格适配器**——"corpus callosum"
- 一组 faithfulness 探针,设计目的就是抓 interpreter 在编故事

整套实验在一台 8 GB 消费级 GPU 上一天之内可以从头到尾跑完。所有
组件都在这个仓库里。

---

## 仓库内容

```
SPLIT-9
├── Go-Net                  (Phase 1 训练完后冻结)
│       │ 来自层 {3, 4, 5} 的激活
│       ▼
├── AsymmetricPerceiverResampler        ← 可训练
│       │ 36 个 latent token (每层 8 / 12 / 16)
│       ▼
├── GatedCrossAttention 块 — 注入到 TinyLlama 的层 {10, 16, 18, 20}
│       │
│       ▼
├── TinyLlama-1.1B          (冻结)
│       │ 32k 词表 logit
│       ▼
└── 合成的自然语言 reasoning trace
```

**数据。** 30 局 self-play × 平均 ~75 手 = 2,293 个样本。每个样本是
`(棋盘, 着手, MCTS principal variation, 合成解释文)`。6 套解释模板 ×
词池变化(verb pool、noun pool)。

**训练。** bf16、AdamW、lr 1e-4、batch 4、3 个 epoch,RTX 3070 Ti
约 30 分钟。只有适配器(~404M 参数)在更新;两个底座网络都冻结。

---

## 我们测了什么

四个探针,从便宜到信息量大。

### 1. 基线

| | val loss | val ppl |
|---|---|---|
| Output-only(适配器旁路)             | 2.78 | 16.14 |
| Random-init 适配器(不加载 checkpoint) | 2.78 | 16.14 |
| **训练好的适配器**                    | **0.50** | **1.65** |

![基线 — Output-only / Random / Trained](runs/figures/fig1.png)

训练好的适配器比 Output-only 把 loss 拉低 **−81.9%**。Random vs
Output-only 几乎相同——因为 Flamingo gate 是零初始化的,随机权重下
适配器实际上是个 no-op。(一个值得说的方法学注脚:Random-Adapter
基线只有在把 gate 初始化成非零值时才有意义。)

### 2. Information Ablation Score (IAS)

把 Go-Net 激活张量送入适配器之前,随机将比例 *r* 的通道置零,看 loss
怎么变化。

| r | loss | base 相对 Δloss |
|---|---|---|
| 0.00 | 0.503 | — |
| 0.25 | 0.507 | +0.4 % |
| 0.50 | 0.521 | +3.5 % |
| 0.75 | 0.556 | +10.6 % |
| 1.00 | 0.607 | +20.6 % |

![IAS 扫描 — 通道掩码比例 vs loss](runs/figures/fig2.png)

曲线是 *凸的(convex)*——即使丢一半通道,loss 几乎不动。这是信号
被冗余编码的典型特征,所以 IAS 单独使用是个偏宽容的探针。

### 3. Activation Patching Consistency (APC)

对同样的 prompt + label 做两次 forward。**matched**(正确激活)vs
**mismatched**(在 batch 内做 derangement——适配器看到的激活属于
*另一局* 棋而不是当前解释对应的那一局)。

| | loss |
|---|---|
| matched          | 0.503 |
| mismatched       | 0.594 |
| **APC**          | **+0.181** |

![APC — matched vs mismatched](runs/figures/fig3.png)

3 个 seed,σ ≈ 0.003——非常稳定。mismatched 的惩罚 **比 IAS@0.5 大
5 倍**,因为整盘 swap 绕开了通道冗余问题。APC = 0.18 确认了棋盘相关
信号在因果链上确实流过适配器,但按我们的启发式阈值,这只到 "PARTIAL",
还没到 "FAITHFUL"。

### 4. 定性检查(笑点)

对 10 个随机 val 棋盘做 prompt 的 greedy 解码。

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

![Mode collapse — 输出坐标频度](runs/figures/fig5.png)

三件事很显眼。

1. **OUTPUT-ONLY 是退化的。** 不管输入是什么,它都生成同一段一般文学
   分析文本。也就是说,TinyLlama 自己对围棋一无所知。这彻底排除了
   "模板被记忆"假设,也确认领域工作的 **100%** 是适配器干的。

2. **TRAINED 完美地重现了合成模板的语法:** `[middlegame/21] B (2, 1)
   (p=0.01, v=-0.99, Δv=+0.11) selected`。phase 标签、颜色、坐标
   括号、`p=`、`v=`、`Δv=`、`selected`,全部对齐。

3. **但 TRAINED 在实际着手上几乎全错。** 10 个样本里,**0 个**和
   actor 选的坐标一致。输出坐标反而集中到 `(2,1)`(4 次)、`(8,8)`
   (3 次)、`pass` 等少数高频值——典型的 mode collapse 到数据集
   常见着手。

---

## Loss 分解(头条结论)

![Loss 分解 — 95% 先验 + 5% 棋盘信号](runs/figures/fig4_loss_decomposition.png)

把三个探针放到 raw loss 单位上一起看:

```
Output-only(没有适配器)         loss 2.78    ┐
                                              │ 总下降的 ~95%
Trained, 全部激活置零(mask=1)    loss 0.61    ┘   = 学到的领域先验
                                              ┐
Trained, mismatched (APC)         loss 0.59   │ 总下降的 ~5%
Trained, matched(正常)           loss 0.50   ┘   = 棋盘相关信号
```

适配器的贡献分解为 **棋盘无关的领域先验**(绝对意义上 78% 的 loss
下降,占总改善的 95%) + **棋盘相关的因果信号**(loss 0.09,占
5%)。后者是真实存在并且统计上稳定的,但只够 *改变 mode collapse 到
哪个值*,不够准确到挑出正确的格子。

---

## 这意味着什么

**Cross-entropy 训练理性地把容量投在先验上。** 一段 30 token 的
reasoning trace 大约只有 3 个棋盘相关 token(坐标),其余 ~27 个是结构
token(phase 标签、括号、`selected`、标点)。优化器最小化 loss 的
最佳策略,就是把 27 个完美对齐,把 3 个 mode-collapse 掉。APC 抓得到
那一点点残留的棋盘敏感性,但 cross-entropy 自己根本不会暴露这个鸿沟。

**这是结构上的天花板,不是调参问题。** 把适配器加大、训练时间拉长,
或者在同样模板分布里加更多数据,都只是把先验拟合得更紧,棋盘相关
内容上的精度几乎不动。要把 faithfulness 真的拉上去,得改 *数据分布
本身*——让每个样本里出现更多棋盘相关内容,或者加上 auxiliary loss
显式奖励"输出里出现选定的坐标"。

**而且这是会泛化的。** 同样的 token 熵权重论证适用于大部分事后
适配器 setup:基于合成 caption 的视觉问答、RL agent 自述行为、
只用 next-token loss 评估的 chain-of-thought rationale——全是同一个
坑。APC 这一类 ablation 探针 + 定性检查是 **必要条件**,光看
train/val perplexity 是看不到这个失败的。

---

## 后续——共同发展的孪生网络

事后适配器只能翻译那些 *被数据分布提前安排好、容易翻译* 的信号。
它没法强行要求特定内容。生物的 corpus callosum 没这个问题——两个
半球在同一个发育窗口里一起长大,从一开始学的就是彼此兼容的表征。

自然的下一步:把两个网络 **联合训练**,中间放一个可训练的 bottleneck,
再加一个 auxiliary loss,强制让上游的特定特征(选定的着手、对手上一手、
被提子的位置)以具体 token 的形式出现在语言头的输出里。这模拟的是
生物学上的共同发育,从上面的结构性论证来看,几乎是唯一能打破
mode collapse 天花板的方向。

---

## 怎么复现

```bash
# 0. 环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. 训练 Go-Net (3070 Ti default config 约 3 小时)
python scripts/train_gonet.py --config configs/gonet/default.yaml

# 2. 生成 Phase 3 数据集 (带 MCTS PV 的 self-play)
python scripts/generate_phase3_data.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --n-games 30 --n-simulations 100 \
    --output runs/phase3_data_small.pkl

# 3. 训练适配器 (~30 分钟)
python scripts/train_adapter.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --dataset runs/phase3_data_small.pkl \
    --epochs 3 --batch-size 4

# 4. Faithfulness 探针
python scripts/baselines.py             --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_faithfulness.py  --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_apc.py           --gonet-ckpt ... --adapter-ckpt ... --dataset ...

# 5. 定性采样
python scripts/sample_generations.py \
    --gonet-ckpt ... --adapter-ckpt ... --dataset ... \
    --n-samples 10 --out runs/samples.txt
```

测试: `pytest tests/ -v` (~200 个,~1 分钟)。

---

## 技术栈

Python 3.12 · PyTorch 2.x · HuggingFace Transformers · OpenSpiel (Go env) ·
NumPy · 1 块 8 GB 消费级 GPU(在 RTX 3070 Ti / WSL2 Ubuntu 22 上验证)。
TinyLlama-1.1B-Chat checkpoint 从 HuggingFace Hub 拉取。

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

## 状态

一人项目。作为 baseline 已完结。
这个 negative result 本身就是结果。
欢迎 PR、issue 和尖锐的批评。

---

*感谢 Claude——把上班时的一个胡思乱想,变成了能跑的代码。* 🦫
