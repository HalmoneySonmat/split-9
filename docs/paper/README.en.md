# SPLIT-9

*A faithfulness audit of post-hoc LLM adapters, on a 9×9 Go-Net.*

> **TL;DR.** We attached a frozen TinyLlama-1.1B to a frozen 9×9
> Go-Net via a Flamingo-style adapter and trained it to "explain"
> the Go-Net's moves in natural language. Validation perplexity
> dropped dramatically from **16.1 to 1.65** — but three independent
> probes show that ~**95% of that gain is a learned domain prior**,
> not faithful per-board signal. The model emits the actor's exact
> move coordinate in **0 / 10** sampled outputs and mode-collapses to
> dataset-frequent coordinates like `(2,1)`, `(8,8)`, and `pass`.
> The adapter learned the *grammar* of explanations, not their
> *content*.

---

## Why this exists

Roger Sperry and Michael Gazzaniga's split-brain experiments in the
60s–70s found something disturbing. In a patient whose corpus
callosum has been severed, an instruction shown only to the left
visual field — i.e. presented only to the right hemisphere — is
acted on without verbal awareness. When asked *why*, the
language-bearing left hemisphere fabricates a coherent justification
for a decision it never saw. Not consciously lying, but genuinely
believing it. Gazzaniga called this module the
**left-hemisphere interpreter**: fluent, plausible, and frequently
wrong.

Modern AI builds the same architecture on purpose. A non-language
network (vision encoder, robot policy, scientific model) is paired
with an LLM that narrates what the first one is doing. RLHF
chain-of-thought, vision-language assistants, "explainable" RL
agents — they're all variants of the same setup. Whether the LLM is
actually *translating* the upstream signal, or merely producing
fluent text aligned with surface statistics, is mostly an empirical
question and mostly unanswered.

SPLIT-9 is the smallest reproducible testbed I could think of for
that question:

- a 9×9 **Go-Net** (AlphaGo Zero–style, ~5M params, fully white-box)
  as the "right hemisphere",
- a frozen **TinyLlama-1.1B** as the "left hemisphere",
- a trainable **Flamingo-style adapter** as the "corpus callosum",
- and a battery of faithfulness probes designed to catch the
  interpreter in the act of confabulating.

Trains end-to-end in under a day on a single 8 GB consumer GPU.
Every component is in this repo.

---

## What's here

```
SPLIT-9
├── Go-Net                  (trained in Phase 1, then frozen)
│       │ activations from layers {3, 4, 5}
│       ▼
├── AsymmetricPerceiverResampler        ← trainable
│       │ 36 latent tokens (8 / 12 / 16 per layer)
│       ▼
├── GatedCrossAttention blocks at TinyLlama layers {10, 16, 18, 20}
│       │
│       ▼
├── TinyLlama-1.1B          (frozen)
│       │ logits over 32k vocab
│       ▼
└── synthesized natural-language reasoning trace
```

**Data.** 30 self-play games × ~75 moves = 2 293 examples. Each
example has `(board, action, MCTS principal variations, synthesized
explanation)`. Six explanation templates with word-pool variation.

**Training.** bf16, AdamW, lr 1e-4, batch 4, 3 epochs, ~30 min on a
3070 Ti. Only the adapter (~404M params) updates; both base networks
are frozen.

---

## What we measured

Four probes, in order of cost and informativeness.

### 1. Baselines

| | val loss | val ppl |
|---|---|---|
| Output-only (adapter bypassed)         | 2.78 | 16.14 |
| Random-init adapter (no checkpoint)    | 2.78 | 16.14 |
| **Trained adapter**                    | **0.50** | **1.65** |

The trained adapter beats Output-only by **−81.9% loss**. The
trained-vs-random gap is the same because Flamingo gates are
zero-initialized — at random init the adapter is a no-op. (A
methodological observation worth flagging: Random-Adapter is not a
useful baseline unless you initialize gates non-zero.)

### 2. Information Ablation Score (IAS)

We zero out a random fraction *r* of channels in Go-Net's activation
tensors before they reach the adapter, and watch the loss change.

| r | loss | Δloss vs base |
|---|---|---|
| 0.00 | 0.503 | — |
| 0.25 | 0.507 | +0.4 % |
| 0.50 | 0.521 | +3.5 % |
| 0.75 | 0.556 | +10.6 % |
| 1.00 | 0.607 | +20.6 % |

The curve is *convex* — half the channels can be zeroed with almost
no penalty. That's the signature of a redundantly encoded signal,
which makes IAS an inherently lenient probe.

### 3. Activation Patching Consistency (APC)

For each batch we run two forwards on the same prompt + labels.
**Matched** (correct activations) versus **mismatched** (deranged
within the batch — adapter sees a *different* game's activations
than the explanation belongs to).

| | loss |
|---|---|
| matched          | 0.503 |
| mismatched       | 0.594 |
| **APC**          | **+0.181** |

3 seeds, σ ≈ 0.003 — extremely stable. The mismatched penalty is
**5× larger than IAS@0.5**, because full board swapping bypasses the
channel-redundancy issue. APC = 0.18 confirms board-specific signal
is causally flowing — but ranks as "PARTIAL," not "FAITHFUL," on our
heuristic scale.

### 4. Qualitative inspection (the punchline)

Greedy-decoded continuations of the prompt, for 10 random val boards.

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

Three things jump out.

1. **OUTPUT-ONLY is degenerate** — for every input it generates
   identical generic literary analysis. TinyLlama on its own knows
   nothing about Go. This rules out template memorization and
   confirms the adapter is doing 100% of the domain work.

2. **TRAINED reproduces the synthesized template grammar exactly:**
   `[middlegame/21] B (2, 1) (p=0.01, v=-0.99, Δv=+0.11) selected`.
   Phase tag, color, coordinate parens, `p=`, `v=`, `Δv=`, `selected`
   — all perfect.

3. **TRAINED is wrong about the actual move.** Across 10 samples,
   **0** match the actor's chosen coordinate. Output coordinates
   instead concentrate on `(2,1)` (4×), `(8,8)` (3×), `pass`, and a
   handful of others — i.e. mode collapse on dataset-frequent moves.

---

## Loss decomposition (the headline)

Combining the three probes in raw loss units:

```
Output-only (no adapter)         loss 2.78    ┐
                                              │ ~95% of total reduction
Trained, fully-masked (mask=1)   loss 0.61    ┘   = learned domain prior
                                              ┐
Trained, mismatched (APC)        loss 0.59    │ ~5% of total reduction
Trained, matched (baseline)      loss 0.50    ┘   = per-board signal
```

The adapter's contribution decomposes into a **board-independent
domain prior** (~78% absolute loss reduction, 95% of the total
improvement) and a much smaller **per-board causal signal** (0.09 in
loss, ~5%). The latter is real and statistically robust, but only
shifts *which mode* the model collapses to — it isn't precise
enough to pick the right square.

---

## What this means

**Cross-entropy training rationally invests in the prior.** A
30-token reasoning trace contains ~3 board-specific tokens (the
coordinate) and ~27 structural tokens (phase tag, parens, `selected`,
punctuation). The optimizer minimizes loss by perfecting the 27 and
mode-collapsing on the 3. APC catches the small residual board
sensitivity; cross-entropy on its own surfaces nothing of this gap.

**This is a structural ceiling, not a tuning problem.** Larger
adapters, longer training, or more data inside the same template
distribution will improve prior fit without much helping per-board
content. Pushing faithfulness up requires changing the *data
distribution itself* — richer board-tied content per example, or
auxiliary losses that explicitly reward including the chosen
coordinate.

**This generalises.** The same token-entropy-weighting argument
applies to most post-hoc adapter setups: visual question answering
on synthetic captions, RL agents narrating their own actions,
chain-of-thought rationales scored only on next-token loss. APC-style
probes plus qualitative inspection are necessary; train/val
perplexity alone hides the failure.

---

## Future work — co-developed twin networks

A post-hoc adapter can only translate signals that the data
distribution already makes *easy to translate*. It can't insist on
specific content. Biological corpus callosum doesn't have that
problem — the two hemispheres co-developed during the same
developmental window and learned compatible representations from the
start.

The natural follow-up: train two networks **jointly**, with a
trainable bottleneck between them, and add an auxiliary loss that
makes specific upstream features (chosen move, opponent's last move,
captures) appear as concrete tokens in the language head's output.
This mirrors biological co-development and is, on the structural
argument above, the only direction likely to break the mode-collapse
ceiling.

---

## Reproducing this

```bash
# 0. setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. train Go-Net (~3 hours on 3070 Ti, default config)
python scripts/train_gonet.py --config configs/gonet/default.yaml

# 2. generate Phase 3 dataset (self-play with MCTS PVs)
python scripts/generate_phase3_data.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --n-games 30 --n-simulations 100 \
    --output runs/phase3_data_small.pkl

# 3. train adapter (~30 min)
python scripts/train_adapter.py \
    --gonet-ckpt runs/checkpoints/best.pt \
    --dataset runs/phase3_data_small.pkl \
    --epochs 3 --batch-size 4

# 4. faithfulness probes
python scripts/baselines.py             --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_faithfulness.py  --gonet-ckpt ... --adapter-ckpt ... --dataset ...
python scripts/measure_apc.py           --gonet-ckpt ... --adapter-ckpt ... --dataset ...

# 5. qualitative samples
python scripts/sample_generations.py \
    --gonet-ckpt ... --adapter-ckpt ... --dataset ... \
    --n-samples 10 --out runs/samples.txt
```

Tests: `pytest tests/ -v` (~200 tests, ~1 min).

---

## Stack

Python 3.12 · PyTorch 2.x · HuggingFace Transformers · OpenSpiel (Go env) ·
NumPy · single 8 GB consumer GPU (tested on RTX 3070 Ti, WSL2 Ubuntu 22).
TinyLlama-1.1B-Chat checkpoint pulled from HuggingFace Hub.

---

## References

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

## Status

Solo project. Complete as a baseline.
The negative result *is* the result.
PRs · issues · pointed criticism welcome.

---

*With thanks to Claude — for turning a daydream-at-work into running code.* 🦫
