# Paper Outline — Split-Brain Go: A Faithfulness Audit of Post-hoc Adapters

**Status**: draft v0 — pending user review.

**Tentative title (3 candidates):**
1. *"Mode Collapse Limits Post-hoc Adapters: A Faithfulness Audit on a Go-Playing Network"*
2. *"Post-hoc Adapters Learn Grammar, Not Content: A Faithfulness Study with Synthetic Go Explanations"*
3. *"The Left-Hemisphere Interpreter Doesn't Read: Quantifying Faithfulness in a Frozen Vision-Language Setup"*

(Recommendation: **(2)**. It signals the headline finding and the methodological contribution at once.)

**Estimated length**: 6–8 pages (workshop) or 8–10 pages (arXiv preprint with full appendices). Solo author. No conference deadline assumed — write for arXiv first, retarget later.

---

## Abstract (sketch, ~150 words)

> Inspired by Gazzaniga's left-hemisphere interpreter, we ask whether a frozen language model can be coupled to a frozen domain-specific actor and learn to *faithfully* explain its decisions. We train a Flamingo-style adapter (Perceiver Resampler + gated cross-attention) connecting a 9×9 Go-Net's hidden activations to TinyLlama-1.1B, using synthesized natural-language reasoning traces as supervision. The adapter dramatically reduces validation perplexity from 16.1 (frozen-LLM baseline) to 1.65, and ablation experiments confirm board-specific signal flows through it (APC = 0.18). However, three separate probes — channel masking, board swapping, and qualitative inspection — reveal that the gain is overwhelmingly carried by a learned domain prior rather than per-board content: 0/10 sampled outputs match the actor's actual move, and outputs collapse to dataset-frequent coordinates. We argue this quantifies a fundamental limit of the post-hoc adapter approach and motivates co-developed twin representations as a path forward.

---

## 1. Introduction (~1 page)

**Hook**: Gazzaniga's split-brain experiments — the left hemisphere narrates decisions made by other modules, often confabulating. As LLMs are increasingly attached to non-language systems (vision, robotics, scientific models) the same question becomes empirical: are the natural-language outputs *faithful translations* of the upstream computation, or learned grammar over disconnected priors?

**This paper**: a small, controllable testbed for the question.
* Replace "right hemisphere" with a 9×9 Go-Net (AlphaGo Zero–style, fully white-box, ~5M params).
* Replace "left hemisphere" with a frozen TinyLlama-1.1B.
* Connect them with a trainable Flamingo-style adapter.
* Train on synthesized rationales generated from MCTS principal variations.

**Three contributions:**
1. **Setup**: a minimal, fully reproducible "split-brain" testbed for studying post-hoc adapter faithfulness. Open-source code; runs on a single 8 GB GPU.
2. **Probes**: three complementary faithfulness measurements — Information Ablation Score (IAS), Activation Patching Consistency (APC), and qualitative coordinate accuracy.
3. **Findings**: the adapter learns a strong *domain prior* (perplexity 16.1 → 1.65) but mode-collapses on board-specific content (0/10 coordinate accuracy, APC 0.18). We decompose the loss reduction into prior-attributable (~95%) and board-attributable (~5%) components.

**Implication**: post-hoc adapters yield *fluent* but not *faithful* explanations on this benchmark. We discuss why (data entropy structure, gate initialisation, architectural decoupling) and sketch co-developed twin networks as future work.

---

## 2. Related Work (~½ page)

* **Vision-language adapters**: Flamingo (Alayrac et al. 2022), Perceiver IO (Jaegle et al. 2022), LLaVA, BLIP-2.
* **AlphaGo Zero–style game models**: Silver et al. 2017; KataGo for 9×9 (Wu 2019).
* **Faithfulness probes for LLM rationales**: chain-of-thought faithfulness (Turpin et al. 2023), counterfactual consistency (Atanasova et al. 2023), activation patching (Wang et al. 2023, Conmy et al. 2023).
* **Mechanistic interpretability**: Olsson et al., Nanda et al. — relevant for the spirit of the probes.
* **Cognitive science background**: Gazzaniga's interpreter (Gazzaniga 1989, 2018).

(Keep terse — this is the framing literature, not the contribution.)

---

## 3. Method (~1.5 pages)

### 3.1 Go-Net (the "right hemisphere")
* AlphaGo Zero–style: 6 residual blocks × 64 channels → policy head (82 logits) + value head (scalar).
* Trained via self-play with MCTS, joint policy/value loss.
* Frozen at Phase 3.
* For each move we extract activations from layers {3, 4, 5}.

### 3.2 Synthesized explanations (the supervision)
* For each move, we run MCTS with `n_simulations=200`, extract top-3 principal variations of depth 5.
* Pick one of 6 templates uniformly: `neutral`, `concise`, `narrative`, `pv_lines`, `imagination`, `comparison`.
* Word variation pools (`_VERB_CHOSE`, `_VERB_CONSIDER`, etc.) reduce literal repetition.
* Phase 3 dataset: 30 self-play games × ~75 moves = 2,293 examples.

### 3.3 Adapter ("the corpus callosum")
* `AsymmetricPerceiverResampler`: per-layer query token counts {3:8, 4:12, 5:16}, total 36 latents → d_model=2048.
* Gated cross-attention blocks injected at layers {10, 16, 18, 20} of TinyLlama.
* Gates zero-initialised (standard Flamingo).
* ~404M trainable parameters; base TinyLlama frozen.

### 3.4 Training
* Reasoning prompt: `"Reasoning trace:\n"`. Label-mask everything before the colon.
* `bf16`, AdamW, lr=1e-4, batch=4, 3 epochs, ~30 minutes on RTX 3070 Ti.

---

## 4. Faithfulness Probes (~1 page)

### 4.1 Baselines
* **Output-only**: same prompt, adapter bypassed. Pure frozen-LLM upper bound on loss.
* **Random Adapter**: adapter built but not trained. Bounds the "noise injected" effect (gates zero-init means this collapses to Output-only — a small methodological observation).

### 4.2 Information Ablation Score (IAS)
* For each ratio r ∈ {0, 0.25, 0.5, 0.75, 1}, randomly zero r-fraction of channels in the activation tensors before they reach the adapter.
* **IAS@0.5** = (loss[0.5] − loss[0]) / loss[0].
* **fully-masked Δ** = (loss[1] − loss[0]) / loss[0].

### 4.3 Activation Patching Consistency (APC)
* For each batch, run two forwards: matched (correct activations) and mismatched (deranged within batch).
* **APC** = (mismatched_loss − matched_loss) / matched_loss.
* APC bypasses channel-redundancy issues that limit IAS.

### 4.4 Qualitative coordinate accuracy
* Greedy-decode 80 tokens for 10 sampled validation boards.
* Manual labelling: did the trained model emit the actor's actual `(row, col)` move?

---

## 5. Results (~1.5–2 pages)

### 5.1 Training
* val loss 2.86 → 0.50 (3 epochs). val perplexity 17.5 → 1.65.
* Train/val gap ~0.04 — not pathologically overfit.
* (figure: training curve)

### 5.2 Baselines
| Configuration | val loss | val ppl |
|---|---|---|
| Output-only | 2.78 | 16.14 |
| Random Adapter | 2.78 | 16.14 |
| Trained Adapter | 0.50 | 1.65 |
* Random ≡ Output-only because gates zero-init → adapter contribution = 0 at random init. (Brief note on this.)

### 5.3 IAS sweep
| r | loss | Δloss vs base |
|---|---|---|
| 0.00 | 0.503 | — |
| 0.25 | 0.507 | +0.4% |
| 0.50 | 0.521 | +3.5% |
| 0.75 | 0.556 | +10.6% |
| 1.00 | 0.607 | +20.6% |
* Convex curve — diagnostic for *channel redundancy*: half the channels can be zeroed with negligible loss.
* (figure: IAS sweep curve, log-y or linear)

### 5.4 APC
* matched 0.503 vs mismatched 0.594 → APC = +0.18, very low per-seed variance (σ ≈ 0.003 across 3 seeds).
* APC ≫ IAS@0.5 — full board swapping is a sharper probe than random channel masking.

### 5.5 Loss decomposition
* Output-only loss: 2.78
* Trained, fully-masked: 0.61 → adapter learned a *constant* domain prior (78% of total reduction)
* Trained, mismatched: 0.59 → board-swap penalty
* Trained, matched: 0.50 → board-specific contribution ≈ 0.09 in raw loss (~5% of total reduction)
* (figure: stacked-bar decomposition of loss components)

### 5.6 Qualitative findings (the punchline)
* 0/10 sampled outputs match the actor's actual move coordinate.
* Output-only is degenerate ("The author's use of the word 'suddenly'..." — generic literary analysis) — confirms the adapter is doing *all* the Go-domain work.
* Trained outputs replicate the synthesized template *grammar* exactly: `[endgame/111] B (2,1) (p=0.01, v=-0.99, Δv=+0.11) selected`.
* Trained outputs **mode-collapse** to a small set of frequent coordinates — `(2,1)` appears 4×, `(8,8)` 3×, `pass` repeatedly — across distinct boards.
* (figure / table: side-by-side ground-truth vs trained vs output-only for 5 representative samples)

---

## 6. Discussion (~1 page)

### 6.1 Why mode collapse?
* Cross-entropy averages over tokens. Most tokens are template structure (`selected.`, `Δv=`, punctuation) — high mass, easy to predict from prior alone. Coordinates (`(4,2)`) carry most of the per-board entropy but are only ~3 tokens out of ~30.
* Optimizer rationally invests most capacity in the high-leverage prior; per-board signal doesn't pay off cross-entropy enough to overcome that gradient pressure.
* The data has a structural ceiling on faithfulness — and we've hit it.

### 6.2 The methodological observation about Random Adapter
* Flamingo-style zero-init gates make the Random Adapter baseline degenerate at init. To use it as a meaningful baseline, gates should be initialised non-zero (or warmed up to non-zero before measuring). Reporting either modification is informative.

### 6.3 Generality
* Our setup is small. Bigger LLMs / bigger adapters / more compute might shift the prior/content split, but the structural argument (token entropy is dominated by template) should generalise.
* The same structural ceiling probably applies to many post-hoc adapter setups — RLHF "reasoning explanations", visual question answering with synthetic captions, etc.

### 6.4 Implications for interpretability
* "Faithful natural language explanations" is a strictly harder objective than "fluent natural language explanations". Cross-entropy training measures the latter.
* APC-style ablations are necessary, not sufficient — a model can pass APC and still mode-collapse on the high-information tokens.

---

## 7. Future Work — Co-developed Twin Networks (~½ page)

* The interpretation: post-hoc adapters can only translate signals *that are easy to translate*. They cannot insist on specific content.
* Sketch: train two networks **jointly**, with a corpus-callosum-like bottleneck. Add an auxiliary loss enforcing that the language head's distribution reflect specific upstream features (e.g., chosen move must appear in the explanation as a coordinate token).
* This mirrors biological co-development of left-and-right hemispheric representations and is the natural next experiment.

---

## 8. Conclusion (~⅓ page)

* Post-hoc adapters can dramatically reduce LLM perplexity on synthetic technical text without faithfully translating upstream signal.
* The gap is sharply visible with the right probes (APC + qualitative inspection); cross-entropy alone hides it.
* This study is small but the structural argument (token-entropy weighting) generalises and motivates co-developed approaches.

---

## Figures (must-have)

| # | Figure | Source |
|---|---|---|
| 1 | Architecture diagram (Go-Net + adapter + LLM) | TBD — draw with mermaid or matplotlib |
| 2 | Training curve (val loss / val ppl over epochs) | `runs/adapter_checkpoints/*.json` |
| 3 | IAS sweep curve | `runs/ias_report.txt` data |
| 4 | Loss decomposition stacked bar | combine baselines + IAS + APC |
| 5 | Sample table (ground truth vs trained vs output-only) | `runs/samples.txt` |
| 6 | (optional) Mode-collapse histogram of output coordinates | re-run sample script with N=100 |

## Appendices

* A. Hyperparameters (full)
* B. Synthetic explanation templates (full code listing)
* C. Per-template results (does any template achieve higher coordinate accuracy?)
* D. Random Adapter discussion + non-zero gate init experiment (optional)
* E. Reproducibility checklist

---

## Writing plan (suggested order)

1. **Outline review** ← *we are here*
2. Section 5 (Results) — easiest, just describe the data we already have
3. Section 4 (Probes) — describes our methods
4. Section 3 (Method) — describes the system
5. Section 6 (Discussion) — interpretation
6. Section 1 (Intro) — written *last* to match the actual contributions
7. Section 2 (Related Work)
8. Section 7 (Future Work)
9. Section 8 (Conclusion)
10. Abstract (last)
11. Figures (parallel)
