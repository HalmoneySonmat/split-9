# Architecture Decision Records

Each ADR captures one significant decision: the context, the choice, the reasoning, and the consequences. We do not delete ADRs; we *supersede* them when a later decision changes course. This keeps a history of why the system looks the way it does.

Format follows Michael Nygard's lightweight ADR template.

---

## ADR-001 — Board size 9×9

**Status**: Accepted, 2026-05-05.

**Context**.
9×9, 13×13, and 19×19 are the standard Go board sizes. Larger boards exhibit richer strategy but require dramatically more self-play to converge. The pre-Phase-0 plan needed a single size fixed before any code was written.

**Decision**.
9×9.

**Reasoning**.
- 7×7 is too tactically thin: most games end on captures alone, leaving no room for "shape" or "territory" patterns that the LLM is supposed to articulate.
- 19×19 is research-grade and would dominate the compute budget on a single RTX 30/40 GPU.
- 9×9 is the minimum board on which professional Go shows meaningful strategic patterns, and academic baselines exist (e.g. KataGo's 9×9 results).

**Consequences**.
- All Phase 4 evaluations are on 9×9. Generalization to larger boards is explicitly out of scope.
- Self-play game length averages ~80 moves, which is tractable for a single-GPU training loop.

---

## ADR-002 — Initial LLM = TinyLlama-1.1B-Chat-v1.0

**Status**: Accepted (provisional), 2026-05-05.

**Context**.
The "Explainer" half of the split-brain architecture needs to be a frozen, pre-trained LLM. Candidates considered: GPT-2 medium (355M), Pythia-1B, TinyLlama-1.1B, Polyglot-Ko-1.3B, Llama-2-7B.

**Decision**.
TinyLlama-1.1B-Chat-v1.0 for Phase 1–3. Re-evaluation point at Phase 2.1.

**Reasoning**.
- Modern Llama architecture (RoPE, RMSNorm, GQA-compatible) → contemporary tooling and adapter recipes.
- 1.1B fits comfortably in 12 GB VRAM at bf16 alongside a small Go-Net + adapter.
- Chat-tuned variant gives reasonable instruction-following from the start; we want focus on the *adapter* learning, not on teaching the LLM English.
- 7B+ models would force aggressive quantization or model-parallel tricks that would distort the research signal.

**Consequences**.
- All synthetic explanations in Phase 3 are in English (see ADR-003).
- If TinyLlama proves too weak in Phase 2.4 forward-pass tests, we fall back to Pythia-1B or step up to Llama-3-3B with LoRA.

**Addendum (2026-05-06)**.
Actual hardware identified: RTX 3070 Ti, 8 GB VRAM (lower bound of the 8–24 GB range we initially scoped). Implications for Phase 2–3 training:
- TinyLlama in bf16 occupies ~2.2 GB. Adapter + Go-Net + activations + optimizer state → realistic budget ~6–7 GB for joint training.
- Training-time mitigations to apply by default: `gradient_checkpointing=True`, batch size ≤ 4 with gradient accumulation, 8-bit Adam (`bitsandbytes`).
- If still OOM after those, drop to Pythia-410M or load TinyLlama in 4-bit (QLoRA) with adapter on top.
- Inference (Phase 4 evaluation) is comfortable at 8 GB — only training is tight.

---

## ADR-003 — Output language = English

**Status**: Accepted, 2026-05-05.

**Context**.
The user's first language is Korean. The pre-trained LLM ecosystem and the relevant academic literature are predominantly English-centric.

**Decision**.
LLM-generated explanations are in English throughout Phase 1–4. Korean translation, if needed for presentation, happens at the demo stage only.

**Reasoning**.
- Wider model selection (TinyLlama, Pythia, Llama, etc. are English-first).
- Synthetic templates and evaluation infrastructure (sentence-BERT, NLI judges, LM-eval harnesses) are mature in English.
- A Korean variant would require redesigning the synthetic-template generator, the human-eval rubric, and would constrain LLM choice to Polyglot-Ko or similar.

**Consequences**.
- Human evaluators must be Go-literate **and** comfortable reading English. Recruitment scope narrowed.
- A Korean follow-up is a clean future-work item, not a requirement.

---

## ADR-004 — Development environment = WSL2 + Ubuntu 22.04

**Status**: Accepted, 2026-05-05.

**Context**.
The user runs Windows 11 with an RTX 30/40 GPU. The core libraries (OpenSpiel, TransformerLens, bitsandbytes) are most stable on Linux.

**Decision**.
WSL2 with Ubuntu 22.04 inside Windows. Native Linux dual-boot is acceptable but not required.

**Reasoning**.
- WSL2 supports CUDA passthrough natively since Windows 11; the host NVIDIA driver is reused with no separate Linux driver install.
- OpenSpiel's Windows-native build path is fragile; the wheel + WSL2 path is the project's one-line install.
- Standard Linux tooling (bash, apt, conda) makes the environment-setup doc reproducible.

**Consequences**.
- The repo is developed against Linux paths. Windows-native users (without WSL2) will hit `open_spiel` import errors and need to re-implement the Go environment, which Phase 1.1 lists as the documented fallback.
- Phase 0's `environment_setup.md` assumes WSL2 throughout.

---

## ADR-005 — Go environment = OpenSpiel

**Status**: Accepted, 2026-05-05.

**Context**.
We need a 9×9 Go simulator with legal-move enumeration, ko detection, suicide-rule handling, and a final-territory scorer. Options: roll our own, use OpenSpiel, use one of the AlphaZero re-implementation libraries (e.g. `alpha-zero-general`).

**Decision**.
OpenSpiel (`pyspiel.load_game("go(board_size=9)")`).

**Reasoning**.
- DeepMind-maintained, used as a reference in dozens of research projects → low risk of subtle rule bugs.
- Phase-0 sandbox verification confirmed the wheel installs cleanly and exposes `legal_actions`, `apply_action`, `is_terminal`, `returns`, and an observation tensor.
- Saves ~1 week of Phase 1.1 work.

**Consequences**.
- We're tied to OpenSpiel's observation tensor shape, which is `[4, 9, 9]`. This is fewer channels than AlphaGo Zero's input format. Phase 1.2 will design the Go-Net input around this — likely augmenting with extra channels (turn indicator, move number, recent-move history) ourselves rather than relying on OpenSpiel's defaults.
- If a future phase needs richer perception (e.g. liberty counts as input), we layer it in `src/split_brain_go/env/encoding.py` rather than modifying OpenSpiel.

---

## ADR-006 — Adapter = Flamingo-style gated cross-attention

**Status**: Accepted, 2026-05-05.

**Context**.
Three architectural families connect a non-text encoder to a frozen LLM: (1) prepend visual tokens to the LLM input (LLaVA-style), (2) cross-attention layers interleaved with self-attention (Flamingo-style), (3) text-converted features (caption-prefix). We need to pick one for the Go-Net → LLM bridge.

**Decision**.
Flamingo-style. Specifically, a Perceiver-Resampler that compresses Go-Net activations into a fixed N tokens, then `GatedCrossAttentionBlock`s injected at a subset of LLM layers.

**Reasoning**.
- The LLM is frozen; option (2) is the standard recipe for that constraint.
- Tanh-gating with zero initialization means the adapter starts as the identity function — the LLM's pre-trained behavior is preserved at step 0 and recovered automatically when the gate is closed during Phase 4 ablation studies.
- Reference implementation (`lucidrains/flamingo-pytorch`, MIT) gives us a clean starting point.

**Consequences**.
- Adapter parameter count is ~5–10% of LLM parameters. Training cost stays modest.
- Activation patching (Phase 4.1) becomes a clean experiment because the gates explicitly mediate Go-Net → LLM information flow.
- Option (3) text-converted features is implemented as Baseline 4 (`evaluation_protocol.md` §D.4) for a fair comparison.

---

## ADR-007 — Training data = self-play + synthetic templates only

**Status**: Accepted, 2026-05-05.

**Context**.
The most direct way to produce a strong Go explainer is to train on professional commentary text. That data exists (e.g. WeiqiTV transcripts, professional commentary books). We choose not to use it.

**Decision**.
All Phase 3 training signals come from (a) self-play game records and (b) automatically-derived synthetic templates over objective game features (policy entropy, value change, territory delta, capture count, etc.). No external human commentary.

**Reasoning**.
- **Methodological**: the research question is whether *activation access* improves faithfulness. Mixing in human commentary introduces a confound — a model could appear faithful by memorizing commentary patterns rather than reflecting Go-Net's internals.
- **Reproducibility**: the synthetic pipeline is deterministic given a Go-Net checkpoint, so other researchers can reproduce the training set without scraping commentary.
- **Licensing**: external commentary corpora carry uncertain license status; staying synthetic sidesteps the question.

**Consequences**.
- Phase 3 is the highest-risk phase of the project. If synthetic templates are too monotone, the LLM will fail to articulate strategic concepts beyond the templates' vocabulary.
- Phase 3.5 lists the explicit fallback: small-scale weakly-supervised LM auxiliary loss on neutral text (not commentary) if synthetic-only proves catastrophic.
- The contribution claim narrows from "a strong Go explainer" to "a faithful Go explainer", which we believe is the more interesting result anyway.

---

## ADR-008 — GoEnv input channels = 8

**Status**: Accepted, 2026-05-06.

**Context**.
OpenSpiel's `observation_tensor_shape` is `[4, 9, 9]`: black stones, white stones, empty cells, all-ones. AlphaGo Zero used 17 channels (8 own + 8 opp history + turn). We need a middle ground that gives GoNet enough perceptual signal without inflating model size on a small board.

**Decision**.
8 input channels, constructed in `env/encoding.py` (not via OpenSpiel's default tensor):

| ch | Meaning |
|----|---------|
| 0  | Own stones (binary) |
| 1  | Opponent stones (binary) |
| 2  | Last move location (one-hot) |
| 3  | 2nd-most-recent move (one-hot) |
| 4  | 3rd-most-recent move (one-hot) |
| 5  | 4th-most-recent move (one-hot) |
| 6  | Turn plane (all 1 if Black to move, else all 0) |
| 7  | Legal-move mask (1 where legal, exc. pass) |

**Reasoning**.
Own/opp split is canonical for two-player perspective games. Four-step history gives GoNet enough recency signal for ko detection and tactical patterns without a 17-channel parameter inflation. Explicit legal mask saves the model from learning rule constraints that the env already knows.

**Consequences**.
GoNet input shape is fixed at `(B, 8, 9, 9)`. Pass actions (action 81) do not contribute to history channels — those frames are zero. Channels 7 (legal mask) is informative but redundant with policy-head softmax masking; we expose it to give the model an inductive bias.

---

## ADR-009 — Residual block depth: PoC 4, full 6

**Status**: Accepted, 2026-05-06.

**Context**.
Plan range is 3–6 residual blocks. PoC and full-training stages have different needs: PoC values fast iteration, full values capacity.

**Decision**.
PoC: 4 residual blocks. Full Phase 1.3b training: 6 residual blocks.

**Reasoning**.
4 blocks gives forward pass < 5 ms on RTX 3070 Ti even unoptimized — fast enough to debug self-play. 6 blocks is the upper bound for keeping parameter count under ~3M with 128 channels.

**Consequences**.
Hyperparam swap between PoC and full requires retraining (no warm-start). Acceptable since PoC is exploratory.

---

## ADR-010 — Channel width: PoC 64, full 128

**Status**: Accepted, 2026-05-06.

**Context**.
Plan range is 64–128 channels.

**Decision**.
PoC: 64. Full: 128.

**Reasoning**.
With 4 res blocks × 64 ch ≈ 0.5M parameters, PoC forward is sub-millisecond on GPU. Full (6 × 128) ≈ 2.5M params, still well within 8 GB VRAM even with batched MCTS. Avoids over-sizing for a 9×9 board.

---

## ADR-011 — MCTS variant: PUCT (AlphaGo Zero)

**Status**: Accepted, 2026-05-06.

**Context**.
Original plan says only "MCTS". Variants include UCT (UCB1), PUCT (AlphaGo Zero), MuZero-style.

**Decision**.
PUCT formula:
```
U(s,a) = c_puct · P(s,a) · √(N(s)) / (1 + N(s,a))
selected_action = argmax_a [Q(s,a) + U(s,a)]
```
with `c_puct = 1.5`, Dirichlet noise α=0.25 added to root prior with weight 0.25.

**Reasoning**.
PUCT is the standard for neural-net-guided MCTS and is what every AlphaGo-Zero re-implementation uses. c_puct=1.5 is the AlphaZero paper's value; α=0.25 is standard for boards under 19×19.

**Consequences**.
MCTS implementation must accept policy prior (P) from GoNet, not just count-based UCB.

---

## ADR-012 — MCTS simulations: PoC 100, full 200

**Status**: Accepted, 2026-05-06.

**Context**.
Plan range is 100–400.

**Decision**.
PoC self-play: 100 sims/move. Full training: 200. Evaluation (Phase 1.4): 400 (more careful play).

**Reasoning**.
200 is the AlphaGo Zero 9×9 reference. 400 at eval gives stronger play for fairer benchmark. PoC at 100 keeps iteration cycles fast.

---

## ADR-013 — Self-play data persistence: SGF + dict

**Status**: Accepted, 2026-05-06.

**Context**.
Self-play games must be reusable across debug runs and replay-buffer warmup. Format choices: SGF (human-readable, standard), pickle (fastest), HDF5 (structured), custom binary.

**Decision**.
Store each game in two parallel files:
- `game_{N}.sgf` — moves only, human-readable, viewable in any Go client.
- `game_{N}.pkl` — additional training data (MCTS visit distributions, value targets) as pickled dict.

**Reasoning**.
SGF preserves the move record in a format any researcher can review (and that we can show in the paper). Pickle handles MCTS distributions efficiently. Storing both is cheap (a 9×9 game is < 5 KB in either format).

**Consequences**.
Replay buffer loader needs both files paired by game number. We use a `runs/games/` directory; disk overflow at 100 GB triggers oldest-first deletion.

---

## ADR-014 — Self-play / training schedule: alternating

**Status**: Accepted, 2026-05-06.

**Context**.
Two extremes: (a) interleaved per-step (each self-play move triggers a training step) — complex sync. (b) Pure alternating — generate a fixed batch of self-play games, then train, then repeat.

**Decision**.
Alternating: generate **N=500 self-play games** with the current model, then train **for 1 epoch over the replay buffer**, then evaluate, then repeat.

**Reasoning**.
Simpler synchronization on a single GPU. 500 games × ~80 moves = ~40k samples added per cycle, which keeps the replay buffer (50 万 limit) fresh while letting old games still contribute. Cycle time on RTX 3070 Ti expected at 1–2 hours, giving ~12 cycles/day.

**Consequences**.
Total training run = roughly 200 cycles ≈ 2–3 weeks of wall-clock at 16 hours/day GPU usage. Fits Phase 1.3b's 4–6-week budget with room.

---

## ADR-015 — DP1 thresholds: 95% vs Random, 70% vs Greedy

**Status**: Accepted, 2026-05-06. **Supersedes** the original plan's 80% vs Greedy.

**Context**.
The original plan set DP1 thresholds at vs Random 95%, vs Greedy 80%. The 80% bar against the *same model's* greedy policy is unusually high — well-trained policy networks often play near-optimally at argmax, leaving MCTS little room. Setting an unreachable bar would block Phase 2 entry on a metric whose threshold has no published precedent.

**Decision**.
- vs Random: ≥ 95% (unchanged).
- vs Greedy (own model's argmax, no MCTS): ≥ 70%.
- Self-play vs prior checkpoint: ELO improvement monotonic across at least 5 consecutive cycles.

**Reasoning**.
70% gives MCTS measurable lift over greedy without demanding that MCTS be dominant. Empirically, AlphaGo Zero papers report MCTS-vs-greedy gaps of 60–75% at convergence on 9×9.

**Consequences**.
Phase 1 entry to Phase 2 is gated on the new thresholds. If even 70% is missed, plan_review's escalation path (more sims, larger model) applies before relaxing further.

---

## Conventions for future ADRs

- One file (`decisions.md`) until count exceeds ~15, then split into `decisions/` directory.
- Number sequentially. Do not reuse numbers.
- A decision is *Accepted*, *Superseded by ADR-NNN*, or *Deprecated*. Edit the status line; do not delete ADRs.
- Add a new ADR whenever a choice is non-trivially reversible, rather than after-the-fact.
