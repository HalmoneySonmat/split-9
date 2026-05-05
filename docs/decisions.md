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

## Conventions for future ADRs

- One file (`decisions.md`) until count exceeds ~15, then split into `decisions/` directory.
- Number sequentially. Do not reuse numbers.
- A decision is *Accepted*, *Superseded by ADR-NNN*, or *Deprecated*. Edit the status line; do not delete ADRs.
- Add a new ADR whenever a choice is non-trivially reversible, rather than after-the-fact.
