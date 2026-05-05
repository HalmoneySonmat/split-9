# Split-Brain Go

A research codebase exploring whether an LLM, given direct access to a Go-playing network's internal activations, can produce more *faithful* natural-language explanations of that network's decisions than a standalone LLM can.

Inspired by Gazzaniga's split-brain experiments and the left-hemisphere interpreter hypothesis: we don't aim for "perfect faithfulness" (an arguably ill-defined target for any post-hoc explanation) but for explanations that are *causally connected* to the activations driving the decision.

## Status

Phase 0 — environment setup and pre-flight. See `phase0/` (in the parent directory) for the planning artifacts and `docs/decisions.md` for the architecture decision records.

## Quick start

Requires Linux (or WSL2 on Windows), CUDA 12.1, Python 3.10, and an NVIDIA GPU.

```bash
git clone <this-repo>
cd split_brain_go
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test verifies CUDA availability, loads TinyLlama-1.1B, and confirms OpenSpiel can launch a 9×9 Go game.

## Repository layout

```
split_brain_go/
├── configs/        # Hydra configs (env, gonet, llm, adapter, train, eval)
├── src/split_brain_go/
│   ├── env/        # 9×9 Go environment (OpenSpiel wrapper)
│   ├── gonet/      # AlphaGo-Zero-style policy/value network + MCTS
│   ├── llm/        # Frozen LLM wrapper with adapter injection points
│   ├── adapter/    # Flamingo-style cross-attention adapter (Go-Net → LLM)
│   ├── data/       # Synthetic explanation generation
│   ├── training/   # Self-play and joint training loops
│   └── eval/       # Faithfulness metrics + baselines
├── scripts/        # CLI entry points
├── tests/          # pytest test suite
├── notebooks/      # Exploratory work (not authoritative)
├── docs/           # Architecture, decisions, evaluation protocol
└── runs/           # Checkpoints and logs (gitignored)
```

## Phases

| Phase | Goal | Duration |
|-------|------|----------|
| 0 | Environment, structure, evaluation protocol | 2–3 weeks |
| 1 | Train Go-Net via self-play (9×9) | 1.5–2 months |
| 2 | Design and integrate cross-attention adapter | 1 month |
| 3 | Generate synthetic explanations and train adapter | 2 months |
| 4 | Evaluate faithfulness (activation patching, counterfactual, ablation) | 1.5 months |
| 5 | Write paper | 1.5 months |

Total: ~7–9 months.

## Design principles

- **Frozen base models**. Only the adapter is trained (≈5–10% of parameters).
- **Pre-registered evaluation**. Faithfulness thresholds (APC, CFC, IAS) are fixed before Phase 3 begins.
- **No external explanation data**. Training signals come from self-play and template synthesis only — see `phase0/codebase_study.md` for the licensing rationale.

## License

Apache 2.0. See `LICENSE`.

## Citation

```bibtex
@misc{splitbraingo2026,
  title  = {Split-Brain Interpreter for Go: Joint Training of a Decision Network and a Frozen LLM Explainer},
  author = {namdo},
  year   = {2026},
  note   = {In progress.}
}
```
