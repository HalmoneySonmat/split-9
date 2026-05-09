"""Baseline comparison — separate template-memorisation from genuine learning.

Runs three configurations on the same validation split, with the same
data pipeline:

    1. **Output-only**  — adapter completely bypassed (``adapter_tokens=None``).
       The frozen LLM sees only the prompt template; this is the loss
       achievable from template structure alone.
    2. **Random Adapter** — adapter built with random weights, no checkpoint
       loaded. This shows the loss when the LLM sees *some* signal from
       Go-Net but the resampler / cross-attention have not learned to
       interpret it. Should be ≥ Output-only (random noise can hurt).
    3. **Trained Adapter** — checkpoint loaded. The actual model under test.

Interpretation of the gaps:

    Output-only loss  ──────  ceiling for "all template memorisation"
                              ↓ if Trained ≈ Output-only, the model gained
                                nothing from Go-Net activations.
    Random Adapter    ──────  reference for "noise injected"
                              ↓ if Trained ≈ Random, the resampler hasn't
                                learned to extract usable signal.
    Trained Adapter   ──────  what we measured in train_adapter.py

A *meaningful* gap (Trained << Output-only) is the strongest evidence
that the LLM is using Go-Net's activations productively.

Usage:
    python scripts/baselines.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --adapter-ckpt runs/adapter_checkpoints/best.pt \\
        --dataset runs/phase3_data_small.pkl
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from split_brain_go.adapter.projection import AsymmetricPerceiverResampler
from split_brain_go.data.dataset import (
    DEFAULT_PROMPT,
    Phase3Dataset,
    phase3_collate,
    split_train_val,
    tokenize_examples,
)
from split_brain_go.data.generation import load_dataset
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM
from split_brain_go.training.adapter_train import AdapterTrainConfig, adapter_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gonet-ckpt", type=Path, required=True)
    p.add_argument(
        "--gonet-config", choices=["poc", "default"], default="default"
    )
    p.add_argument("--adapter-ckpt", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--llm-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches for a fast sanity run (None = full val).",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional file to write the comparison table to.",
    )
    return p.parse_args()


@torch.no_grad()
def _eval_loop(
    instrumented: InstrumentedLLM,
    gonet: GoNet,
    resampler: AsymmetricPerceiverResampler | None,
    val_loader: DataLoader,
    act_layers: list[int],
    device: torch.device,
    *,
    use_adapter: bool,
    adapter_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
) -> dict[str, float]:
    """One validation pass.

    Args:
        use_adapter: If False, ``adapter_tokens=None`` is passed and the
            resampler is not run. This corresponds to *Output-only*.
            Go-Net is also skipped in that mode (irrelevant).
    """
    instrumented.eval()
    if resampler is not None:
        resampler.eval()
    gonet.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        adapter_tokens = None
        if use_adapter:
            assert resampler is not None
            boards = batch["board"].to(device)
            _, _, acts = gonet.forward_with_acts(boards, layers=act_layers)
            acts = {k: v.to(adapter_dtype) for k, v in acts.items()}
            adapter_tokens = resampler(acts)

        logits = instrumented(
            input_ids=input_ids,
            adapter_tokens=adapter_tokens,
            attention_mask=attention_mask,
        )
        loss = adapter_loss(logits, labels)
        n_real = int((labels[:, 1:] != -100).sum().item())
        total_loss += float(loss.item()) * n_real
        total_tokens += n_real
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan")}
    avg = total_loss / total_tokens
    return {
        "loss": avg,
        "perplexity": float(math.exp(min(avg, 50.0))),
        "n_tokens": total_tokens,
    }


def _format_table(results: dict[str, dict[str, float]]) -> str:
    lines = []
    lines.append("baseline                 loss     perplexity   n_tokens")
    lines.append("-" * 60)
    for name in ("Output-only", "Random Adapter", "Trained Adapter"):
        r = results[name]
        lines.append(
            f"  {name:<22} {r['loss']:.4f}     {r['perplexity']:7.3f}    "
            f"{int(r.get('n_tokens', 0))}"
        )
    lines.append("-" * 60)

    out = results["Output-only"]["loss"]
    rnd = results["Random Adapter"]["loss"]
    trn = results["Trained Adapter"]["loss"]

    gap_out = (out - trn) / out
    gap_rnd = (rnd - trn) / max(rnd, 1e-9)
    lines.append(f"Trained vs Output-only  : Δloss = {trn - out:+.4f}  ({-gap_out:+.2%})")
    lines.append(f"Trained vs Random       : Δloss = {trn - rnd:+.4f}  ({-gap_rnd:+.2%})")
    lines.append("")

    # Heuristic verdict.
    lines.append("INTERPRETATION:")
    if trn >= out - 0.02:
        lines.append(
            "  Trained adapter is ~no better than Output-only.\n"
            "  → Adapter contribution is template-memorisation dominated."
        )
    elif trn >= rnd - 0.02:
        lines.append(
            "  Trained adapter is ~no better than Random Adapter.\n"
            "  → Resampler has not learned to encode useful signal "
            "from Go-Net activations."
        )
    elif (out - trn) / out < 0.10:
        lines.append(
            "  Trained adapter beats Output-only by less than 10%.\n"
            "  → Modest learning signal; most loss comes from template structure."
        )
    elif (out - trn) / out < 0.30:
        lines.append(
            "  Trained adapter beats Output-only by 10–30%.\n"
            "  → Meaningful learning signal, but template still contributes a lot."
        )
    else:
        lines.append(
            "  Trained adapter beats Output-only by more than 30%.\n"
            "  → Strong evidence the adapter is exploiting Go-Net activations."
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if not args.gonet_ckpt.is_file():
        raise FileNotFoundError(f"Go-Net checkpoint not found: {args.gonet_ckpt}")
    if not args.adapter_ckpt.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {args.adapter_ckpt}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------- Go-Net
    cfg_go = (
        GoNetConfig.poc() if args.gonet_config == "poc" else GoNetConfig.default()
    )
    gonet = GoNet(cfg_go).to(device)
    ckpt_go = torch.load(args.gonet_ckpt, map_location=device, weights_only=False)
    gonet.load_state_dict(ckpt_go["model"])
    gonet.eval()
    for p in gonet.parameters():
        p.requires_grad = False
    print(f"Loaded Go-Net from {args.gonet_ckpt} (frozen)")

    # ---------------------------------------------------------------- LLM
    print(f"\nLoading {args.llm_id} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    base_llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id, torch_dtype=torch.bfloat16
    ).to(device)

    # ---------------------------------------------------------------- adapter
    adapter_cfg = AdapterTrainConfig(seed=args.seed)
    layer_channels = {lid: cfg_go.channels for lid in adapter_cfg.act_layers}
    layer_token_counts = dict(
        zip(adapter_cfg.act_layers, adapter_cfg.act_layer_tokens)
    )

    # Build with deterministic random init via the seed (reproducible).
    torch.manual_seed(args.seed)
    resampler = (
        AsymmetricPerceiverResampler(
            layer_token_counts=layer_token_counts,
            layer_channels=layer_channels,
            d_model=base_llm.config.hidden_size,
            n_heads=8,
            n_blocks=2,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    instrumented = InstrumentedLLM(
        base_llm,
        inject_layers=adapter_cfg.inject_layers,
        n_heads=8,
    ).to(device)
    instrumented.adapter_blocks = instrumented.adapter_blocks.to(torch.bfloat16)

    # ---------------------------------------------------------------- data
    print(f"\nLoading dataset from {args.dataset} ...")
    examples = load_dataset(args.dataset)
    tokenized = tokenize_examples(
        examples,
        tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt,
        verbose=False,
    )
    _train, val_data = split_train_val(
        tokenized, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"  val: {len(val_data)} examples")

    val_loader = DataLoader(
        Phase3Dataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=phase3_collate,
    )

    # ---------------------------------------------------------------- run baselines
    results: dict[str, dict[str, float]] = {}

    print("\n[1/3] Output-only (adapter bypassed) ...")
    results["Output-only"] = _eval_loop(
        instrumented, gonet, resampler, val_loader,
        adapter_cfg.act_layers, device,
        use_adapter=False,
        max_batches=args.max_batches,
    )
    print(f"      loss={results['Output-only']['loss']:.4f}  ppl={results['Output-only']['perplexity']:.3f}")

    print("\n[2/3] Random Adapter (random init, no checkpoint) ...")
    results["Random Adapter"] = _eval_loop(
        instrumented, gonet, resampler, val_loader,
        adapter_cfg.act_layers, device,
        use_adapter=True,
        max_batches=args.max_batches,
    )
    print(f"      loss={results['Random Adapter']['loss']:.4f}  ppl={results['Random Adapter']['perplexity']:.3f}")

    # Now load the trained checkpoint into the same modules.
    class _AdapterBundle(torch.nn.Module):
        def __init__(self, resampler, blocks) -> None:
            super().__init__()
            self.resampler = resampler
            self.adapter_blocks = blocks

    bundle = _AdapterBundle(resampler, instrumented.adapter_blocks)
    payload = torch.load(args.adapter_ckpt, map_location=device, weights_only=False)
    bundle.load_state_dict(payload["model"])
    print(f"\nLoaded trained adapter from {args.adapter_ckpt}")

    print("\n[3/3] Trained Adapter ...")
    results["Trained Adapter"] = _eval_loop(
        instrumented, gonet, resampler, val_loader,
        adapter_cfg.act_layers, device,
        use_adapter=True,
        max_batches=args.max_batches,
    )
    print(f"      loss={results['Trained Adapter']['loss']:.4f}  ppl={results['Trained Adapter']['perplexity']:.3f}")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 60)
    table = _format_table(results)
    print(table)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(table + "\n")
        print(f"\nReport saved to {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
