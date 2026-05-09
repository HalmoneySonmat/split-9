"""Measure adapter faithfulness via Information Ablation Score (IAS).

Loads a trained adapter checkpoint, sweeps over channel-mask ratios, and
reports per-ratio loss / perplexity on the validation split. The
interpretation of the resulting numbers is documented in
``evaluation_protocol.md`` and ``src/split_brain_go/eval/faithfulness.py``.

Usage:
    python scripts/measure_faithfulness.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --adapter-ckpt runs/adapter_checkpoints/best.pt \\
        --dataset runs/phase3_data_small.pkl

To run a fast sanity sweep (only a few batches per ratio):
    python scripts/measure_faithfulness.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --adapter-ckpt runs/adapter_checkpoints/best.pt \\
        --dataset runs/phase3_data_small.pkl \\
        --max-batches 5 \\
        --n-seeds 1
"""

from __future__ import annotations

import argparse
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
from split_brain_go.eval.faithfulness import format_ias_report, measure_ias
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM
from split_brain_go.training.adapter_train import AdapterTrainConfig


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
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Channel-mask fractions to sweep.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Distinct mask seeds to average per non-trivial ratio.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per (ratio, seed) for speed. None = full val loader.",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional file to write the formatted report to.",
    )
    return p.parse_args()


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

    # Reload adapter checkpoint into the same _AdapterBundle layout used at save.
    class _AdapterBundle(torch.nn.Module):
        def __init__(self, resampler, blocks) -> None:
            super().__init__()
            self.resampler = resampler
            self.adapter_blocks = blocks

    bundle = _AdapterBundle(resampler, instrumented.adapter_blocks)
    payload = torch.load(args.adapter_ckpt, map_location=device, weights_only=False)
    bundle.load_state_dict(payload["model"])
    print(f"Loaded adapter from {args.adapter_ckpt}")

    # ---------------------------------------------------------------- data
    print(f"\nLoading dataset from {args.dataset} ...")
    examples = load_dataset(args.dataset)
    print(f"  {len(examples)} examples loaded")

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

    # ---------------------------------------------------------------- IAS
    print("\n=== Measuring IAS ===")
    print(f"  mask_ratios = {tuple(args.mask_ratios)}")
    print(f"  n_seeds     = {args.n_seeds}")
    print(f"  max_batches = {args.max_batches}\n")

    result = measure_ias(
        instrumented,
        gonet,
        resampler,
        val_loader,
        act_layers=adapter_cfg.act_layers,
        device=device,
        mask_ratios=tuple(args.mask_ratios),
        n_seeds=args.n_seeds,
        adapter_dtype=torch.bfloat16,
        rng_seed=args.seed,
        max_batches=args.max_batches,
    )

    report = format_ias_report(result)
    print(report)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n")
        print(f"\nReport saved to {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
