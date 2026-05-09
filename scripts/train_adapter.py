"""Train the Phase 3 adapter (Resampler + cross-attn blocks).

Frozen: Go-Net (trained in Phase 1) + TinyLlama (pretrained).
Trainable: AsymmetricPerceiverResampler + GatedCrossAttentionBlocks.

Usage:
    python scripts/train_adapter.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --dataset runs/phase3_data.pkl \\
        --epochs 3

PoC small run (≈ 1 hour on RTX 3070 Ti, depending on dataset size):
    python scripts/train_adapter.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --dataset runs/phase3_data_small.pkl \\
        --epochs 1 \\
        --batch-size 2
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
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM
from split_brain_go.training.adapter_train import AdapterTrainConfig, train_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gonet-ckpt", type=Path, required=True)
    p.add_argument(
        "--gonet-config", choices=["poc", "default"], default="default"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pickled list of Phase3Example (from generate_phase3_data.py).",
    )
    p.add_argument("--llm-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/adapter_checkpoints"),
    )
    p.add_argument(
        "--log-run-name",
        default=None,
        help="If unset, auto-generated as phase3-adapter-<date>.",
    )
    p.add_argument(
        "--log-train-steps",
        action="store_true",
        help="Also log per-step train metrics (high volume).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.gonet_ckpt.is_file():
        raise FileNotFoundError(f"Go-Net checkpoint not found: {args.gonet_ckpt}")
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
    ckpt = torch.load(args.gonet_ckpt, map_location=device, weights_only=False)
    gonet.load_state_dict(ckpt["model"])
    gonet.eval()
    for p in gonet.parameters():
        p.requires_grad = False
    print(f"\nLoaded Go-Net from {args.gonet_ckpt} (frozen)")

    # ---------------------------------------------------------------- LLM
    print(f"\nLoading {args.llm_id} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    base_llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id, torch_dtype=torch.bfloat16
    ).to(device)
    print(f"  hidden_size = {base_llm.config.hidden_size}")
    print(f"  n_layers    = {len(base_llm.model.layers)}")

    # ---------------------------------------------------------------- adapter
    adapter_cfg = AdapterTrainConfig(
        seed=args.seed,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=str(args.checkpoint_dir),
        log_run_name=args.log_run_name,
        log_train_steps=args.log_train_steps,
    )

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
    print(
        f"\nResampler: layers {adapter_cfg.act_layers} → "
        f"tokens {adapter_cfg.act_layer_tokens} (total {sum(adapter_cfg.act_layer_tokens)})"
    )

    instrumented = InstrumentedLLM(
        base_llm,
        inject_layers=adapter_cfg.inject_layers,
        n_heads=8,
    ).to(device)
    instrumented.adapter_blocks = instrumented.adapter_blocks.to(torch.bfloat16)
    print(f"Inject layers: {adapter_cfg.inject_layers}")

    # ---------------------------------------------------------------- data
    print(f"\nLoading dataset from {args.dataset} ...")
    examples = load_dataset(args.dataset)
    print(f"  {len(examples)} examples loaded")

    print(f"Tokenizing (max_length={args.max_length}, prompt={args.prompt!r}) ...")
    tokenized = tokenize_examples(
        examples,
        tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt,
        verbose=True,
    )

    train_data, val_data = split_train_val(
        tokenized, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"  train: {len(train_data)} | val: {len(val_data)}")

    train_loader = DataLoader(
        Phase3Dataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=phase3_collate,
    )
    val_loader = DataLoader(
        Phase3Dataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=phase3_collate,
    )

    # ---------------------------------------------------------------- train
    print("\n=== Training ===\n")
    final = train_adapter(
        instrumented,
        gonet,
        resampler,
        train_loader,
        val_loader,
        adapter_cfg,
        device=device,
    )

    print("\n=== Done ===")
    print(f"Final train loss: {final.get('train_loss', float('nan')):.4f}")
    print(f"Final val loss  : {final.get('val_loss', float('nan')):.4f}")
    print(f"Final val ppl   : {final.get('val_perplexity', float('nan')):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
