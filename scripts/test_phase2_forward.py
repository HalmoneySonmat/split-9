"""Phase 2 acceptance gate — integrated Go-Net + Adapter + LLM forward.

Loads the real TinyLlama, a (possibly trained) Go-Net, and runs a single
end-to-end forward through the Perceiver Resampler and the gated
cross-attention adapter blocks. Reports peak VRAM, wall time, and the
shapes of every intermediate tensor.

Run *after* Phase 1.3b training has finished (so you can pass
``--gonet-ckpt runs/checkpoints/best.pt``). The gates are zero-init,
so the first forward should produce *the same logits* as the bare LLM
with no adapter — that's the property we're verifying.

Usage:
    python scripts/test_phase2_forward.py
    python scripts/test_phase2_forward.py --gonet-ckpt runs/checkpoints/best.pt
    python scripts/test_phase2_forward.py --inject-layers 6 12 18 --act-layers 1 3 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from split_brain_go.adapter.projection import PerceiverResampler
from split_brain_go.env.go_env import GoEnv
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--llm-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model id of the base LLM.",
    )
    p.add_argument(
        "--gonet-ckpt",
        type=Path,
        default=None,
        help="Path to a Go-Net checkpoint .pt. If unset, uses random weights.",
    )
    p.add_argument(
        "--gonet-config",
        choices=["poc", "default"],
        default="default",
        help="GoNetConfig variant.",
    )
    p.add_argument(
        "--act-layers",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Which Go-Net residual blocks to feed into the resampler.",
    )
    p.add_argument(
        "--inject-layers",
        type=int,
        nargs="+",
        default=[6, 12, 18],
        help="Which TinyLlama layers receive a cross-attention adapter.",
    )
    p.add_argument(
        "--n-latents",
        type=int,
        default=32,
        help="Output token count of the Perceiver Resampler.",
    )
    p.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Cross-attention heads.",
    )
    p.add_argument(
        "--prompt",
        default="Explain in one sentence why this Go move makes sense.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------- Go-Net
    gonet_cfg = (
        GoNetConfig.poc() if args.gonet_config == "poc" else GoNetConfig.default()
    )
    print(f"\nGo-Net config: {gonet_cfg}")
    gonet = GoNet(gonet_cfg).to(device)
    if args.gonet_ckpt is not None:
        ckpt = torch.load(args.gonet_ckpt, map_location=device, weights_only=False)
        gonet.load_state_dict(ckpt["model"])
        print(f"Loaded Go-Net from {args.gonet_ckpt}")
    else:
        print("(Go-Net weights are random — pass --gonet-ckpt for trained weights)")
    gonet.eval()

    # ---------------------------------------------------------------- LLM
    print(f"\nLoading {args.llm_id} ...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    base_llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id, torch_dtype=torch.bfloat16
    ).to(device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  hidden_size = {base_llm.config.hidden_size}")
    print(f"  n_layers    = {len(base_llm.model.layers)}")

    # ---------------------------------------------------------------- adapter
    layer_channels = {layer_id: gonet_cfg.channels for layer_id in args.act_layers}
    print(
        f"\nResampler: {len(layer_channels)} layers × {gonet_cfg.channels}ch "
        f"→ {args.n_latents} tokens × {base_llm.config.hidden_size}d"
    )
    resampler = (
        PerceiverResampler(
            layer_channels=layer_channels,
            n_latents=args.n_latents,
            d_model=base_llm.config.hidden_size,
            n_heads=args.n_heads,
            n_blocks=2,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    print(f"Instrumenting LLM at layers {args.inject_layers}")
    instrumented = InstrumentedLLM(
        base_llm,
        inject_layers=args.inject_layers,
        n_heads=args.n_heads,
    ).to(device)
    # InstrumentedLLM keeps the base in bf16 (already moved); the adapter
    # blocks default to fp32 — cast them to bf16 to match.
    instrumented.adapter_blocks = instrumented.adapter_blocks.to(torch.bfloat16)

    n_train = sum(p.numel() for p in resampler.parameters()) + sum(
        p.numel() for p in instrumented.adapter_blocks.parameters()
    )
    n_frozen = sum(p.numel() for p in instrumented.base.parameters())
    print(
        f"\nParameters:\n"
        f"  trainable (adapter + resampler): {n_train / 1e6:.1f} M\n"
        f"  frozen    (LLM)                : {n_frozen / 1e9:.2f} B"
    )

    # ---------------------------------------------------------------- forward
    print("\n--- Forward pass ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Build a non-trivial board state.
    env = GoEnv()
    env.reset()
    for action in (40, 20, 50, 30, 10):
        env.step(action)
    board = env.encode().unsqueeze(0).to(device)

    # 1. Go-Net forward with activation capture.
    t = time.time()
    gonet_in_fp32 = board.float()
    with torch.no_grad():
        policy_logits, value, acts = gonet.forward_with_acts(
            gonet_in_fp32, layers=args.act_layers
        )
    t_gonet = time.time() - t
    print(
        f"  Go-Net : policy {tuple(policy_logits.shape)} "
        f"value {value.item():+.3f}  ({t_gonet * 1e3:.1f} ms)"
    )

    # Cast activations to bf16 to match the rest of the pipeline.
    acts_bf16 = {k: v.to(torch.bfloat16) for k, v in acts.items()}

    # 2. Resampler.
    t = time.time()
    with torch.no_grad():
        adapter_tokens = resampler(acts_bf16)
    t_res = time.time() - t
    print(
        f"  Resampler tokens: {tuple(adapter_tokens.shape)} "
        f"dtype={adapter_tokens.dtype}  ({t_res * 1e3:.1f} ms)"
    )

    # 3. Tokenize prompt.
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"  Prompt   : {tuple(input_ids.shape)}  '{args.prompt}'")

    # 4. Instrumented LLM forward.
    t = time.time()
    with torch.no_grad():
        logits = instrumented(input_ids, adapter_tokens=adapter_tokens)
    t_llm = time.time() - t
    print(f"  LLM logits: {tuple(logits.shape)}  ({t_llm * 1e3:.1f} ms)")

    # ---------------------------------------------------------------- checks
    print("\n--- Sanity checks ---")
    assert torch.isfinite(logits).all().item(), "logits contain NaN/Inf"
    print("  ✓ logits all finite")
    assert logits.shape[0] == input_ids.shape[0]
    assert logits.shape[2] == base_llm.config.vocab_size
    print("  ✓ logits shape matches (B, T, vocab)")

    # Compare to bare LLM (no adapter): with zero-init gates, must match.
    with torch.no_grad():
        bare_logits = instrumented(input_ids, adapter_tokens=None)
    delta = (logits - bare_logits).abs().max().item()
    print(f"  ✓ |logits − bare logits|_max = {delta:.2e} (zero gates → identity)")
    assert delta < 1e-3, "Zero-init adapter is altering output beyond rounding"

    # ---------------------------------------------------------------- profile
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak VRAM: {peak_gb:.2f} GB / 8.00 GB")

    total = t_gonet + t_res + t_llm
    print(f"Total forward: {total * 1e3:.1f} ms")
    print("\n✓ Phase 2 forward integration test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
