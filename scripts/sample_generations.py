"""Qualitative samples — generate model outputs for inspection.

Picks N validation examples and, for each, prints:

    [board id, move number, action]
    GROUND TRUTH : the synthesized target explanation.
    TRAINED      : greedy-decoded continuation of the prompt with the
                   trained adapter active.
    OUTPUT-ONLY  : greedy-decoded continuation with the adapter
                   bypassed (adapter_tokens=None) — frozen LLM only.

Eyeballing this side-by-side answers questions the metrics can't:
    * Does the trained model actually emit board-specific tokens
      (move coordinates, PV moves) or only generic Go vocabulary?
    * Is the per-board signal we measured with APC manifesting in
      recognisable language, or just in low-entropy structural tokens?

Usage:
    python scripts/sample_generations.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --adapter-ckpt runs/adapter_checkpoints/best.pt \\
        --dataset runs/phase3_data_small.pkl \\
        --n-samples 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from split_brain_go.adapter.projection import AsymmetricPerceiverResampler
from split_brain_go.data.dataset import DEFAULT_PROMPT
from split_brain_go.data.generation import load_dataset
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM
from split_brain_go.training.adapter_train import AdapterTrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gonet-ckpt", type=Path, required=True)
    p.add_argument("--gonet-config", choices=["poc", "default"], default="default")
    p.add_argument("--adapter-ckpt", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--llm-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file to write the formatted samples to.",
    )
    return p.parse_args()


def _action_to_str(a: int) -> str:
    if a == 81:
        return "PASS"
    r, c = a // 9, a % 9
    return f"({r},{c})"


@torch.no_grad()
def greedy_generate(
    instrumented: InstrumentedLLM,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    adapter_tokens: torch.Tensor | None,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Naive greedy decoding (no KV cache; fine for small N).

    Repeatedly forwards the growing sequence and appends argmax. Returns
    the full sequence including the prompt.
    """
    ids = prompt_ids
    am = attention_mask
    for _ in range(max_new_tokens):
        logits = instrumented(
            input_ids=ids, adapter_tokens=adapter_tokens, attention_mask=am
        )
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
        am = torch.cat([am, torch.ones_like(next_tok)], dim=1)
        if eos_id is not None and (next_tok == eos_id).all():
            break
    return ids


def main() -> int:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # ---------------------------------------------------------------- LLM
    print(f"Loading {args.llm_id} ...")
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
        base_llm, inject_layers=adapter_cfg.inject_layers, n_heads=8,
    ).to(device)
    instrumented.adapter_blocks = instrumented.adapter_blocks.to(torch.bfloat16)

    class _AdapterBundle(torch.nn.Module):
        def __init__(self, resampler, blocks) -> None:
            super().__init__()
            self.resampler = resampler
            self.adapter_blocks = blocks

    bundle = _AdapterBundle(resampler, instrumented.adapter_blocks)
    payload = torch.load(args.adapter_ckpt, map_location=device, weights_only=False)
    bundle.load_state_dict(payload["model"])
    print("Adapter loaded.")

    instrumented.eval()
    resampler.eval()

    # ---------------------------------------------------------------- data
    examples = load_dataset(args.dataset)
    print(f"{len(examples)} examples loaded; sampling {args.n_samples}.")
    rng = random.Random(args.seed)
    chosen = rng.sample(examples, k=min(args.n_samples, len(examples)))

    # ---------------------------------------------------------------- generate
    prompt_text = args.prompt
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_attn = prompt_enc["attention_mask"].to(device)
    prompt_len = prompt_ids.shape[1]
    eos_id = tokenizer.eos_token_id

    blocks = []
    for k, ex in enumerate(chosen):
        board = ex.board.unsqueeze(0).to(device)
        _, _, acts = gonet.forward_with_acts(board, layers=adapter_cfg.act_layers)
        acts = {kk: v.to(torch.bfloat16) for kk, v in acts.items()}
        adapter_tokens = resampler(acts)

        # Trained: adapter ON
        out_trained = greedy_generate(
            instrumented, prompt_ids, prompt_attn, adapter_tokens,
            max_new_tokens=args.max_new_tokens, eos_id=eos_id,
        )
        trained_text = tokenizer.decode(
            out_trained[0, prompt_len:], skip_special_tokens=True
        )

        # Output-only: adapter OFF
        out_off = greedy_generate(
            instrumented, prompt_ids, prompt_attn, None,
            max_new_tokens=args.max_new_tokens, eos_id=eos_id,
        )
        off_text = tokenizer.decode(
            out_off[0, prompt_len:], skip_special_tokens=True
        )

        block = []
        block.append("=" * 78)
        block.append(
            f"[{k:02d}] game={ex.game_id} move={ex.move_number}  "
            f"action={ex.action}={_action_to_str(ex.action)}  "
            f"value_before={ex.value_before:+.2f}"
        )
        block.append("-" * 78)
        block.append("GROUND TRUTH : " + ex.explanation.strip())
        block.append("TRAINED      : " + trained_text.strip())
        block.append("OUTPUT-ONLY  : " + off_text.strip())
        text = "\n".join(block)
        print(text)
        blocks.append(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n\n".join(blocks) + "\n")
        print(f"\nSamples saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
