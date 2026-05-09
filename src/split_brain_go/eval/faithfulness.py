"""Faithfulness measurement -- IAS (Information Ablation Score).

The first of three faithfulness probes from ``evaluation_protocol.md``.
The question:

    *If we randomly zero out fraction r of Go-Net's activation channels
    before they reach the adapter, how much does the LLM's perplexity on
    the validation set increase?*

Interpretation:
    * If perplexity rises **a lot** when channels are masked -> the LLM
      is using the activations, learning is *faithful*.
    * If perplexity stays the same -> the LLM has memorised the
      synthetic template structure and ignores the activations.

APC (board-swap consistency) lives in ``apc.py``; CFC is TBD.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..training.adapter_train import adapter_loss

if TYPE_CHECKING:
    from ..adapter.projection import (
        AsymmetricPerceiverResampler,
        PerceiverResampler,
    )
    from ..gonet.network import GoNet
    from ..llm.instrumented import InstrumentedLLM


# ============================================================ helpers


def mask_activation_channels(
    acts: dict[int, torch.Tensor],
    mask_ratio: float,
    rng: np.random.Generator,
) -> dict[int, torch.Tensor]:
    """Zero out ``mask_ratio`` fraction of channels per Go-Net layer."""
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if mask_ratio == 0.0:
        return acts
    masked: dict[int, torch.Tensor] = {}
    for lid, a in acts.items():
        c = a.shape[1]
        n_keep = max(0, int(round(c * (1.0 - mask_ratio))))
        if n_keep >= c:
            masked[lid] = a
            continue
        keep = rng.choice(c, size=n_keep, replace=False)
        mask_vec = torch.zeros(c, dtype=a.dtype, device=a.device)
        mask_vec[keep] = 1.0
        masked[lid] = a * mask_vec.view(1, c, 1, 1)
    return masked


# ============================================================ IAS


@torch.no_grad()
def measure_ias(
    instrumented: "InstrumentedLLM",
    gonet: "GoNet",
    resampler: "PerceiverResampler | AsymmetricPerceiverResampler",
    val_loader: DataLoader,
    act_layers: list[int],
    device: torch.device,
    mask_ratios: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_seeds: int = 3,
    adapter_dtype: torch.dtype = torch.bfloat16,
    rng_seed: int = 0,
    max_batches: int | None = None,
) -> dict:
    """Sweep over ``mask_ratios``, return per-ratio loss + summary IAS."""
    instrumented.eval()
    resampler.eval()
    gonet.eval()

    results: dict[float, dict[str, float]] = {}

    for ratio in mask_ratios:
        seeds_for_this = 1 if ratio in (0.0, 1.0) else n_seeds
        per_seed_losses = []
        for s_idx in range(seeds_for_this):
            seed = rng_seed + s_idx * 1009 + int(ratio * 1000)
            rng = np.random.default_rng(seed)

            total_loss = 0.0
            total_tokens = 0
            n_batches = 0
            for batch in val_loader:
                boards = batch["board"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, _, acts = gonet.forward_with_acts(boards, layers=act_layers)
                if ratio > 0.0:
                    acts = mask_activation_channels(acts, ratio, rng)
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

            avg = total_loss / max(total_tokens, 1)
            per_seed_losses.append(avg)

        avg_loss = sum(per_seed_losses) / len(per_seed_losses)
        results[float(ratio)] = {
            "loss": avg_loss,
            "perplexity": float(math.exp(min(avg_loss, 50.0))),
            "n_seeds": len(per_seed_losses),
        }

    baseline = results[0.0]["loss"]
    fully_masked = results.get(1.0, {"loss": float("nan")})["loss"]
    ias_05 = (
        (results[0.5]["loss"] - baseline) / baseline
        if 0.5 in results
        else float("nan")
    )
    return {
        "by_ratio": results,
        "baseline_loss": baseline,
        "ias_at_0.5": ias_05,
        "fully_masked_loss": fully_masked,
        "fully_masked_loss_delta": (
            (fully_masked - baseline) / baseline
            if not math.isnan(fully_masked)
            else float("nan")
        ),
    }


def format_ias_report(result: dict) -> str:
    """Render the IAS result as a small text table."""
    lines = ["mask_ratio   loss    perplexity   n_seeds"]
    lines.append("-" * 48)
    for ratio in sorted(result["by_ratio"].keys()):
        r = result["by_ratio"][ratio]
        lines.append(
            f"  {ratio:.2f}      {r['loss']:.4f}   {r['perplexity']:8.3f}   {r['n_seeds']}"
        )
    lines.append("-" * 48)
    lines.append(f"baseline loss          : {result['baseline_loss']:.4f}")
    lines.append(f"IAS at 0.5             : {result['ias_at_0.5']:+.4f}")
    lines.append(
        f"fully-masked d (vs base): {result['fully_masked_loss_delta']:+.4f}"
    )
    interp = _interpret_ias(result["ias_at_0.5"], result["fully_masked_loss_delta"])
    lines.append("")
    lines.append(f"INTERPRETATION: {interp}")
    return "\n".join(lines)


def _interpret_ias(ias_05: float, full_delta: float) -> str:
    if math.isnan(ias_05):
        return "no IAS could be computed"
    if full_delta < 0.05:
        return (
            "UNFAITHFUL: even fully-masked activations barely change loss. "
            "The adapter is likely ignoring Go-Net activations."
        )
    if ias_05 < 0.05:
        return (
            "WEAK / NEAR-UNFAITHFUL: half-masked loss is barely above baseline. "
            "Most learning may be from the prompt + template structure."
        )
    if ias_05 < 0.30:
        return (
            "PARTIAL: half-masked loss is meaningfully higher than baseline, "
            "but the adapter may still rely heavily on template structure."
        )
    return (
        "FAITHFUL signal: half-masked loss is much higher than baseline, "
        "suggesting the LLM substantively uses Go-Net activations."
    )


__all__ = [
    "mask_activation_channels",
    "measure_ias",
    "format_ias_report",
]
