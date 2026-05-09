"""APC -- Activation Patching Consistency.

For each batch we run two forwards on the same ``input_ids`` and
``labels``:

    matched     -- adapter sees activations from the matching board.
    mismatched  -- activations are deranged within the batch, so the
                   adapter sees activations from a *different* board
                   than the explanation belongs to.

If the adapter is faithfully passing board-specific signal, the
mismatched forward should produce a higher loss. If the adapter is
mostly contributing a constant domain prior, the two losses will be
approximately equal.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of ``range(n)`` with no fixed points.

    Falls back to a cyclic shift if rejection sampling spins for too
    long. For ``n == 1`` returns ``[0]`` and the caller should skip.
    """
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([1, 0])
    for _ in range(50):
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm
    return np.concatenate([np.arange(1, n), [0]])


@torch.no_grad()
def measure_apc(
    instrumented: "InstrumentedLLM",
    gonet: "GoNet",
    resampler: "PerceiverResampler | AsymmetricPerceiverResampler",
    val_loader: DataLoader,
    act_layers: list[int],
    device: torch.device,
    n_seeds: int = 3,
    adapter_dtype: torch.dtype = torch.bfloat16,
    rng_seed: int = 0,
    max_batches: int | None = None,
) -> dict:
    """Activation Patching Consistency -- does the LLM care which board?

    Returns:
        ``{"matched_loss", "mismatched_loss", "apc", "n_seeds",
           "per_seed_matched", "per_seed_mismatched"}`` with
        ``apc = (mismatched - matched) / matched``.
    """
    instrumented.eval()
    resampler.eval()
    gonet.eval()

    per_seed_matched: list[float] = []
    per_seed_mismatched: list[float] = []

    for s_idx in range(max(1, n_seeds)):
        rng = np.random.default_rng(rng_seed + s_idx * 1009)

        m_loss = 0.0
        m_tok = 0
        mm_loss = 0.0
        mm_tok = 0
        n_batches = 0

        for batch in val_loader:
            boards = batch["board"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = boards.shape[0]

            if B < 2:
                continue

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
            m_loss += float(loss.item()) * n_real
            m_tok += n_real

            perm_np = _derangement(B, rng)
            perm = torch.from_numpy(perm_np).to(device).long()
            acts_shuf = {k: v.index_select(0, perm) for k, v in acts.items()}
            adapter_tokens_shuf = resampler(acts_shuf)
            logits = instrumented(
                input_ids=input_ids,
                adapter_tokens=adapter_tokens_shuf,
                attention_mask=attention_mask,
            )
            loss_mm = adapter_loss(logits, labels)
            mm_loss += float(loss_mm.item()) * n_real
            mm_tok += n_real

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        per_seed_matched.append(m_loss / max(m_tok, 1))
        per_seed_mismatched.append(mm_loss / max(mm_tok, 1))

    matched = sum(per_seed_matched) / len(per_seed_matched)
    mismatched = sum(per_seed_mismatched) / len(per_seed_mismatched)
    apc = (mismatched - matched) / max(matched, 1e-9)

    return {
        "matched_loss": matched,
        "mismatched_loss": mismatched,
        "apc": apc,
        "n_seeds": len(per_seed_matched),
        "per_seed_matched": per_seed_matched,
        "per_seed_mismatched": per_seed_mismatched,
    }


def format_apc_report(result: dict) -> str:
    """Render the APC result as a small text block."""
    lines = []
    lines.append(f"matched loss     : {result['matched_loss']:.4f}")
    lines.append(f"mismatched loss  : {result['mismatched_loss']:.4f}")
    lines.append(f"APC              : {result['apc']:+.4f}")
    lines.append(f"n_seeds          : {result['n_seeds']}")
    if len(result.get("per_seed_matched", [])) > 1:
        lines.append("")
        lines.append("per-seed (matched / mismatched):")
        for m, mm in zip(
            result["per_seed_matched"], result["per_seed_mismatched"]
        ):
            lines.append(f"  {m:.4f}   /   {mm:.4f}")
    lines.append("")
    lines.append(f"INTERPRETATION: {_interpret_apc(result['apc'])}")
    return "\n".join(lines)


def _interpret_apc(apc: float) -> str:
    if math.isnan(apc):
        return "no APC could be computed"
    if apc < 0.02:
        return (
            "UNFAITHFUL: mismatched activations barely change the loss. "
            "The adapter's contribution is essentially board-independent."
        )
    if apc < 0.10:
        return (
            "WEAK: small but non-zero board specificity. The adapter "
            "leaks some per-board information."
        )
    if apc < 0.30:
        return (
            "PARTIAL: mismatched loss meaningfully exceeds matched. The "
            "adapter encodes per-board signal that the LLM uses."
        )
    return (
        "FAITHFUL signal: mismatched activations significantly hurt "
        "loss; the adapter is causally passing board-specific information."
    )


__all__ = [
    "measure_apc",
    "format_apc_report",
]
