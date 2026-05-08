"""Gated cross-attention block (the corpus-callosum-equivalent of the project).

Inserted into selected layers of a frozen LLM. Reads ``adapter_tokens``
(produced by the Perceiver Resampler) and merges that information into
the LLM's hidden state via cross-attention. Two scalar gates control how
much of the cross-attn output and FFN output to mix in.

At initialisation both gates are 0, so ``tanh(0) = 0`` and the block
becomes the identity function: the LLM's pretrained behaviour is fully
preserved. As training progresses, gates open and adapter influence
grows. This is Flamingo's recipe (Alayrac et al., 2022) and the standard
for inserting new modules into a frozen base model.

Layout (pre-norm everywhere):
    out = hidden + tanh(g_attn) · CrossAttn(LN(hidden), LN(adapter))
    out = out    + tanh(g_ffn)  · FFN     (LN(out))
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedCrossAttentionBlock(nn.Module):
    """Single gated cross-attention layer with a gated FFN.

    Args:
        d_model: Hidden dim. Must match the LLM's hidden size.
        n_heads: Attention heads in the cross-attention.
        ffn_mult: FFN hidden = d_model * ffn_mult. 4 is standard.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Cross-attention sublayer
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))

        # FFN sublayer
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------- forward

    def forward(
        self,
        hidden: Tensor,
        adapter_tokens: Tensor,
        adapter_mask: Tensor | None = None,
    ) -> Tensor:
        """Augment ``hidden`` with information from ``adapter_tokens``.

        Args:
            hidden: ``(B, T, d_model)`` from a preceding LLM layer.
            adapter_tokens: ``(B, N, d_model)`` from the Perceiver Resampler.
            adapter_mask: optional ``(B, N)`` boolean. ``True`` marks
                positions to *attend to*; ``False`` masks out (used if some
                adapter slots are padding). If None, all positions attend.

        Returns:
            ``(B, T, d_model)`` — same shape as ``hidden``.
        """
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be 3D, got shape {tuple(hidden.shape)}")
        if adapter_tokens.dim() != 3:
            raise ValueError(
                f"adapter_tokens must be 3D, got shape {tuple(adapter_tokens.shape)}"
            )
        if hidden.shape[0] != adapter_tokens.shape[0]:
            raise ValueError(
                f"batch mismatch: hidden {hidden.shape[0]} vs adapter "
                f"{adapter_tokens.shape[0]}"
            )
        if hidden.shape[2] != self.d_model or adapter_tokens.shape[2] != self.d_model:
            raise ValueError(
                f"d_model mismatch: expected {self.d_model}, got "
                f"hidden {hidden.shape[2]}, adapter {adapter_tokens.shape[2]}"
            )

        # Cross-attention: Q from hidden, K/V from adapter_tokens.
        q = self.norm_q(hidden)
        kv = self.norm_kv(adapter_tokens)

        # MHA's key_padding_mask convention: True means "padding, mask OUT".
        # Our adapter_mask uses True = "attend"; invert.
        key_padding_mask = None if adapter_mask is None else ~adapter_mask
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        hidden = hidden + torch.tanh(self.gate_attn) * attn_out

        # Gated FFN
        ffn_out = self.ffn(self.norm_ffn(hidden))
        hidden = hidden + torch.tanh(self.gate_ffn) * ffn_out
        return hidden

    # ------------------------------------------------------------- helpers

    @property
    def gate_values(self) -> tuple[float, float]:
        """Current ``(tanh(gate_attn), tanh(gate_ffn))`` for diagnostics."""
        with torch.no_grad():
            return (
                float(torch.tanh(self.gate_attn).item()),
                float(torch.tanh(self.gate_ffn).item()),
            )


__all__ = ["GatedCrossAttentionBlock"]
