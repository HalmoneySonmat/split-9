"""Perceiver Resampler — Go-Net activations → fixed-length token sequence.

Maps the 9×9 spatial activations from one or more Go-Net residual blocks
into ``n_latents`` tokens of dimension ``d_model``, ready to be consumed
by an LLM via cross-attention. The layout mirrors Flamingo's design.

Pipeline:
    1. Per-layer activation (B, C_l, 9, 9) → flatten spatial → (B, 81, C_l)
    2. Per-layer linear projection to d_model: (B, 81, d_model)
    3. Add a layer-id embedding so the LLM can tell where each token came from.
    4. Concatenate across all layers: (B, sum(81 per layer), d_model) — KV.
    5. Cross-attend ``n_latents`` learned query tokens to KV.
    6. Optional further self-attn-like Perceiver blocks for refinement.

Why a Resampler. The LLM's cross-attention cost scales with KV length.
``num_layers × 81`` could be ~243 tokens; that already strains attention
during many decoding steps. Resampling down to 32–64 tokens keeps cost
flat regardless of how many Go-Net activations we expose.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ============================================================== block


class PerceiverBlock(nn.Module):
    """One pass of cross-attention (latents <- KV) followed by an FFN.

    No self-attention on the latents — they're so few that mixing them is
    cheap and doesn't change much. Adding self-attn would double the cost.
    Pre-norm layout: norm before each sublayer, residual after.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, latents: Tensor, kv: Tensor) -> Tensor:
        q = self.norm_q(latents)
        k = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, k, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


# ============================================================ resampler


class PerceiverResampler(nn.Module):
    """Fixed-length token output from variable-many Go-Net activations.

    Args:
        layer_channels: ``{layer_id: channel_count}``. The forward expects
            an ``acts`` dict with the *same* keys.
        spatial_size: H == W of each activation. Default 9 (board size).
        n_latents: Output sequence length.
        d_model: Output hidden size — must match the LLM that consumes the
            tokens (e.g. 2048 for TinyLlama).
        n_heads: Attention heads in each Perceiver block.
        n_blocks: Stack depth.
    """

    def __init__(
        self,
        layer_channels: dict[int, int],
        spatial_size: int = 9,
        n_latents: int = 32,
        d_model: int = 2048,
        n_heads: int = 8,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        if not layer_channels:
            raise ValueError("layer_channels must be non-empty")
        self.layer_ids = sorted(layer_channels.keys())
        self.spatial_size = spatial_size
        self.n_latents = n_latents
        self.d_model = d_model

        # Per-layer projection: linear over the C dim of (B, 81, C) → (B, 81, d_model).
        self.projections = nn.ModuleDict(
            {str(lid): nn.Linear(c, d_model) for lid, c in layer_channels.items()}
        )

        # Layer-id embedding so LLM tokens know provenance.
        self.layer_emb = nn.Embedding(len(self.layer_ids), d_model)

        # Position embedding over the 81 spatial positions (shared across layers).
        n_pos = spatial_size * spatial_size
        self.pos_emb = nn.Embedding(n_pos, d_model)

        # Learned latent queries; small init for stable start.
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [PerceiverBlock(d_model, n_heads) for _ in range(n_blocks)]
        )

    # -------------------------------------------------------------- forward

    def forward(self, acts: dict[int, Tensor]) -> Tensor:
        """Return ``(B, n_latents, d_model)`` from a dict of activations.

        The dict's keys must exactly match the ``layer_channels`` used at
        construction. Missing or extra keys are an error.
        """
        if set(acts.keys()) != set(self.layer_ids):
            raise KeyError(
                f"acts keys {sorted(acts.keys())} != expected {self.layer_ids}"
            )

        # Build KV: per-layer (B, 81, d_model) with layer + pos embeddings.
        all_tokens = []
        n_pos = self.spatial_size * self.spatial_size
        for layer_index, lid in enumerate(self.layer_ids):
            x = acts[lid]
            if x.shape[-2:] != (self.spatial_size, self.spatial_size):
                raise ValueError(
                    f"layer {lid}: expected spatial {self.spatial_size}x{self.spatial_size}, "
                    f"got {tuple(x.shape[-2:])}"
                )
            B, C, H, W = x.shape
            # (B, C, H, W) → (B, H*W, C)
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            # Linear to d_model
            x = self.projections[str(lid)](x)  # (B, 81, d_model)

            # Add positional + layer embeddings (broadcasted over batch).
            pos_ids = torch.arange(n_pos, device=x.device)
            layer_id_t = torch.tensor(layer_index, device=x.device)
            x = x + self.pos_emb(pos_ids).unsqueeze(0)
            x = x + self.layer_emb(layer_id_t).view(1, 1, -1)

            all_tokens.append(x)

        kv = torch.cat(all_tokens, dim=1)  # (B, num_layers * 81, d_model)

        # Expand latent queries to batch.
        B = kv.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Iterate cross-attention.
        for block in self.blocks:
            latents = block(latents, kv)
        return latents

    # ---------------------------------------------------------- introspect

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["PerceiverResampler", "PerceiverBlock"]
