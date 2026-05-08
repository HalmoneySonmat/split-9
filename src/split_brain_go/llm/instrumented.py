"""Instrumented LLM — frozen base + gated cross-attention injections.

Wraps a HuggingFace causal language model (e.g. TinyLlama-1.1B) and inserts
``GatedCrossAttentionBlock`` instances at user-chosen layer indices. The
base model's parameters are frozen at construction; only the inserted
adapter blocks are trainable. On forward, the caller passes the usual
``input_ids`` plus a tensor of adapter tokens (typically from a Perceiver
Resampler over Go-Net activations); the blocks merge that signal into
the LLM's hidden state via the registered forward hooks.

Why hooks (not subclassing). HF Llama (and most causal LMs) have a
deeply nested forward; subclassing means re-implementing a moving target
across versions. Forward hooks let us intercept *between* layers without
touching the library internals. Each hook receives the layer's output,
applies our cross-attention block, and returns the modified output.

Two design points worth flagging:
    * Adapter tokens are stashed on ``self._current_adapter_tokens`` at
      forward entry and cleared in ``finally``. Synchronous, single-thread.
    * If ``adapter_tokens=None`` the hooks degrade to no-ops, so the
      wrapped model is identical to the base. Useful for ablation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..adapter.xattn import GatedCrossAttentionBlock


class InstrumentedLLM(nn.Module):
    """Frozen LLM with gated cross-attention injections at chosen layers.

    Args:
        base_model: A HuggingFace ``PreTrainedModel`` (e.g. ``LlamaForCausalLM``).
            We expect ``base_model.model.layers`` to be a ``ModuleList``.
        inject_layers: Indices into ``base_model.model.layers``. Each gets
            its own ``GatedCrossAttentionBlock`` whose output is *added*
            (gated) to the layer's output.
        n_heads: Heads in each cross-attention. Defaults to 8.
        d_model: LLM hidden size. Auto-detected from ``base_model.config``
            if available; pass explicitly only for non-standard models.
        ffn_mult: Multiplier for FFN inside each block.

    Notes:
        * Base model is frozen (``requires_grad=False`` for every param).
        * Adapter blocks are trainable.
        * No layer-index validation against the actual layer count is done
          at construction; an out-of-range index will fail at hook
          registration with a clear traceback.
    """

    def __init__(
        self,
        base_model: nn.Module,
        inject_layers: list[int],
        n_heads: int = 8,
        d_model: int | None = None,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.base = base_model

        # Detect hidden size.
        if d_model is None:
            try:
                d_model = int(base_model.config.hidden_size)
            except AttributeError as exc:
                raise ValueError(
                    "d_model not given and base_model has no .config.hidden_size"
                ) from exc
        self.d_model = d_model

        if not inject_layers:
            raise ValueError("inject_layers must be non-empty")
        # Sort + de-dup to make hook registration deterministic.
        self.inject_layers = sorted(set(int(i) for i in inject_layers))

        # Freeze the base.
        for p in self.base.parameters():
            p.requires_grad = False

        # One adapter block per injection point. Use a ModuleList so
        # PyTorch tracks them as children (parameters will be returned
        # by self.parameters()).
        self.adapter_blocks = nn.ModuleList(
            [
                GatedCrossAttentionBlock(d_model, n_heads, ffn_mult=ffn_mult)
                for _ in self.inject_layers
            ]
        )

        # Adapter tokens for the current forward, stashed by forward(),
        # read by the hooks. ``None`` means the adapter is bypassed.
        self._current_adapter_tokens: Tensor | None = None

        # Register hooks; keep handles in case caller wants to remove them.
        self._hook_handles: list[Any] = []
        layers = _get_decoder_layers(self.base)
        for layer_idx, block in zip(self.inject_layers, self.adapter_blocks):
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(
                    f"inject_layer {layer_idx} out of range "
                    f"[0, {len(layers) - 1}]"
                )
            handle = layers[layer_idx].register_forward_hook(
                self._make_hook(block)
            )
            self._hook_handles.append(handle)

    # --------------------------------------------------------------- hooks

    def _make_hook(self, block: GatedCrossAttentionBlock):
        """Build a forward_hook closure capturing ``block``.

        HF decoder layers commonly return a tuple ``(hidden, attn_weights, ...)``;
        a few return a bare tensor. We handle both.
        """

        def hook(_module, _args, output):
            adapter = self._current_adapter_tokens
            if adapter is None:
                return output  # unchanged

            if isinstance(output, tuple):
                hidden = output[0]
                modified = block(hidden, adapter)
                return (modified, *output[1:])
            else:
                return block(output, adapter)

        return hook

    # ------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: Tensor,
        adapter_tokens: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run the base LLM with adapter injections.

        Args:
            input_ids: ``(B, T)`` token ids.
            adapter_tokens: ``(B, N, d_model)`` from the Perceiver Resampler.
                If ``None``, the wrapper acts as the bare frozen base.
            attention_mask: optional ``(B, T)`` standard HF attention mask.
            **kwargs: forwarded to the base model (e.g. ``past_key_values``).

        Returns:
            Logits: ``(B, T, vocab_size)``.
        """
        try:
            self._current_adapter_tokens = adapter_tokens
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            self._current_adapter_tokens = None

        # HF returns a CausalLMOutputWithPast or similar; logits is standard.
        logits = getattr(outputs, "logits", None)
        if logits is None:
            # Some (rare) bare models return a tensor directly.
            return outputs
        return logits

    # ----------------------------------------------------- diagnostics

    def gate_values(self) -> dict[int, tuple[float, float]]:
        """Map each injection layer index to its current ``(gate_attn, gate_ffn)``."""
        return {
            li: blk.gate_values
            for li, blk in zip(self.inject_layers, self.adapter_blocks)
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All adapter parameters (the only trainable ones)."""
        return list(self.adapter_blocks.parameters())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_frozen(self) -> int:
        return sum(p.numel() for p in self.base.parameters())

    # --------------------------------------------------------- cleanup

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


# -------------------------------------------------------------- helpers


def _get_decoder_layers(base_model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer ModuleList. Works for Llama-family models."""
    # Most HF causal LMs: model.model.layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
        if isinstance(layers, nn.ModuleList):
            return layers
    # GPT-2 style: model.transformer.h
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        h = base_model.transformer.h
        if isinstance(h, nn.ModuleList):
            return h
    raise AttributeError(
        "Could not locate decoder layers. Expected either "
        "base_model.model.layers (Llama-family) or "
        "base_model.transformer.h (GPT-style)."
    )


__all__ = ["InstrumentedLLM"]
