"""Tests for APC (Activation Patching Consistency) faithfulness measurement.

Mocks parallel to test_faithfulness.py so we can exercise the
derangement + matched/mismatched eval logic without GPU or 1B-parameter LLMs.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from split_brain_go.data.dataset import (
    LABEL_IGNORE,
    Phase3Dataset,
    TokenizedExample,
    phase3_collate,
)
from split_brain_go.eval.apc import (
    _derangement,
    format_apc_report,
    measure_apc,
)


# ---------------------------------------------------- mocks


class _MockGoNet(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.channels = channels

    def forward_with_acts(self, board, layers):
        B = board.shape[0]
        return (
            torch.zeros(B, 82, device=board.device),
            torch.zeros(B, device=board.device),
            {l: torch.randn(B, self.channels, 9, 9, device=board.device) for l in layers},
        )


class _MockResampler(nn.Module):
    def __init__(self, layer_ids, channels, n_latents, d_model):
        super().__init__()
        self.layer_ids = sorted(layer_ids)
        self.n_latents = n_latents
        self.proj = nn.Linear(channels, d_model)
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))

    def forward(self, acts):
        means = []
        for lid in self.layer_ids:
            means.append(acts[lid].mean(dim=(2, 3)))
        avg = torch.stack(means, dim=0).mean(dim=0)
        z = self.proj(avg).unsqueeze(1)
        return z.expand(-1, self.n_latents, -1).contiguous() + self.latents


class _MockInstrumented(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.adapter_blocks = nn.ModuleList([nn.Linear(d_model, d_model)])

    def forward(self, input_ids, adapter_tokens=None, attention_mask=None):
        x = self.embed(input_ids)
        if adapter_tokens is not None:
            x = x + self.adapter_blocks[0](adapter_tokens.mean(dim=1, keepdim=True))
        return self.lm_head(x)


def _make_loader(n=4, T=8, V=50, batch=2):
    tokenized = []
    for _ in range(n):
        ids = torch.randint(0, V, (T,), dtype=torch.long)
        attn = torch.ones(T, dtype=torch.long)
        labels = ids.clone()
        labels[:3] = LABEL_IGNORE
        tokenized.append(
            TokenizedExample(
                board=torch.randn(8, 9, 9),
                input_ids=ids,
                attention_mask=attn,
                labels=labels,
            )
        )
    return DataLoader(
        Phase3Dataset(tokenized), batch_size=batch, collate_fn=phase3_collate
    )


# ============================================================ derangement


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5, 10, 32):
        for _ in range(20):
            perm = _derangement(n, rng)
            assert perm.shape == (n,)
            assert sorted(perm.tolist()) == list(range(n))
            assert not (perm == np.arange(n)).any(), f"fixed point at n={n}"


def test_derangement_singleton_returns_self():
    rng = np.random.default_rng(0)
    perm = _derangement(1, rng)
    assert perm.tolist() == [0]


# ============================================================ measure_apc


def test_measure_apc_returns_expected_keys():
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1, 3], 16, 4, 32)
    llm = _MockInstrumented(50, 32)
    loader = _make_loader(n=4, batch=2)

    result = measure_apc(
        llm, gonet, resampler, loader,
        act_layers=[1, 3],
        device=torch.device("cpu"),
        n_seeds=2,
        adapter_dtype=torch.float32,
    )
    assert "matched_loss" in result
    assert "mismatched_loss" in result
    assert "apc" in result
    assert "n_seeds" in result
    assert result["n_seeds"] == 2
    assert math.isfinite(result["matched_loss"])
    assert math.isfinite(result["mismatched_loss"])


def test_measure_apc_skips_singleton_batches():
    """Batches of size 1 must not crash measure_apc."""
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1], 16, 4, 32)
    llm = _MockInstrumented(50, 32)
    loader = _make_loader(n=2, batch=1)
    result = measure_apc(
        llm, gonet, resampler, loader, [1],
        device=torch.device("cpu"),
        n_seeds=1,
        adapter_dtype=torch.float32,
    )
    # No batches contributed; safe-divide fallback gives 0.0.
    assert result["matched_loss"] == 0.0
    assert result["mismatched_loss"] == 0.0


def test_format_apc_report_contains_key_pieces():
    fake = {
        "matched_loss": 1.0,
        "mismatched_loss": 1.4,
        "apc": 0.4,
        "n_seeds": 3,
        "per_seed_matched": [1.0, 1.0, 1.0],
        "per_seed_mismatched": [1.4, 1.4, 1.4],
    }
    text = format_apc_report(fake)
    assert "matched loss" in text
    assert "mismatched loss" in text
    assert "APC" in text
    assert "INTERPRETATION" in text
    assert (
        "FAITHFUL" in text
        or "PARTIAL" in text
        or "WEAK" in text
        or "UNFAITHFUL" in text
    )
