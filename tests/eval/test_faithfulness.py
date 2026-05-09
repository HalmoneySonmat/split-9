"""Tests for IAS faithfulness measurement.

Mocks the same components as test_adapter_train so we can exercise the
masking + evaluation logic without GPU or 1B-parameter LLMs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from split_brain_go.data.dataset import LABEL_IGNORE, Phase3Dataset, TokenizedExample, phase3_collate
from split_brain_go.eval.apc import (
    _derangement,
    format_apc_report,
    measure_apc,
)
from split_brain_go.eval.faithfulness import (
    format_ias_report,
    mask_activation_channels,
    measure_ias,
)


# ---------------------------------------------------- mocks (parallel to adapter test)


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
    return DataLoader(Phase3Dataset(tokenized), batch_size=batch, collate_fn=phase3_collate)


# ============================================================ masking


def test_mask_ratio_zero_is_identity():
    a = torch.randn(2, 16, 9, 9)
    out = mask_activation_channels({1: a}, 0.0, np.random.default_rng(0))
    assert torch.equal(out[1], a)


def test_mask_ratio_one_zeros_everything():
    a = torch.randn(2, 16, 9, 9)
    out = mask_activation_channels({1: a}, 1.0, np.random.default_rng(0))
    assert (out[1] == 0).all()


def test_mask_ratio_half_zeros_half_channels():
    a = torch.ones(2, 16, 9, 9)
    out = mask_activation_channels({1: a}, 0.5, np.random.default_rng(0))
    # Each channel is either fully kept (1.0) or fully zero
    per_channel_sums = out[1].sum(dim=(0, 2, 3))
    n_kept = (per_channel_sums > 0).sum().item()
    n_zero = (per_channel_sums == 0).sum().item()
    assert n_kept == 8
    assert n_zero == 8


def test_mask_invalid_ratio_raises():
    with pytest.raises(ValueError):
        mask_activation_channels({1: torch.zeros(1, 4, 9, 9)}, -0.1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        mask_activation_channels({1: torch.zeros(1, 4, 9, 9)}, 1.5, np.random.default_rng(0))


def test_mask_preserves_shape_and_dtype():
    a = torch.randn(3, 32, 9, 9, dtype=torch.float32)
    out = mask_activation_channels({2: a}, 0.5, np.random.default_rng(0))
    assert out[2].shape == a.shape
    assert out[2].dtype == a.dtype


# ============================================================ measure_ias


def test_measure_ias_returns_expected_keys():
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1, 3], 16, 4, 32)
    llm = _MockInstrumented(50, 32)
    loader = _make_loader()

    result = measure_ias(
        llm, gonet, resampler, loader,
        act_layers=[1, 3],
        device=torch.device("cpu"),
        mask_ratios=(0.0, 0.5, 1.0),
        n_seeds=1,
        adapter_dtype=torch.float32,
    )
    assert "by_ratio" in result
    assert "baseline_loss" in result
    assert "ias_at_0.5" in result
    assert "fully_masked_loss" in result
    assert set(result["by_ratio"].keys()) == {0.0, 0.5, 1.0}


def test_measure_ias_baseline_finite():
    torch.manual_seed(1)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1], 16, 4, 32)
    llm = _MockInstrumented(50, 32)
    loader = _make_loader()
    result = measure_ias(
        llm, gonet, resampler, loader, [1],
        device=torch.device("cpu"),
        mask_ratios=(0.0, 0.5),
        n_seeds=1,
        adapter_dtype=torch.float32,
    )
    assert math.isfinite(result["baseline_loss"])


def test_format_ias_report_contains_key_pieces():
    fake = {
        "by_ratio": {
            0.0: {"loss": 1.0, "perplexity": math.e, "n_seeds": 1},
            0.5: {"loss": 1.5, "perplexity": math.exp(1.5), "n_seeds": 3},
            1.0: {"loss": 2.0, "perplexity": math.exp(2.0), "n_seeds": 1},
        },
        "baseline_loss": 1.0,
        "ias_at_0.5": 0.5,
        "fully_masked_loss": 2.0,
        "fully_masked_loss_delta": 1.0,
    }
    text = format_ias_report(fake)
    assert "0.50" in text
    assert "INTERPRETATION" in text