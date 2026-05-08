"""Unit tests for GatedCrossAttentionBlock.

Most important test: ``test_zero_init_is_identity``. The block at init must
produce ``hidden`` unchanged — otherwise inserting it into a pretrained
LLM would corrupt the model on the first forward.
"""

from __future__ import annotations

import math

import pytest
import torch

from split_brain_go.adapter.xattn import GatedCrossAttentionBlock


# ============================================================ shape


def test_output_shape_matches_input():
    block = GatedCrossAttentionBlock(d_model=64, n_heads=4)
    hidden = torch.randn(3, 12, 64)
    adapter = torch.randn(3, 8, 64)
    out = block(hidden, adapter)
    assert out.shape == hidden.shape


def test_works_with_different_seq_lengths():
    block = GatedCrossAttentionBlock(d_model=64, n_heads=4)
    hidden = torch.randn(2, 100, 64)
    adapter = torch.randn(2, 4, 64)
    out = block(hidden, adapter)
    assert out.shape == (2, 100, 64)


# ============================================================ identity


def test_zero_init_is_identity():
    """At initialisation both gates are 0, so output must equal input
    exactly. This is the critical Flamingo property — inserting the
    block into a frozen LLM cannot alter forward output until training
    has actually moved the gates."""
    block = GatedCrossAttentionBlock(d_model=64, n_heads=4)
    hidden = torch.randn(2, 16, 64)
    adapter = torch.randn(2, 8, 64)
    out = block(hidden, adapter)
    # Even with random inputs and weights, gates of zero zero out both
    # contributions, so out == hidden bitwise.
    assert torch.allclose(out, hidden, atol=0)


def test_open_gate_changes_output():
    """If we manually open the gates, the output differs from input."""
    block = GatedCrossAttentionBlock(d_model=64, n_heads=4)
    with torch.no_grad():
        block.gate_attn.fill_(2.0)  # tanh(2) ≈ 0.96
        block.gate_ffn.fill_(2.0)
    hidden = torch.randn(2, 16, 64)
    adapter = torch.randn(2, 8, 64)
    out = block(hidden, adapter)
    assert not torch.allclose(out, hidden, atol=1e-4), (
        "Output should differ from hidden once gates open"
    )


def test_different_adapter_changes_output_when_open():
    """With nonzero gates, different adapter tokens should yield
    different outputs — i.e. the cross-attention is actually consulting
    the adapter input."""
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    with torch.no_grad():
        block.gate_attn.fill_(2.0)
    hidden = torch.randn(1, 8, 32)
    adapter_a = torch.randn(1, 4, 32)
    adapter_b = torch.randn(1, 4, 32)
    out_a = block(hidden, adapter_a)
    out_b = block(hidden, adapter_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


# ============================================================ gradient


def test_gradient_flows_to_gates():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    hidden = torch.randn(2, 4, 32)
    adapter = torch.randn(2, 4, 32)
    out = block(hidden, adapter)
    loss = out.pow(2).mean()
    loss.backward()
    # Gates start at 0 — but the gradient w.r.t. them should be
    # non-zero when training data flows through.
    assert block.gate_attn.grad is not None
    assert block.gate_ffn.grad is not None
    # At least one of the two should have a non-zero gradient.
    assert (
        block.gate_attn.grad.abs().sum() > 0
        or block.gate_ffn.grad.abs().sum() > 0
    )


def test_gradient_to_all_parameters():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    with torch.no_grad():
        block.gate_attn.fill_(0.5)
        block.gate_ffn.fill_(0.5)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    adapter = torch.randn(1, 4, 32, requires_grad=True)
    out = block(hidden, adapter)
    loss = out.pow(2).mean()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


# ============================================================ mask


def test_adapter_mask_shape_accepted():
    """A boolean mask over adapter tokens should be accepted without error."""
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    with torch.no_grad():
        block.gate_attn.fill_(1.0)
    hidden = torch.randn(2, 6, 32)
    adapter = torch.randn(2, 4, 32)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    out = block(hidden, adapter, adapter_mask=mask)
    assert out.shape == (2, 6, 32)
    assert torch.isfinite(out).all()


# ============================================================ errors


def test_d_model_mismatch_raises():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    hidden = torch.randn(2, 4, 64)  # wrong d
    adapter = torch.randn(2, 4, 32)
    with pytest.raises(ValueError):
        block(hidden, adapter)


def test_batch_mismatch_raises():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    hidden = torch.randn(2, 4, 32)
    adapter = torch.randn(3, 4, 32)  # batch mismatch
    with pytest.raises(ValueError):
        block(hidden, adapter)


def test_2d_input_raises():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    hidden = torch.randn(4, 32)  # missing batch dim
    adapter = torch.randn(2, 4, 32)
    with pytest.raises(ValueError):
        block(hidden, adapter)


# ============================================================ diagnostics


def test_gate_values_property():
    block = GatedCrossAttentionBlock(d_model=32, n_heads=4)
    a, f = block.gate_values
    assert math.isclose(a, 0.0, abs_tol=1e-7)
    assert math.isclose(f, 0.0, abs_tol=1e-7)
    with torch.no_grad():
        block.gate_attn.fill_(1.0)
    a2, _ = block.gate_values
    assert math.isclose(a2, math.tanh(1.0), abs_tol=1e-5)
