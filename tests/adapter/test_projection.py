"""Unit tests for PerceiverResampler.

CPU-only. Verifies shape, parameter count, gradient flow, and rejection
of mismatched activation dicts.
"""

from __future__ import annotations

import pytest
import torch

from split_brain_go.adapter.projection import (
    AsymmetricPerceiverResampler,
    PerceiverBlock,
    PerceiverResampler,
)


# ============================================================ basics


def test_resampler_output_shape():
    """Standard PoC settings: 3 layers of (8, 64, 9, 9) → (8, 32, 256)."""
    resampler = PerceiverResampler(
        layer_channels={1: 64, 3: 64, 5: 64},
        n_latents=32,
        d_model=256,
        n_heads=4,
        n_blocks=2,
    )
    acts = {
        1: torch.randn(8, 64, 9, 9),
        3: torch.randn(8, 64, 9, 9),
        5: torch.randn(8, 64, 9, 9),
    }
    out = resampler(acts)
    assert out.shape == (8, 32, 256)


def test_resampler_handles_different_channel_widths():
    """Layers may have different channel counts (e.g. early/late residual)."""
    resampler = PerceiverResampler(
        layer_channels={0: 32, 2: 64, 4: 128},
        n_latents=16,
        d_model=128,
        n_heads=4,
        n_blocks=1,
    )
    acts = {
        0: torch.randn(2, 32, 9, 9),
        2: torch.randn(2, 64, 9, 9),
        4: torch.randn(2, 128, 9, 9),
    }
    out = resampler(acts)
    assert out.shape == (2, 16, 128)


def test_single_layer():
    resampler = PerceiverResampler(
        layer_channels={5: 128},
        n_latents=8,
        d_model=64,
        n_heads=2,
        n_blocks=1,
    )
    acts = {5: torch.randn(4, 128, 9, 9)}
    out = resampler(acts)
    assert out.shape == (4, 8, 64)


# ============================================================ errors


def test_missing_layer_raises():
    resampler = PerceiverResampler(
        layer_channels={1: 64, 3: 64},
        n_latents=8,
        d_model=64,
    )
    acts = {1: torch.randn(2, 64, 9, 9)}  # missing layer 3
    with pytest.raises(KeyError):
        resampler(acts)


def test_extra_layer_raises():
    resampler = PerceiverResampler(
        layer_channels={1: 64},
        n_latents=8,
        d_model=64,
    )
    acts = {
        1: torch.randn(2, 64, 9, 9),
        9: torch.randn(2, 64, 9, 9),  # not declared
    }
    with pytest.raises(KeyError):
        resampler(acts)


def test_wrong_spatial_size_raises():
    resampler = PerceiverResampler(
        layer_channels={1: 64},
        spatial_size=9,
        n_latents=4,
        d_model=32,
    )
    bad_act = torch.randn(1, 64, 7, 7)  # 7x7 instead of 9x9
    with pytest.raises(ValueError):
        resampler({1: bad_act})


def test_empty_layer_channels_raises():
    with pytest.raises(ValueError):
        PerceiverResampler(layer_channels={}, n_latents=8, d_model=32)


# ============================================================ gradient


def test_gradient_flows_to_all_inputs():
    """A scalar loss should produce gradients on every activation tensor."""
    resampler = PerceiverResampler(
        layer_channels={1: 32, 3: 32},
        n_latents=8,
        d_model=64,
        n_heads=4,
        n_blocks=2,
    )
    acts = {
        1: torch.randn(2, 32, 9, 9, requires_grad=True),
        3: torch.randn(2, 32, 9, 9, requires_grad=True),
    }
    out = resampler(acts)
    loss = out.sum()
    loss.backward()
    assert acts[1].grad is not None and acts[1].grad.abs().sum() > 0
    assert acts[3].grad is not None and acts[3].grad.abs().sum() > 0


def test_gradient_flows_to_parameters():
    resampler = PerceiverResampler(
        layer_channels={2: 64},
        n_latents=4,
        d_model=64,
        n_heads=2,
        n_blocks=1,
    )
    acts = {2: torch.randn(1, 64, 9, 9)}
    out = resampler(acts)
    loss = out.pow(2).mean()
    loss.backward()
    for name, p in resampler.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


# ============================================================ params


def test_parameter_count_within_expected_range():
    """For PoC sizes, total params should be a few M, not exploding."""
    resampler = PerceiverResampler(
        layer_channels={1: 64, 3: 64, 5: 64},
        n_latents=32,
        d_model=2048,  # TinyLlama hidden size
        n_heads=8,
        n_blocks=2,
    )
    n = resampler.num_parameters()
    # Loose bounds: 5M to 200M. Most cost is in the cross-attn projections.
    assert 5_000_000 < n < 200_000_000, f"Params={n}"


# ============================================================ block alone


def test_perceiver_block_shapes():
    block = PerceiverBlock(d_model=64, n_heads=4)
    latents = torch.randn(3, 8, 64)
    kv = torch.randn(3, 50, 64)
    out = block(latents, kv)
    assert out.shape == (3, 8, 64)


# ====================================================== layer-emb effect


def test_layer_embedding_distinguishes_layers():
    """Same activation values in different 'layers' should produce different
    KV tokens because of the layer embedding. We just check the resampler
    runs and produces non-NaN output — embeddings are added before
    cross-attn, so identity inputs across layers won't collapse."""
    resampler = PerceiverResampler(
        layer_channels={0: 32, 1: 32},
        n_latents=4,
        d_model=64,
        n_heads=2,
        n_blocks=1,
    )
    same = torch.ones(1, 32, 9, 9)
    out = resampler({0: same.clone(), 1: same.clone()})
    assert torch.isfinite(out).all()


# ============================================================ asymmetric


def test_asymmetric_output_shape_sums_per_layer_tokens():
    """Total output tokens = sum of per-layer counts."""
    res = AsymmetricPerceiverResampler(
        layer_token_counts={3: 8, 4: 12, 5: 16},
        layer_channels={3: 64, 4: 64, 5: 64},
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    acts = {
        3: torch.randn(2, 64, 9, 9),
        4: torch.randn(2, 64, 9, 9),
        5: torch.randn(2, 64, 9, 9),
    }
    out = res(acts)
    # 8 + 12 + 16 = 36 tokens total
    assert out.shape == (2, 36, 64)
    assert res.total_tokens == 36


def test_asymmetric_handles_different_channel_widths():
    res = AsymmetricPerceiverResampler(
        layer_token_counts={3: 4, 5: 8},
        layer_channels={3: 32, 5: 128},
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    acts = {3: torch.randn(1, 32, 9, 9), 5: torch.randn(1, 128, 9, 9)}
    out = res(acts)
    assert out.shape == (1, 12, 64)


def test_asymmetric_keys_must_match():
    with pytest.raises(ValueError):
        AsymmetricPerceiverResampler(
            layer_token_counts={3: 4, 5: 8},
            layer_channels={3: 32, 4: 32},  # key 4 vs 5 mismatch
            d_model=32,
        )


def test_asymmetric_zero_tokens_raises():
    with pytest.raises(ValueError):
        AsymmetricPerceiverResampler(
            layer_token_counts={3: 0, 5: 4},
            layer_channels={3: 32, 5: 32},
            d_model=32,
        )


def test_asymmetric_empty_raises():
    with pytest.raises(ValueError):
        AsymmetricPerceiverResampler(
            layer_token_counts={},
            layer_channels={},
            d_model=32,
        )


def test_asymmetric_missing_layer_at_forward_raises():
    res = AsymmetricPerceiverResampler(
        layer_token_counts={3: 4, 5: 4},
        layer_channels={3: 32, 5: 32},
        d_model=32,
    )
    acts = {3: torch.randn(1, 32, 9, 9)}  # missing layer 5
    with pytest.raises(KeyError):
        res(acts)


def test_asymmetric_gradient_flows_to_each_layer_pool():
    """Each layer's resampler has its own params; gradient should reach all."""
    res = AsymmetricPerceiverResampler(
        layer_token_counts={3: 4, 4: 4, 5: 4},
        layer_channels={3: 32, 4: 32, 5: 32},
        d_model=32,
        n_heads=4,
        n_blocks=1,
    )
    acts = {
        3: torch.randn(1, 32, 9, 9),
        4: torch.randn(1, 32, 9, 9),
        5: torch.randn(1, 32, 9, 9),
    }
    out = res(acts)
    loss = out.pow(2).mean()
    loss.backward()
    # Every per-layer sub-resampler must have grad
    for lid in (3, 4, 5):
        sub = res.per_layer[str(lid)]
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in sub.parameters()
        )
        assert any_grad, f"layer {lid} resampler got no gradient"


def test_asymmetric_higher_layer_more_tokens_works():
    """Smoke test of the actual configuration we're going to use:
    layer 3 → 8 tokens, layer 4 → 12, layer 5 → 16."""
    res = AsymmetricPerceiverResampler(
        layer_token_counts={3: 8, 4: 12, 5: 16},
        layer_channels={3: 128, 4: 128, 5: 128},
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    acts = {
        3: torch.randn(2, 128, 9, 9),
        4: torch.randn(2, 128, 9, 9),
        5: torch.randn(2, 128, 9, 9),
    }
    out = res(acts)
    assert out.shape == (2, 36, 64)
    assert torch.isfinite(out).all()
