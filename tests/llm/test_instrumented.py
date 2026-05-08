"""Unit tests for InstrumentedLLM, using a fake Llama-shaped base model.

We don't download TinyLlama here — that's a 2 GB file and needs network.
Instead we build a minimal shape-compatible mock so the wrapper logic
(hook registration, adapter forwarding, frozen base, gates) is fully
exercised in CPU under a second.

Real-model integration testing happens in a separate scripted check
once Phase 1.3b training is done.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from split_brain_go.llm.instrumented import InstrumentedLLM, _get_decoder_layers


# ============================================================ mock LLM


class _FakeLayer(nn.Module):
    """Simulates one Llama decoder layer: returns ``(hidden, ...)`` tuple."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.fc(x) + x,)


class _FakeBaseInner(nn.Module):
    """Simulates ``LlamaModel`` — embeddings + layer stack + final norm."""

    def __init__(self, n_layers: int, d: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return self.norm(x)


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeBaseLLM(nn.Module):
    """Pretends to be ``LlamaForCausalLM`` for the purposes of InstrumentedLLM."""

    def __init__(self, n_layers: int = 4, d: int = 32, vocab: int = 100) -> None:
        super().__init__()

        class _Cfg:
            hidden_size = d

        self.config = _Cfg()
        self.model = _FakeBaseInner(n_layers, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> _FakeOutputs:
        h = self.model(input_ids)
        logits = self.lm_head(h)
        return _FakeOutputs(logits)


# ============================================================ fixtures


@pytest.fixture()
def fake_base() -> _FakeBaseLLM:
    torch.manual_seed(0)
    return _FakeBaseLLM(n_layers=4, d=32, vocab=50)


# ============================================================ helpers


def test_get_decoder_layers_finds_llama_path(fake_base):
    layers = _get_decoder_layers(fake_base)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 4


def test_get_decoder_layers_raises_on_unknown_shape():
    weird = nn.Linear(10, 10)
    with pytest.raises(AttributeError):
        _get_decoder_layers(weird)


# ============================================================ basic


def test_construct_with_inject_layers(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[1, 3], n_heads=4)
    assert wrapped.inject_layers == [1, 3]
    assert len(wrapped.adapter_blocks) == 2


def test_empty_inject_layers_raises(fake_base):
    with pytest.raises(ValueError):
        InstrumentedLLM(fake_base, inject_layers=[], n_heads=4)


def test_out_of_range_inject_layer_raises(fake_base):
    with pytest.raises(ValueError):
        InstrumentedLLM(fake_base, inject_layers=[10], n_heads=4)  # only 4 layers


# ============================================================ frozen


def test_base_parameters_frozen(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[1], n_heads=4)
    for p in wrapped.base.parameters():
        assert not p.requires_grad


def test_adapter_parameters_trainable(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[1], n_heads=4)
    for p in wrapped.adapter_blocks.parameters():
        assert p.requires_grad


def test_trainable_count_is_just_adapters(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[1, 3], n_heads=4)
    assert wrapped.num_trainable() == sum(
        p.numel() for p in wrapped.adapter_blocks.parameters()
    )
    assert wrapped.num_frozen() > 0


# ============================================================ forward


def test_forward_with_none_adapter_matches_base(fake_base):
    """adapter_tokens=None should give the same logits as the bare base.

    Construct two paths: bare base, wrapped without adapter. They must
    produce identical logits, since the wrapper's hooks no-op when
    ``_current_adapter_tokens is None``.
    """
    torch.manual_seed(1)
    input_ids = torch.randint(0, 50, (2, 7))

    # Wrapped with no adapter — should equal base's forward.
    wrapped = InstrumentedLLM(fake_base, inject_layers=[2], n_heads=4)
    wrapped.eval()
    with torch.no_grad():
        logits_wrapped = wrapped(input_ids, adapter_tokens=None)
        logits_base = fake_base(input_ids).logits
    assert torch.allclose(logits_wrapped, logits_base, atol=1e-6)


def test_forward_with_adapter_zero_init_still_matches(fake_base):
    """At init the gates are 0, so even WITH adapter tokens the output
    must equal the base. This is the Flamingo-style preservation."""
    torch.manual_seed(2)
    input_ids = torch.randint(0, 50, (2, 5))
    adapter = torch.randn(2, 8, 32)

    wrapped = InstrumentedLLM(fake_base, inject_layers=[1, 3], n_heads=4)
    wrapped.eval()
    with torch.no_grad():
        logits_wrapped = wrapped(input_ids, adapter_tokens=adapter)
        logits_base = fake_base(input_ids).logits
    assert torch.allclose(logits_wrapped, logits_base, atol=1e-6)


def test_forward_with_open_gate_changes_logits(fake_base):
    """If we open one gate manually, output must diverge from base."""
    torch.manual_seed(3)
    input_ids = torch.randint(0, 50, (1, 4))
    adapter = torch.randn(1, 6, 32)

    wrapped = InstrumentedLLM(fake_base, inject_layers=[2], n_heads=4)
    with torch.no_grad():
        wrapped.adapter_blocks[0].gate_attn.fill_(2.0)
        logits_wrapped = wrapped(input_ids, adapter_tokens=adapter)
        logits_base = fake_base(input_ids).logits
    assert not torch.allclose(logits_wrapped, logits_base, atol=1e-3)


def test_forward_returns_logits_shape(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[2], n_heads=4)
    input_ids = torch.randint(0, 50, (3, 11))
    out = wrapped(input_ids)
    # vocab=50 set in fixture
    assert out.shape == (3, 11, 50)


# ============================================================ gradient


def test_gradient_only_to_adapter(fake_base):
    """Loss backward populates adapter grads but leaves base params with
    grad=None (or zero), since base is frozen via requires_grad=False."""
    wrapped = InstrumentedLLM(fake_base, inject_layers=[2], n_heads=4)
    # Open a gate so the adapter actually contributes.
    with torch.no_grad():
        wrapped.adapter_blocks[0].gate_attn.fill_(0.5)

    input_ids = torch.randint(0, 50, (1, 4))
    adapter = torch.randn(1, 6, 32)
    out = wrapped(input_ids, adapter_tokens=adapter)
    loss = out.pow(2).mean()
    loss.backward()

    # Adapter gradients exist
    for name, p in wrapped.adapter_blocks.named_parameters():
        assert p.grad is not None, f"adapter param {name} has no grad"

    # Base parameters either have None grad or zero (autograd may set
    # zero in some PyTorch versions even when requires_grad=False).
    for name, p in wrapped.base.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"frozen base param {name} got nonzero grad"
        )


# ============================================================ diagnostics


def test_gate_values_initial_zero(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[1, 2, 3], n_heads=4)
    gv = wrapped.gate_values()
    assert set(gv.keys()) == {1, 2, 3}
    for li in gv:
        a, f = gv[li]
        assert abs(a) < 1e-7 and abs(f) < 1e-7


def test_remove_hooks_restores_base_behavior(fake_base):
    wrapped = InstrumentedLLM(fake_base, inject_layers=[2], n_heads=4)
    with torch.no_grad():
        wrapped.adapter_blocks[0].gate_attn.fill_(2.0)

    input_ids = torch.randint(0, 50, (1, 4))
    adapter = torch.randn(1, 6, 32)

    wrapped.eval()
    with torch.no_grad():
        before_remove = wrapped(input_ids, adapter_tokens=adapter)
        wrapped.remove_hooks()
        after_remove = wrapped(input_ids, adapter_tokens=adapter)
        base_only = fake_base(input_ids).logits

    # After removing hooks, even with adapter tokens, output equals base.
    assert torch.allclose(after_remove, base_only, atol=1e-6)
    assert not torch.allclose(before_remove, after_remove, atol=1e-3)
