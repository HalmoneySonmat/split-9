"""Phase 2 integration test using mocks (no real LLM download).

Combines GoNet → PerceiverResampler → InstrumentedLLM in a single forward
pass, verifying shape compatibility and the zero-init identity property
end-to-end.

Real-LLM integration is in ``scripts/test_phase2_forward.py`` (slow,
needs the TinyLlama download + GPU).
"""

from __future__ import annotations

import torch
from torch import nn

from split_brain_go.adapter.projection import PerceiverResampler
from split_brain_go.env.go_env import GoEnv
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.llm.instrumented import InstrumentedLLM


# ---- mock LLM that mimics Llama's structure (same as test_instrumented) ----


class _FakeLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.fc(x) + x,)


class _FakeBaseInner(nn.Module):
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
    def __init__(self, n_layers: int = 6, d: int = 64, vocab: int = 200) -> None:
        super().__init__()

        class _Cfg:
            hidden_size = d

        self.config = _Cfg()
        self.model = _FakeBaseInner(n_layers, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.model(input_ids)
        return _FakeOutputs(self.lm_head(h))


# ============================================================ tests


def test_full_pipeline_shape():
    """End-to-end: board → Go-Net → activations → adapter → LLM → logits."""
    torch.manual_seed(0)

    # Setup
    gonet = GoNet(GoNetConfig.poc())  # 4 blocks, 64 channels
    fake_llm = _FakeBaseLLM(n_layers=6, d=64, vocab=200)
    resampler = PerceiverResampler(
        layer_channels={1: 64, 3: 64},
        n_latents=8,
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    instrumented = InstrumentedLLM(
        fake_llm, inject_layers=[2, 4], n_heads=4
    )

    # Simulate one Go state
    env = GoEnv()
    env.reset()
    env.step(40)
    env.step(20)
    board = env.encode().unsqueeze(0)  # (1, 8, 9, 9)

    # 1. Go-Net forward with activations
    policy_logits, value, acts = gonet.forward_with_acts(board, layers=[1, 3])
    assert policy_logits.shape == (1, 82)
    assert value.shape == (1,)
    assert acts[1].shape == (1, 64, 9, 9)
    assert acts[3].shape == (1, 64, 9, 9)

    # 2. Resampler
    adapter_tokens = resampler(acts)
    assert adapter_tokens.shape == (1, 8, 64)

    # 3. LLM
    input_ids = torch.randint(0, 200, (1, 5))
    logits = instrumented(input_ids, adapter_tokens=adapter_tokens)
    assert logits.shape == (1, 5, 200)
    assert torch.isfinite(logits).all()


def test_zero_init_identity_end_to_end():
    """With zero-init adapter gates, the wrapped LLM gives the same logits
    as the bare LLM. This is the critical Flamingo property — wrapping
    a frozen LLM should not change its forward output until training has
    moved the gates."""
    torch.manual_seed(1)

    gonet = GoNet(GoNetConfig.poc())
    fake_llm = _FakeBaseLLM(n_layers=6, d=64, vocab=200)
    resampler = PerceiverResampler(
        layer_channels={2: 64},
        n_latents=4,
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    instrumented = InstrumentedLLM(fake_llm, inject_layers=[1, 3], n_heads=4)

    env = GoEnv()
    env.reset()
    board = env.encode().unsqueeze(0)
    _, _, acts = gonet.forward_with_acts(board, layers=[2])
    adapter_tokens = resampler(acts)

    input_ids = torch.randint(0, 200, (1, 7))

    instrumented.eval()
    with torch.no_grad():
        wrapped_logits = instrumented(input_ids, adapter_tokens=adapter_tokens)
        bare_logits = fake_llm(input_ids).logits

    # Flamingo property: identical logits at init.
    assert torch.allclose(wrapped_logits, bare_logits, atol=1e-6)


def test_adapter_grad_flows_full_pipeline():
    """Loss backward through all three components produces gradients on
    the adapter trainables (resampler params + xattn params), and zero/None
    on the frozen LLM base."""
    torch.manual_seed(2)
    gonet = GoNet(GoNetConfig.poc())
    fake_llm = _FakeBaseLLM(n_layers=6, d=64, vocab=200)
    resampler = PerceiverResampler(
        layer_channels={2: 64},
        n_latents=4,
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )
    instrumented = InstrumentedLLM(fake_llm, inject_layers=[1, 3], n_heads=4)

    # Open one gate so cross-attn contributes
    with torch.no_grad():
        instrumented.adapter_blocks[0].gate_attn.fill_(1.0)

    env = GoEnv()
    env.reset()
    board = env.encode().unsqueeze(0)

    # Note: gonet weights are NOT trainable in Phase 2 — but its outputs
    # feed into the resampler, which IS trainable. So the chain must
    # backprop through gonet's *outputs* without crashing, even though
    # gonet's parameters won't get gradient updates.
    _, _, acts = gonet.forward_with_acts(board, layers=[2])
    adapter_tokens = resampler(acts)

    input_ids = torch.randint(0, 200, (1, 4))
    logits = instrumented(input_ids, adapter_tokens=adapter_tokens)
    loss = logits.pow(2).mean()
    loss.backward()

    # Resampler grads exist
    for name, p in resampler.named_parameters():
        assert p.grad is not None, f"resampler param {name} has no grad"

    # Adapter block grads exist
    for name, p in instrumented.adapter_blocks.named_parameters():
        assert p.grad is not None, f"adapter param {name} has no grad"

    # Base LLM grads are None (or zero)
    for name, p in instrumented.base.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"frozen base param {name} got grad"
        )


def test_different_boards_yield_different_adapter_tokens():
    """Sanity: different inputs to Go-Net → different activations →
    different adapter tokens. Confirms the pipeline isn't constant."""
    torch.manual_seed(3)
    gonet = GoNet(GoNetConfig.poc())
    resampler = PerceiverResampler(
        layer_channels={2: 64},
        n_latents=8,
        d_model=64,
        n_heads=4,
        n_blocks=1,
    )

    env_a = GoEnv()
    env_a.reset()
    env_b = GoEnv()
    env_b.reset()
    env_b.step(40)  # different state

    _, _, acts_a = gonet.forward_with_acts(env_a.encode().unsqueeze(0), layers=[2])
    _, _, acts_b = gonet.forward_with_acts(env_b.encode().unsqueeze(0), layers=[2])

    tokens_a = resampler(acts_a)
    tokens_b = resampler(acts_b)
    assert not torch.allclose(tokens_a, tokens_b, atol=1e-4)
