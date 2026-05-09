"""Tests for Phase 3 adapter training loop, using mocks throughout.

We mock:
    * GoNet — returns dummy activations of correct shape
    * Resampler — small linear projection
    * InstrumentedLLM — returns random logits

This tests the *training loop logic* (loss masking, gradient flow,
optimizer step, evaluation) without GPU or 1B-parameter models.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from split_brain_go.data.dataset import LABEL_IGNORE, Phase3Dataset, TokenizedExample, phase3_collate
from split_brain_go.training.adapter_train import (
    AdapterTrainConfig,
    adapter_loss,
    evaluate,
    train_adapter,
    train_step,
    trainable_adapter_params,
)


# ---------------------------------------------------- mock components


class _MockGoNet(nn.Module):
    """Returns dummy activations of expected shape."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        # Frozen "weights" so the test doesn't accidentally see grads.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.channels = channels

    def forward_with_acts(self, board, layers):
        B = board.shape[0]
        out = (
            torch.zeros(B, 82, device=board.device),
            torch.zeros(B, device=board.device),
            {l: torch.randn(B, self.channels, 9, 9, device=board.device) for l in layers},
        )
        return out


class _MockResampler(nn.Module):
    """Project per-layer (B, C, 9, 9) → mean over space → linear to d_model."""

    def __init__(self, layer_ids: list[int], channels: int, n_latents: int, d_model: int) -> None:
        super().__init__()
        self.layer_ids = sorted(layer_ids)
        self.n_latents = n_latents
        self.proj = nn.Linear(channels, d_model)
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))

    def forward(self, acts):
        # Average each layer's activation over space, sum across layers, broadcast to N latents.
        layer_means = []
        for lid in self.layer_ids:
            x = acts[lid]  # (B, C, 9, 9)
            layer_means.append(x.mean(dim=(2, 3)))
        avg = torch.stack(layer_means, dim=0).mean(dim=0)  # (B, C)
        z = self.proj(avg).unsqueeze(1)  # (B, 1, d_model)
        return z.expand(-1, self.n_latents, -1).contiguous() + self.latents


class _MockInstrumentedLLM(nn.Module):
    """Tiny LM that uses adapter_tokens to influence logits.

    Has a synthetic ``adapter_blocks`` ModuleList for the optimizer to
    pick up (we stuff a dummy linear layer into it as a stand-in for
    the real cross-attention blocks)."""

    def __init__(self, vocab: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        # Adapter-side trainables
        self.adapter_blocks = nn.ModuleList([nn.Linear(d_model, d_model)])
        # Frozen "base" — pretend
        self.base_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.training_flag = True

    def forward(self, input_ids, adapter_tokens=None, attention_mask=None):
        x = self.embed(input_ids)  # (B, T, D)
        if adapter_tokens is not None:
            # Mix in a single adapter token's signal (averaged across N).
            x = x + self.adapter_blocks[0](adapter_tokens.mean(dim=1, keepdim=True))
        return self.lm_head(x)


# ---------------------------------------------------- helpers


def _make_loader(n_examples: int = 8, T: int = 16, V: int = 50, batch_size: int = 2) -> DataLoader:
    tokenized = []
    for _ in range(n_examples):
        input_ids = torch.randint(0, V, (T,), dtype=torch.long)
        attn = torch.ones(T, dtype=torch.long)
        labels = input_ids.clone()
        labels[:4] = LABEL_IGNORE  # mask first 4 tokens (prompt-like)
        tokenized.append(
            TokenizedExample(
                board=torch.randn(8, 9, 9),
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
            )
        )
    return DataLoader(Phase3Dataset(tokenized), batch_size=batch_size, collate_fn=phase3_collate)


# ============================================================ loss


def test_adapter_loss_returns_scalar():
    logits = torch.randn(2, 8, 50)
    labels = torch.randint(0, 50, (2, 8))
    labels[:, :3] = LABEL_IGNORE
    loss = adapter_loss(logits, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


def test_adapter_loss_ignores_minus_100():
    """When a position the loss actually consumes is masked, loss changes.

    Note: ``adapter_loss`` shifts internally (predict t+1 from t), so
    masking ``labels[:, 0]`` has no effect — position 0 is dropped by the
    shift. We must mask a position that survives the shift, i.e. index ≥ 1.
    """
    logits = torch.randn(1, 6, 5)
    labels_a = torch.tensor([[0, 1, 2, 3, 4, 0]])
    labels_b = labels_a.clone()
    labels_b[0, 2] = LABEL_IGNORE  # mask a post-shift position
    loss_a = adapter_loss(logits, labels_a).item()
    loss_b = adapter_loss(logits, labels_b).item()
    assert loss_a != loss_b


def test_adapter_loss_position_0_mask_is_noop():
    """Masking position 0 should NOT change loss (it's dropped by shift)."""
    logits = torch.randn(1, 6, 5)
    labels_a = torch.tensor([[0, 1, 2, 3, 4, 0]])
    labels_b = labels_a.clone()
    labels_b[0, 0] = LABEL_IGNORE
    loss_a = adapter_loss(logits, labels_a).item()
    loss_b = adapter_loss(logits, labels_b).item()
    assert loss_a == loss_b


# ============================================================ train_step


def test_train_step_runs_and_decreases_loss():
    """Single-batch overfitting: with the same batch repeated, loss should drop."""
    torch.manual_seed(0)
    V, D = 50, 32
    gonet = _MockGoNet(channels=16)
    resampler = _MockResampler(layer_ids=[1, 3], channels=16, n_latents=4, d_model=D)
    llm = _MockInstrumentedLLM(vocab=V, d_model=D)
    params = trainable_adapter_params(llm, resampler)
    opt = torch.optim.Adam(params, lr=1e-2)

    loader = _make_loader(n_examples=2, T=12, V=V, batch_size=2)
    batch = next(iter(loader))

    initial = train_step(
        llm, gonet, resampler, opt, batch, [1, 3], torch.device("cpu"),
        adapter_dtype=torch.float32,
    )
    losses = []
    for _ in range(50):
        m = train_step(
            llm, gonet, resampler, opt, batch, [1, 3], torch.device("cpu"),
            adapter_dtype=torch.float32,
        )
        losses.append(m["loss"])
    # Final loss should be substantially below initial
    assert losses[-1] < initial["loss"], (
        f"Loss did not decrease: initial={initial['loss']:.3f}, final={losses[-1]:.3f}"
    )


def test_train_step_returns_finite_metrics():
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1], 16, 4, 32)
    llm = _MockInstrumentedLLM(50, 32)
    opt = torch.optim.Adam(trainable_adapter_params(llm, resampler), lr=1e-3)
    batch = next(iter(_make_loader(n_examples=2, T=8, V=50, batch_size=2)))
    metrics = train_step(
        llm, gonet, resampler, opt, batch, [1], torch.device("cpu"),
        adapter_dtype=torch.float32,
    )
    import math
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["perplexity"])
    assert metrics["perplexity"] > 1.0


# ============================================================ evaluate


def test_evaluate_returns_loss_and_perplexity():
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1], 16, 4, 32)
    llm = _MockInstrumentedLLM(50, 32)
    val_loader = _make_loader(n_examples=4, T=8, V=50, batch_size=2)
    metrics = evaluate(
        llm, gonet, resampler, val_loader, [1], torch.device("cpu"),
        max_batches=10, adapter_dtype=torch.float32,
    )
    assert "val_loss" in metrics
    assert "val_perplexity" in metrics
    assert 0 < metrics["val_loss"] < 100


# ============================================================ full loop


def test_train_adapter_smoke(tmp_path):
    """Tiny end-to-end run: 1 epoch, few batches, tmp_path for ckpts."""
    torch.manual_seed(0)
    gonet = _MockGoNet(16)
    resampler = _MockResampler([1, 3], 16, 4, 32)
    llm = _MockInstrumentedLLM(50, 32)

    train_loader = _make_loader(n_examples=4, T=8, V=50, batch_size=2)
    val_loader = _make_loader(n_examples=2, T=8, V=50, batch_size=1)

    cfg = AdapterTrainConfig(
        seed=0,
        act_layers=[1, 3],
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        log_every=1,
        eval_every=2,
        eval_max_batches=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_to="noop",
        adapter_dtype="float32",
    )
    final = train_adapter(
        llm, gonet, resampler, train_loader, val_loader, cfg,
        device=torch.device("cpu"),
    )
    assert "train_loss" in final
    # We don't check exact values — just that the loop completed.
    import math
    assert math.isfinite(final["train_loss"])
