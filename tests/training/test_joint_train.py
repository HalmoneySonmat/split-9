"""Tests for the joint training loop.

The last test (``test_train_gonet_smoke``) is the Phase 1.3a acceptance gate:
if it passes, the full pipeline (self-play → buffer → train → eval → ckpt)
runs end-to-end without crashing and produces a real winrate.

We use a *minimal* config (1 cycle, 2 games, 5 sims) so the test runs in
under a minute on CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.training.joint_train import (
    TrainConfig,
    alphazero_loss,
    eval_vs_random,
    train_gonet,
    train_step,
)


# --------------------------------------------------------------- losses


def test_alphazero_loss_decreases_with_perfect_target():
    """If the model already predicts the target perfectly, total loss should
    be at its minimum compared to a wrong prediction."""
    B = 4
    target_policy = torch.zeros(B, 82)
    target_policy[:, 40] = 1.0  # all batches want action 40
    target_value = torch.tensor([0.5, -0.5, 0.5, -0.5])

    # Strongly correct logits -> low loss
    correct_logits = torch.full((B, 82), -10.0)
    correct_logits[:, 40] = 10.0
    correct_value = target_value.clone()
    losses_correct = alphazero_loss(
        correct_logits, correct_value, target_policy, target_value
    )

    # Random logits -> higher loss
    torch.manual_seed(0)
    bad_logits = torch.randn(B, 82)
    bad_value = torch.zeros(B)
    losses_bad = alphazero_loss(
        bad_logits, bad_value, target_policy, target_value
    )

    assert losses_correct["loss"].item() < losses_bad["loss"].item()


# ---------------------------------------------------------- train_step


def test_train_step_no_nan():
    """100 small steps must not produce NaN/Inf in any parameter."""
    torch.manual_seed(0)
    net = GoNet(GoNetConfig.poc())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    device = torch.device("cpu")

    for _ in range(100):
        boards = torch.randn(8, 8, 9, 9)
        # Soft random distributions (Dirichlet)
        policies = torch.from_numpy(
            np.random.dirichlet([1.0] * 82, size=8).astype(np.float32)
        )
        values = torch.empty(8).uniform_(-1, 1)
        train_step(net, opt, (boards, policies, values), device)

    for name, p in net.named_parameters():
        assert torch.isfinite(p).all(), f"{name} contains NaN/Inf"


# -------------------------------------------------------- eval helper


def test_eval_vs_random_returns_unit_interval():
    """eval_vs_random now returns a PairwiseResult (from eval.winrate)."""
    torch.manual_seed(0)
    net = GoNet(GoNetConfig.poc())
    result = eval_vs_random(
        net, n_games=2, n_simulations=2, max_moves=30,
        rng=np.random.default_rng(0),
    )
    assert 0.0 <= result.a_winrate <= 1.0
    assert result.n_games == 2


# ============================================================ smoke
# Phase 1.3a acceptance gate.


@pytest.mark.slow
def test_train_gonet_smoke(tmp_path):
    """Tiny end-to-end run: 1 cycle, 1 game, 2 sims, 2 train steps.

    Verifies the entire pipeline (self-play → buffer → train → eval →
    checkpoint) runs without crashing and produces a finite winrate in
    [0, 1]. This is the Phase 1.3a acceptance gate.
    """
    cfg = TrainConfig(
        seed=42,
        n_blocks=1,
        channels=16,
        n_cycles=1,
        games_per_cycle=1,
        train_steps_per_cycle=2,
        n_simulations=2,
        batch_size=2,
        min_buffer=2,
        max_moves=20,
        eval_n_games=1,
        eval_n_simulations=2,
        eval_max_moves=20,
        log_every=1,
        checkpoint_dir=str(tmp_path),
        log_to="noop",
        eval_against_greedy=False,  # speed up smoke
    )
    result = train_gonet(cfg)
    assert isinstance(result, dict)
    assert "winrate" in result and "loss" in result
    assert 0.0 <= result["winrate"] <= 1.0
    assert math.isfinite(result["loss"])

    # A checkpoint file should exist
    saved = list(tmp_path.glob("step_*.pt"))
    assert len(saved) == 1
    assert (tmp_path / "best.pt").exists()


@pytest.mark.slow
def test_train_gonet_smoke_batched(tmp_path):
    """Same end-to-end gate as the smoke test, but with batched self-play.

    Verifies the use_batched=True branch of train_gonet runs without
    crashing and produces the same output shape.
    """
    cfg = TrainConfig(
        seed=42,
        n_blocks=1,
        channels=16,
        n_cycles=1,
        games_per_cycle=2,         # batched chunk = both games at once
        train_steps_per_cycle=2,
        n_simulations=2,
        batch_size=2,
        min_buffer=2,
        max_moves=20,
        eval_n_games=1,
        eval_n_simulations=2,
        eval_max_moves=20,
        log_every=1,
        checkpoint_dir=str(tmp_path),
        use_batched=True,
        batch_n_games=4,
        log_to="noop",
    )
    result = train_gonet(cfg)
    assert isinstance(result, dict)
    assert 0.0 <= result["winrate"] <= 1.0
    assert math.isfinite(result["loss"])


@pytest.mark.slow
def test_train_gonet_writes_csv_log(tmp_path):
    """When log_to='csv', a metrics.csv with cycle/* columns must appear."""
    log_dir = tmp_path / "runs"
    cfg = TrainConfig(
        seed=42,
        n_blocks=1,
        channels=16,
        n_cycles=2,
        games_per_cycle=1,
        train_steps_per_cycle=2,
        n_simulations=2,
        batch_size=2,
        min_buffer=2,
        max_moves=20,
        eval_n_games=1,
        eval_n_simulations=2,
        eval_max_moves=20,
        log_every=1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_to="csv",
        log_run_name="csv-smoke",
        log_dir=str(log_dir),
    )
    train_gonet(cfg)

    metrics_csv = log_dir / "csv-smoke" / "metrics.csv"
    config_json = log_dir / "csv-smoke" / "config.json"
    assert metrics_csv.is_file()
    assert config_json.is_file()

    import csv as _csv

    with metrics_csv.open() as f:
        rows = list(_csv.DictReader(f))
    # Two cycles → two rows
    assert len(rows) == 2
    assert "cycle/winrate_vs_random" in rows[0]
    assert "cycle/loss" in rows[0]
