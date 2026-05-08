"""Unit tests for self-play.

These run a few short games with a tiny PoC GoNet — no GPU needed, takes
under a minute total. Not a smoke test for *quality* (random net plays
random Go); only for *correctness of the data pipeline*.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.gonet.mcts import N_ACTIONS
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.training.selfplay import (
    SelfPlayGame,
    TrainingExample,
    default_temperature,
    play_one_game,
)


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


# ----------------------------------------------------------------- shapes


def test_play_one_game_terminates(random_net):
    """Random net must produce a finished SelfPlayGame within max_moves."""
    game = play_one_game(random_net, n_simulations=10, max_moves=200)
    assert isinstance(game, SelfPlayGame)
    assert len(game.examples) > 0
    assert len(game.move_history) == len(game.examples)


def test_examples_have_correct_shapes(random_net):
    game = play_one_game(random_net, n_simulations=10, max_moves=50)
    for ex in game.examples:
        assert isinstance(ex, TrainingExample)
        assert ex.observation.shape == (8, 9, 9)
        assert ex.observation.dtype == torch.float32
        assert ex.policy_target.shape == (N_ACTIONS,)
        assert ex.policy_target.dtype == np.float32
        # Distribution should sum to ~1 (visits normalized)
        assert abs(float(ex.policy_target.sum()) - 1.0) < 1e-4
        assert isinstance(ex.value_target, float)
        assert -1.0 <= ex.value_target <= 1.0


# ----------------------------------------------------------- value targets


def test_value_targets_alternate_with_outcome(random_net):
    """Black moves at even indices (0, 2, ...); value_target should follow
    returns[mover]. So if Black wins, even indices have value=+1, odd=-1.
    """
    game = play_one_game(
        random_net,
        n_simulations=10,
        max_moves=200,
        rng=np.random.default_rng(42),
    )
    if game.truncated or game.returns == (0.0, 0.0):
        pytest.skip("Game truncated or drew; cannot test alternation.")

    for i, ex in enumerate(game.examples):
        mover = i % 2  # 0=Black, 1=White (Black always starts in 9x9 Go)
        expected = game.returns[mover]
        assert ex.value_target == expected, (
            f"Move {i} (mover={mover}): expected {expected}, got {ex.value_target}"
        )


# ----------------------------------------------------------- temperature


def test_temperature_schedule_zero_after_30():
    assert default_temperature(0) == 1.0
    assert default_temperature(15) == 1.0
    assert default_temperature(29) == 1.0
    assert default_temperature(30) == 0.0
    assert default_temperature(100) == 0.0


# ------------------------------------------------------- terminal handling


def test_passes_in_late_game_handled(random_net):
    """Even if the game ends via two passes, no crash."""
    game = play_one_game(
        random_net,
        n_simulations=10,
        max_moves=200,
        rng=np.random.default_rng(7),
    )
    # Game must complete; returns must be a real tuple.
    assert isinstance(game.returns, tuple)
    assert len(game.returns) == 2
    # If terminal, returns sum should be 0 (one wins, one loses) or 0/0 (draw)
    if not game.truncated:
        a, b = game.returns
        assert {a, b} <= {-1.0, 0.0, 1.0}, f"Unexpected returns: {(a, b)}"
