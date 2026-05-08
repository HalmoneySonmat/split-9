"""Tests for play_batched_games (batched self-play)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.gonet.mcts import N_ACTIONS
from split_brain_go.gonet.network import GoNet, GoNetConfig
from split_brain_go.training.selfplay import (
    SelfPlayGame,
    _sample_action_from_dist,
    play_batched_games,
)


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


# ---------------------------------------------------- sampling helper


def test_sample_argmax_at_temp_zero():
    dist = np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 78, dtype=np.float32)
    rng = np.random.default_rng(0)
    a = _sample_action_from_dist(dist, temperature=0.0, rng=rng)
    assert a == 1


def test_sample_distribution_at_temp_one():
    """T=1 should sample roughly proportional to the distribution."""
    dist = np.zeros(N_ACTIONS, dtype=np.float32)
    dist[10] = 0.5
    dist[20] = 0.5
    rng = np.random.default_rng(0)
    counts = {10: 0, 20: 0, "other": 0}
    for _ in range(200):
        a = _sample_action_from_dist(dist, temperature=1.0, rng=rng)
        if a in counts:
            counts[a] += 1
        else:
            counts["other"] += 1
    # Both target actions hit; "other" stays zero (mass is zero elsewhere)
    assert counts[10] > 50
    assert counts[20] > 50
    assert counts["other"] == 0


# ---------------------------------------------------- batched games


def test_returns_n_games(random_net):
    games = play_batched_games(
        random_net,
        n_games=3,
        n_simulations=4,
        max_moves=20,
        rng=np.random.default_rng(0),
    )
    assert len(games) == 3
    for g in games:
        assert isinstance(g, SelfPlayGame)


def test_each_game_has_examples(random_net):
    games = play_batched_games(
        random_net,
        n_games=2,
        n_simulations=4,
        max_moves=30,
        rng=np.random.default_rng(0),
    )
    for g in games:
        assert len(g.examples) > 0
        for ex in g.examples:
            assert ex.observation.shape == (8, 9, 9)
            assert ex.policy_target.shape == (N_ACTIONS,)
            assert -1.0 <= ex.value_target <= 1.0


def test_independent_game_outcomes(random_net):
    """With different RNGs, two batched-game collections produce different
    move histories (sampling has noise from temperature + Dirichlet)."""
    g1 = play_batched_games(
        random_net,
        n_games=2,
        n_simulations=4,
        max_moves=20,
        rng=np.random.default_rng(1),
    )
    g2 = play_batched_games(
        random_net,
        n_games=2,
        n_simulations=4,
        max_moves=20,
        rng=np.random.default_rng(99),
    )
    # At least one of the four games (2 from each call) should differ
    h_all_1 = [g.move_history for g in g1]
    h_all_2 = [g.move_history for g in g2]
    differ = any(a != b for a, b in zip(h_all_1, h_all_2))
    assert differ


def test_truncated_games_have_zero_returns(random_net):
    """If max_moves is small, some games will be truncated → returns=(0,0)."""
    games = play_batched_games(
        random_net,
        n_games=2,
        n_simulations=2,
        max_moves=3,  # very short, will truncate
        rng=np.random.default_rng(0),
    )
    truncated = [g for g in games if g.truncated]
    # All games should be truncated at max_moves=3 on 9x9
    for g in truncated:
        assert g.returns == (0.0, 0.0)
        for ex in g.examples:
            assert ex.value_target == 0.0
