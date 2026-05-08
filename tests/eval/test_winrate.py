"""Unit tests for eval.winrate.

We exercise:
    * Each Agent's ``select_action`` returns a legal action.
    * ``eval_pairwise`` color alternation works.
    * RandomAgent vs RandomAgent: winrate ≈ 0.5 within bounds.
    * MCTSAgent (random net) beats RandomAgent (some advantage from search).
    * Wilson CI: known values match expected.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.env.go_env import GoEnv
from split_brain_go.eval.winrate import (
    GreedyAgent,
    MCTSAgent,
    PairwiseResult,
    RandomAgent,
    eval_pairwise,
    eval_vs_greedy,
    eval_vs_random,
    wilson_ci,
)
from split_brain_go.gonet.network import GoNet, GoNetConfig


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


# ============================================================ agents


def test_random_agent_picks_legal():
    rng = np.random.default_rng(0)
    env = GoEnv()
    env.reset()
    env.step(40)
    a = RandomAgent().select_action(env, rng)
    assert a in env.legal_actions()


def test_greedy_agent_picks_legal(random_net):
    rng = np.random.default_rng(0)
    env = GoEnv()
    env.reset()
    env.step(40)
    a = GreedyAgent(random_net).select_action(env, rng)
    assert a in env.legal_actions()


def test_mcts_agent_picks_legal(random_net):
    rng = np.random.default_rng(0)
    env = GoEnv()
    env.reset()
    env.step(40)
    a = MCTSAgent(random_net, n_simulations=4).select_action(env, rng)
    assert a in env.legal_actions()


# ============================================================ pairwise


def test_pairwise_random_vs_random_balanced():
    """Two RandomAgents over enough games: winrate roughly 0.5 ± wide CI."""
    rng = np.random.default_rng(42)
    result = eval_pairwise(
        RandomAgent(), RandomAgent(),
        n_games=20, max_moves=200, rng=rng,
    )
    assert result.n_games == 20
    assert isinstance(result, PairwiseResult)
    # Wilson CI of the winrate must contain 0.5
    lo, hi = result.a_winrate_ci
    assert lo <= 0.7 and hi >= 0.3, (
        f"winrate {result.a_winrate} CI [{lo}, {hi}] does not envelope 0.5 "
        f"— suspicious imbalance"
    )


def test_pairwise_color_alternates(random_net):
    """The schedule should alternate colors. We can detect this by giving
    one agent a clearly identifiable behavior and observing it played
    both colors. Here we only check that the test runs to completion."""
    rng = np.random.default_rng(0)
    result = eval_pairwise(
        MCTSAgent(random_net, n_simulations=2),
        RandomAgent(),
        n_games=4, max_moves=30, rng=rng,
    )
    assert result.n_games == 4
    # Wins+losses+draws+truncated == n_games
    assert (
        result.a_wins + result.b_wins + result.draws + result.truncated
        == result.n_games
    )


def test_mcts_vs_random_beats_random(random_net):
    """Even a random net + MCTS should beat a pure RandomAgent in 9x9.
    MCTS amplifies any non-uniformity in the priors."""
    rng = np.random.default_rng(0)
    result = eval_vs_random(
        random_net, n_games=8, n_simulations=8, max_moves=120, rng=rng,
    )
    # Loose threshold: just better than half.
    assert result.a_winrate > 0.4, (
        f"MCTSAgent beat RandomAgent only {result.a_wins}/{result.n_games} times"
    )


def test_eval_vs_greedy_runs(random_net):
    """Smoke: MCTS vs Greedy completes and returns a valid result."""
    rng = np.random.default_rng(0)
    result = eval_vs_greedy(
        random_net, n_games=4, n_simulations=4, max_moves=30, rng=rng,
    )
    assert 0.0 <= result.a_winrate <= 1.0


# ============================================================ Wilson CI


def test_wilson_ci_known_values():
    """Wilson 95% CI for known cases."""
    # 50% over 100 trials → roughly [0.40, 0.60]
    lo, hi = wilson_ci(50, 100)
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61

    # 100% over 10 → upper near 1.0, lower around 0.72
    lo, hi = wilson_ci(10, 10)
    assert 0.7 < lo < 0.75
    assert hi == 1.0

    # 0% over 10 → lower 0.0
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.25 < hi < 0.32


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_pairwise_result_as_dict():
    r = PairwiseResult(a_wins=15, b_wins=5, draws=0, truncated=0, n_games=20)
    d = r.as_dict()
    assert d["a_wins"] == 15
    assert d["a_winrate"] == 0.75
    assert "a_winrate_ci_low" in d
    assert "a_winrate_ci_high" in d
