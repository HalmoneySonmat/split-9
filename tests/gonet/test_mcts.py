"""Unit tests for PUCT MCTS.

Coverage targets, in order of importance:
    * sign convention in backup (most error-prone)
    * legal-action masking (visit dist never assigns to illegal actions)
    * temperature behaviour at action selection
    * Dirichlet noise effect at root
    * lazy child creation
    * terminal handling without crash
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from split_brain_go.env.go_env import GoEnv
from split_brain_go.gonet.mcts import N_ACTIONS, PASS_ACTION, MCTS, Node
from split_brain_go.gonet.network import GoNet, GoNetConfig


# ============================================================ test fixtures


class _ConstNet(nn.Module):
    """Test double: returns a fixed (logits, value)."""

    def __init__(self, value: float = 0.0, logits_seed: int = 0):
        super().__init__()
        self._value = value
        torch.manual_seed(logits_seed)
        # Random but reproducible logits
        self._logits = torch.randn(N_ACTIONS)

    def forward(self, x):  # type: ignore[override]
        b = x.shape[0]
        return self._logits.unsqueeze(0).expand(b, -1).clone(), torch.full(
            (b,), self._value
        )


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


@pytest.fixture()
def empty_env() -> GoEnv:
    e = GoEnv()
    e.reset()
    return e


# ============================================================ shape & legal


def test_search_returns_82_distribution(random_net, empty_env):
    mcts = MCTS(random_net, n_simulations=20, dirichlet_weight=0.0)
    dist = mcts.search(empty_env, add_root_noise=False)
    assert dist.shape == (N_ACTIONS,)
    assert math.isclose(dist.sum(), 1.0, abs_tol=1e-5)
    assert dist.min() >= 0


def test_search_with_random_net_terminates(random_net, empty_env):
    """Sanity: a moderately-sized search completes without infinite loop."""
    mcts = MCTS(random_net, n_simulations=50, dirichlet_weight=0.0)
    _ = mcts.search(empty_env, add_root_noise=False)


def test_legal_actions_only(random_net):
    """Visit distribution must place zero mass on occupied cells."""
    env = GoEnv()
    env.reset()
    env.step(40)  # B plays center
    env.step(20)  # W plays
    mcts = MCTS(random_net, n_simulations=30, dirichlet_weight=0.0)
    dist = mcts.search(env, add_root_noise=False)
    legal_set = set(env.legal_actions())
    for a in range(N_ACTIONS):
        if a not in legal_set:
            assert dist[a] == 0.0, f"action {a} got mass but is illegal"


# =========================================================== sign / backup


def test_node_q_property():
    n = Node(state=GoEnv())
    assert n.Q == 0.0  # visit_count == 0
    n.visit_count = 4
    n.value_sum = 2.0
    assert n.Q == 0.5


def test_backup_sign_alternates(empty_env):
    """One simulation with a network that always returns value=+1.

    Expected after the simulation:
        leaf (depth 1) gets value_sum = +1 (its own POV).
        root (depth 0) gets value_sum = -1 (sign flipped at parent).
    This is the canonical "did you flip the sign on the way up" check.
    """
    const_net = _ConstNet(value=+1.0)
    mcts = MCTS(const_net, n_simulations=1, dirichlet_weight=0.0)

    # Build tree manually so we can introspect
    root = Node(state=empty_env.clone())
    mcts._expand(root)
    mcts._simulate(root)

    assert root.visit_count == 1
    assert math.isclose(root.value_sum, -1.0, abs_tol=1e-5)

    # The single child created should have value_sum = +1.
    assert len(root.children) == 1
    child = next(iter(root.children.values()))
    assert child.visit_count == 1
    assert math.isclose(child.value_sum, 1.0, abs_tol=1e-5)


def test_backup_with_zero_value_net(empty_env):
    """If network always returns value=0, all backed-up sums stay at 0."""
    mcts = MCTS(_ConstNet(value=0.0), n_simulations=10, dirichlet_weight=0.0)
    root = Node(state=empty_env.clone())
    mcts._expand(root)
    for _ in range(10):
        mcts._simulate(root)
    assert root.visit_count == 10
    assert math.isclose(root.value_sum, 0.0, abs_tol=1e-5)


# ========================================================= temperature/select


def test_select_action_argmax_at_temp_zero(random_net, empty_env):
    """T=0 must be deterministic given identical state and seed."""
    mcts = MCTS(random_net, n_simulations=20, dirichlet_weight=0.0)
    a1, _ = mcts.select_action(
        empty_env, temperature=0.0, add_root_noise=False, rng=np.random.default_rng(42)
    )
    a2, _ = mcts.select_action(
        empty_env, temperature=0.0, add_root_noise=False, rng=np.random.default_rng(99)
    )
    # No noise + T=0 -> argmax visits, deterministic regardless of rng.
    assert a1 == a2


def test_select_action_samples_at_temp_one(random_net, empty_env):
    """T=1 with no noise: still deterministic given same rng seed; different
    seeds may give different actions."""
    mcts = MCTS(random_net, n_simulations=30, dirichlet_weight=0.0)
    seen = set()
    for seed in range(8):
        a, _ = mcts.select_action(
            empty_env,
            temperature=1.0,
            add_root_noise=False,
            rng=np.random.default_rng(seed),
        )
        seen.add(a)
    # With reasonable visit spread, at least 2 distinct actions sampled.
    assert len(seen) >= 2, f"Sampling collapsed to {seen}"


# ============================================================ root noise


def test_dirichlet_noise_changes_distribution(random_net, empty_env):
    """The same RNG seed for noise should make the noise reproducible; a
    different seed should yield a meaningfully different distribution.
    """
    mcts = MCTS(random_net, n_simulations=50, dirichlet_weight=0.5)
    dist_a = mcts.search(empty_env, add_root_noise=True, rng=np.random.default_rng(1))
    dist_b = mcts.search(empty_env, add_root_noise=True, rng=np.random.default_rng(2))
    # They should differ
    assert not np.allclose(dist_a, dist_b), "Noise had no effect"


# =========================================================== lazy expansion


def test_lazy_child_creation(random_net, empty_env):
    """After N simulations from a fresh root, root.children + descendant
    count should be at most N+1 (root + ≤N freshly-created leaves).
    """
    mcts = MCTS(random_net, n_simulations=10, dirichlet_weight=0.0)
    root = Node(state=empty_env.clone())
    mcts._expand(root)
    for _ in range(10):
        mcts._simulate(root)

    # Root's *direct* children must be ≤ n_simulations (and far below 82)
    assert len(root.children) <= 10
    # Total tree nodes
    def count(n: Node) -> int:
        return 1 + sum(count(c) for c in n.children.values())

    total = count(root)
    assert total <= 11, f"Expected ≤11 nodes, got {total}"


# ========================================================== terminal leaves


def test_terminal_handling_two_passes(random_net):
    """Set up a state where one of the legal actions immediately terminates
    the game (a pass after an existing pass). MCTS must not crash when that
    branch is selected."""
    env = GoEnv()
    env.reset()
    env.step(PASS_ACTION)  # Black passes; one more pass terminates.
    mcts = MCTS(random_net, n_simulations=20, dirichlet_weight=0.0)
    dist = mcts.search(env, add_root_noise=False)
    assert dist.shape == (N_ACTIONS,)
    # The pass action is legal; it may or may not have visits. Either is fine.


def test_strong_prior_concentrates_visits(empty_env):
    """With an informative prior, MCTS should focus visits on the preferred
    action.

    Note: with a *random* network, more simulations spread visits *more*
    broadly, not less. This is correct PUCT behaviour: the U term keeps
    pulling the search toward unvisited actions when neither prior nor Q
    differentiates them. So the meaningful test of "MCTS uses its inputs"
    is to give it a strong signal and verify it follows.
    """

    class BiasedNet(nn.Module):
        """Strongly prefers action 40 (center of the 9x9 board)."""

        def forward(self, x):  # type: ignore[override]
            b = x.shape[0]
            logits = torch.full((b, N_ACTIONS), -10.0)
            logits[:, 40] = 10.0
            return logits, torch.zeros(b)

    mcts = MCTS(BiasedNet(), n_simulations=80, dirichlet_weight=0.0)
    dist = mcts.search(empty_env, add_root_noise=False)

    # The vast majority of visits should land on the preferred action.
    assert dist[40] > 0.5, (
        f"Expected concentration on action 40 (center); got dist[40]={dist[40]}"
    )
