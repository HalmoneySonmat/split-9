"""Unit tests for BatchedMCTS.

Two test categories:
    * *Equivalence*: with N=1 and no root noise, BatchedMCTS must match the
      single MCTS exactly. With identical RNG seeds, distributions agree.
    * *Independence*: with N>1, each game's tree is independent; differing
      states yield differing distributions; identical states with identical
      RNG yield identical distributions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from split_brain_go.env.go_env import GoEnv
from split_brain_go.gonet.batched_mcts import BatchedMCTS
from split_brain_go.gonet.mcts import N_ACTIONS, MCTS
from split_brain_go.gonet.network import GoNet, GoNetConfig


# ============================================================ fixtures


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


def _empty_envs(n: int) -> list[GoEnv]:
    out = []
    for _ in range(n):
        e = GoEnv()
        e.reset()
        out.append(e)
    return out


# ============================================================== shapes


def test_search_batch_shapes(random_net):
    bmcts = BatchedMCTS(random_net, n_simulations=8, dirichlet_weight=0.0)
    dists = bmcts.search_batch(_empty_envs(4), add_root_noise=False)
    assert isinstance(dists, list)
    assert len(dists) == 4
    for d in dists:
        assert d.shape == (N_ACTIONS,)
        assert math.isclose(float(d.sum()), 1.0, abs_tol=1e-4)
        assert d.min() >= 0


def test_search_batch_legal_only(random_net):
    """No mass on illegal actions."""
    env = GoEnv()
    env.reset()
    env.step(40)
    bmcts = BatchedMCTS(random_net, n_simulations=12, dirichlet_weight=0.0)
    [dist] = bmcts.search_batch([env], add_root_noise=False)
    legal = set(env.legal_actions())
    for a in range(N_ACTIONS):
        if a not in legal:
            assert dist[a] == 0.0


# ===================================================== equivalence (N=1)


def test_n1_visit_count_matches_single_mcts(random_net):
    """BatchedMCTS(N=1, no noise) and MCTS(no noise) should produce identical
    visit distributions on the same input.

    Both implementations are deterministic given a fixed network and no
    Dirichlet noise — they descend the same paths in the same order.
    """
    env = GoEnv()
    env.reset()

    single = MCTS(random_net, n_simulations=20, dirichlet_weight=0.0)
    batched = BatchedMCTS(random_net, n_simulations=20, dirichlet_weight=0.0)

    dist_single = single.search(env, add_root_noise=False)
    [dist_batched] = batched.search_batch([env], add_root_noise=False)

    np.testing.assert_allclose(dist_single, dist_batched, atol=1e-5)


# =========================================================== independence


def test_different_envs_give_different_distributions(random_net):
    """Two distinct positions should produce different distributions."""
    env_a = GoEnv()
    env_a.reset()
    env_b = GoEnv()
    env_b.reset()
    env_b.step(40)  # Black plays center — different state

    bmcts = BatchedMCTS(random_net, n_simulations=16, dirichlet_weight=0.0)
    dists = bmcts.search_batch([env_a, env_b], add_root_noise=False)

    # The two distributions must differ on at least one action
    assert not np.allclose(dists[0], dists[1])


def test_identical_envs_with_same_rng_match(random_net):
    """Two identical envs with no noise should give identical distributions."""
    envs = _empty_envs(2)
    bmcts = BatchedMCTS(random_net, n_simulations=12, dirichlet_weight=0.0)
    dists = bmcts.search_batch(envs, add_root_noise=False)
    np.testing.assert_allclose(dists[0], dists[1], atol=1e-5)


# ============================================================ terminals


def test_mixed_terminal_and_active_in_batch(random_net):
    """Two games where one is one-pass-from-terminal: a mid-batch step may
    produce a terminal leaf for game A while game B's leaf is active. The
    code must handle the mismatch (forward only over non-terminal leaves)
    without crashing.
    """
    env_a = GoEnv()
    env_a.reset()
    env_a.step(81)  # Black passes — next move can also pass to terminate

    env_b = GoEnv()
    env_b.reset()  # Fresh, far from terminal

    bmcts = BatchedMCTS(random_net, n_simulations=16, dirichlet_weight=0.0)
    dists = bmcts.search_batch([env_a, env_b], add_root_noise=False)
    assert dists[0].shape == (N_ACTIONS,)
    assert dists[1].shape == (N_ACTIONS,)


def test_root_noise_makes_results_differ(random_net):
    """Same env, two different RNG seeds for noise → different distributions."""
    [env] = _empty_envs(1)
    bmcts = BatchedMCTS(random_net, n_simulations=20, dirichlet_weight=0.5)
    [d1] = bmcts.search_batch(
        [env], add_root_noise=True, rng=np.random.default_rng(1)
    )
    [d2] = bmcts.search_batch(
        [env], add_root_noise=True, rng=np.random.default_rng(2)
    )
    assert not np.allclose(d1, d2)
