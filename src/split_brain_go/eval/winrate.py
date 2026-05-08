"""Winrate evaluation — pairwise games between two agents.

The plan_review (and our PoC training results) showed that *vs Random* is
not a sensitive enough indicator of learning progress: a randomly-init'd
network with MCTS already beats Random ~80% of the time, so the metric
saturates early. This module provides the broader baselines:

    RandomAgent  — uniform random over legal moves.
    GreedyAgent  — argmax of the policy head (no MCTS, no value head).
    MCTSAgent    — full MCTS search over a network.

The interesting comparisons are then:
    eval_pairwise(MCTSAgent(net), GreedyAgent(net))   ← MCTS lift
    eval_pairwise(MCTSAgent(curr), MCTSAgent(prev))   ← cross-checkpoint ELO

Color is *alternated* per game so neither side benefits from komi or
first-move advantage. Wilson 95% CI accompanies the point estimate.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..env.go_env import GoEnv
from ..gonet.mcts import MCTS, N_ACTIONS

if TYPE_CHECKING:
    from ..gonet.network import GoNet


# ============================================================ Agents


class Agent(ABC):
    """Selects a legal action given a GoEnv. Stateless w.r.t. game tree."""

    @abstractmethod
    def select_action(self, env: GoEnv, rng: np.random.Generator) -> int:
        ...


class RandomAgent(Agent):
    """Uniform random over legal actions."""

    def select_action(self, env: GoEnv, rng: np.random.Generator) -> int:
        return int(rng.choice(env.legal_actions()))


class GreedyAgent(Agent):
    """Argmax of the policy head over legal actions. No MCTS, no value."""

    def __init__(self, network: "GoNet", device: torch.device | None = None) -> None:
        self.network = network
        self.device = device or next(network.parameters()).device

    def select_action(self, env: GoEnv, rng: np.random.Generator) -> int:
        obs = env.encode().unsqueeze(0).to(self.device)
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                logits, _ = self.network(obs)
        finally:
            self.network.train(was_training)

        probs = F.softmax(logits[0], dim=-1).cpu().numpy()
        legal = env.legal_actions()
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_mask[legal] = 1.0
        masked = probs * legal_mask
        if masked.sum() == 0:
            return int(rng.choice(legal))
        return int(np.argmax(masked))


class MCTSAgent(Agent):
    """MCTS with the given network. ``add_root_noise=False`` for evaluation."""

    def __init__(
        self,
        network: "GoNet",
        n_simulations: int = 200,
        c_puct: float = 1.5,
        device: torch.device | None = None,
    ) -> None:
        self.network = network
        self.mcts = MCTS(
            network,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_weight=0.0,
            device=device or next(network.parameters()).device,
        )

    def select_action(self, env: GoEnv, rng: np.random.Generator) -> int:
        action, _ = self.mcts.select_action(
            env, temperature=0.0, add_root_noise=False, rng=rng
        )
        return action


# ============================================================ pairwise


@dataclass
class PairwiseResult:
    a_wins: int
    b_wins: int
    draws: int
    truncated: int
    n_games: int

    @property
    def a_winrate(self) -> float:
        if self.n_games == 0:
            return 0.0
        return self.a_wins / self.n_games

    @property
    def a_winrate_ci(self) -> tuple[float, float]:
        return wilson_ci(self.a_wins, self.n_games)

    def as_dict(self) -> dict[str, float | int]:
        lo, hi = self.a_winrate_ci
        return {
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "truncated": self.truncated,
            "n_games": self.n_games,
            "a_winrate": self.a_winrate,
            "a_winrate_ci_low": lo,
            "a_winrate_ci_high": hi,
        }


def eval_pairwise(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int = 20,
    max_moves: int = 200,
    rng: np.random.Generator | None = None,
) -> PairwiseResult:
    """Play ``n_games`` games, alternating colors. Return aggregate result.

    Color schedule:
        Even index: agent_a is Black (moves first), agent_b is White.
        Odd index : agent_a is White, agent_b is Black.

    A truncated game (hit ``max_moves`` without natural termination) counts
    as neither side winning — incremented in ``truncated`` but excluded
    from ``a_wins`` / ``b_wins``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    a_wins = 0
    b_wins = 0
    draws = 0
    truncated = 0

    for game_idx in range(n_games):
        env = GoEnv()
        env.reset()
        # Even idx → A=Black(0); odd → A=White(1)
        a_color = game_idx % 2
        b_color = 1 - a_color

        moves = 0
        while not env.is_terminal() and moves < max_moves:
            actor = agent_a if env.current_player() == a_color else agent_b
            env.step(actor.select_action(env, rng))
            moves += 1

        if not env.is_terminal():
            truncated += 1
            continue

        r = env.returns()
        a_score = r[a_color]
        b_score = r[b_color]
        if a_score > b_score:
            a_wins += 1
        elif b_score > a_score:
            b_wins += 1
        else:
            draws += 1

    return PairwiseResult(
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        truncated=truncated,
        n_games=n_games,
    )


# ============================================================ shortcuts


def eval_vs_random(
    network: "GoNet",
    n_games: int = 20,
    n_simulations: int = 50,
    max_moves: int = 200,
    rng: np.random.Generator | None = None,
) -> PairwiseResult:
    return eval_pairwise(
        MCTSAgent(network, n_simulations=n_simulations),
        RandomAgent(),
        n_games=n_games,
        max_moves=max_moves,
        rng=rng,
    )


def eval_vs_greedy(
    network: "GoNet",
    n_games: int = 20,
    n_simulations: int = 50,
    max_moves: int = 200,
    rng: np.random.Generator | None = None,
) -> PairwiseResult:
    """The MCTS lift over a non-search policy. Sensitive to learning progress."""
    return eval_pairwise(
        MCTSAgent(network, n_simulations=n_simulations),
        GreedyAgent(network),
        n_games=n_games,
        max_moves=max_moves,
        rng=rng,
    )


def eval_vs_checkpoint(
    network: "GoNet",
    opponent_network: "GoNet",
    n_games: int = 20,
    n_simulations: int = 50,
    max_moves: int = 200,
    rng: np.random.Generator | None = None,
) -> PairwiseResult:
    """Current network vs a previous checkpoint, both with MCTS."""
    return eval_pairwise(
        MCTSAgent(network, n_simulations=n_simulations),
        MCTSAgent(opponent_network, n_simulations=n_simulations),
        n_games=n_games,
        max_moves=max_moves,
        rng=rng,
    )


# ============================================================ stats


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion.

    Returns (low, high) clipped to [0, 1]. For the standard 95% CI,
    ``z`` ≈ 1.96. We compute z from the normal quantile.
    """
    if n <= 0:
        return 0.0, 1.0
    # Inverse CDF of standard normal at (1 + confidence)/2.
    # For 0.95 we want 0.975 quantile → 1.959963985.
    # Pre-computed table for common values, fallback to scipy if available.
    z = _normal_quantile((1 + confidence) / 2)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _normal_quantile(p: float) -> float:
    """Inverse normal CDF for common confidence levels.

    For exact values use scipy.stats.norm.ppf; here we hard-code the
    typical ones to avoid the scipy dependency just for one number.
    """
    table = {
        0.975: 1.959963985,  # 95% CI
        0.995: 2.575829304,  # 99% CI
        0.95: 1.644853627,   # one-sided 90%
        0.99: 2.326347874,   # one-sided 98%
    }
    if p in table:
        return table[p]
    # Fallback: try scipy.
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        # Crude approximation; only used for nonstandard confidence levels.
        return 1.96


__all__ = [
    "Agent",
    "RandomAgent",
    "GreedyAgent",
    "MCTSAgent",
    "PairwiseResult",
    "eval_pairwise",
    "eval_vs_random",
    "eval_vs_greedy",
    "eval_vs_checkpoint",
    "wilson_ci",
]
