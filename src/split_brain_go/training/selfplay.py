"""Self-play game runner.

Plays one game of 9x9 Go with MCTS-guided moves, recording per-move data
that becomes training examples for the next iteration of GoNet learning.

Per-move data captured during the game:
    (board observation, MCTS visit distribution, mover identity)

After the game ends, value targets are *backfilled* from the final game
result so that each example carries (obs, policy_target, value_target).
This is the standard AlphaZero data shape.

The game terminates either by natural end (two passes / no legal moves)
or by ``max_moves`` cap. Truncated games carry value_target=0 (treated
as a draw) — this is rare in 9x9 with reasonable models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..env.go_env import GoEnv
from ..gonet.batched_mcts import BatchedMCTS
from ..gonet.mcts import MCTS, N_ACTIONS

if TYPE_CHECKING:
    from ..gonet.network import GoNet


# ============================================================ data classes


@dataclass
class TrainingExample:
    """One (state, target) pair from a self-play game.

    The value target is *from the moving player's perspective* at this state:
    +1 means the player to move went on to win, -1 lost, 0 drew or truncated.
    """

    observation: torch.Tensor          # (8, 9, 9) float32
    policy_target: np.ndarray          # (82,) float32, sums to 1
    value_target: float                # in [-1, 1]


@dataclass
class SelfPlayGame:
    """Output of one self-play game."""

    examples: list[TrainingExample] = field(default_factory=list)
    move_history: list[int] = field(default_factory=list)
    returns: tuple[float, float] = (0.0, 0.0)  # (black_outcome, white_outcome)
    truncated: bool = False  # True if game hit max_moves cap


# ========================================================== temperature


def default_temperature(move_idx: int) -> float:
    """Standard schedule: T=1 (sample) for first 30 moves, then T=0 (argmax).

    Early-game randomness produces diverse openings (data variety), late-game
    determinism produces accurate end-game play (clean value targets).
    """
    return 1.0 if move_idx < 30 else 0.0


# =============================================================== runner


def _network_device(network: torch.nn.Module) -> torch.device:
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def play_one_game(
    network: "GoNet",
    n_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.25,
    dirichlet_weight: float = 0.25,
    temperature_schedule: Callable[[int], float] = default_temperature,
    max_moves: int = 200,
    device: str | torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> SelfPlayGame:
    """Run one self-play game and return its training data.

    Args:
        network: A GoNet (or compatible). Need not be in eval mode; MCTS
            handles mode switching.
        n_simulations: MCTS rollouts per move. ADR-012 PoC=100 / full=200.
        c_puct, dirichlet_alpha, dirichlet_weight: PUCT params (ADR-011).
        temperature_schedule: ``move_idx -> T`` for action sampling.
        max_moves: Safety cap. 9x9 games rarely exceed ~120 moves.
        device: Where to run the network. Default: infer from network.
        rng: NumPy random Generator. Deterministic if seeded.
    """
    rng = rng if rng is not None else np.random.default_rng()
    device = device if device is not None else _network_device(network)

    env = GoEnv()
    env.reset()
    mcts = MCTS(
        network,
        n_simulations=n_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
        device=device,
    )

    # Stage 1: play through the game, recording (obs, dist, mover) per move.
    pre_value: list[tuple[torch.Tensor, np.ndarray, int]] = []
    history: list[int] = []

    while not env.is_terminal() and len(pre_value) < max_moves:
        mover = env.current_player()
        obs = env.encode()
        T = temperature_schedule(len(pre_value))
        action, visit_dist = mcts.select_action(
            env, temperature=T, add_root_noise=True, rng=rng
        )
        pre_value.append((obs, visit_dist, mover))
        history.append(action)
        env.step(action)

    # Stage 2: backfill value targets from the final game outcome.
    truncated = not env.is_terminal()
    if truncated:
        # No winner; use 0 as the target for all examples. Still valid training
        # signal (the value head will learn that ambiguous states ≈ 0).
        returns: tuple[float, float] = (0.0, 0.0)
    else:
        r0, r1 = env.returns()
        returns = (float(r0), float(r1))

    examples = [
        TrainingExample(
            observation=obs,
            policy_target=dist.astype(np.float32, copy=False),
            value_target=returns[mover],
        )
        for (obs, dist, mover) in pre_value
    ]

    return SelfPlayGame(
        examples=examples,
        move_history=history,
        returns=returns,
        truncated=truncated,
    )


# ============================================================ batched runner


def _sample_action_from_dist(
    dist: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    """Sample an action from a visit distribution with temperature.

    T <= 0 → argmax. T > 0 → sample proportional to ``dist ** (1/T)``.
    Falls back to a uniform sample over legal (positive-mass) actions if
    the distribution collapses to zero — shouldn't happen in practice.
    """
    if temperature <= 0:
        return int(np.argmax(dist))
    transformed = dist ** (1.0 / temperature)
    total = transformed.sum()
    if total <= 0:
        legal_idx = np.flatnonzero(dist > 0)
        if len(legal_idx) > 0:
            return int(rng.choice(legal_idx))
        return int(rng.integers(0, len(dist)))
    probs = transformed / total
    return int(rng.choice(len(dist), p=probs))


def play_batched_games(
    network: "GoNet",
    n_games: int = 8,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.25,
    dirichlet_weight: float = 0.25,
    temperature_schedule: Callable[[int], float] = default_temperature,
    max_moves: int = 200,
    device: str | torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> list[SelfPlayGame]:
    """Play ``n_games`` parallel self-play games using BatchedMCTS.

    All games progress in lockstep, one move at a time. At each move, the
    BatchedMCTS searches every still-active game in one batched forward.
    Games that hit terminal drop out; the loop continues until all games
    end or every game reaches ``max_moves``.

    Returns a list of ``n_games`` ``SelfPlayGame`` instances in the same
    order as the games were created.

    Notes:
        * Throughput on RTX 3070 Ti is roughly 4–6× faster than calling
          ``play_one_game`` ``n_games`` times back-to-back, with N=8.
        * Each game's visit distribution is the same as it would be from
          a single MCTS at N=1 (verified by ``test_batched_mcts``).
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive")
    rng = rng if rng is not None else np.random.default_rng()
    if device is None:
        device = _network_device(network)

    envs = [GoEnv() for _ in range(n_games)]
    for e in envs:
        e.reset()

    bmcts = BatchedMCTS(
        network,
        n_simulations=n_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
        device=device,
    )

    pre_value: list[list[tuple[torch.Tensor, np.ndarray, int]]] = [
        [] for _ in range(n_games)
    ]
    histories: list[list[int]] = [[] for _ in range(n_games)]
    move_count = 0

    while move_count < max_moves:
        active_indices = [i for i in range(n_games) if not envs[i].is_terminal()]
        if not active_indices:
            break

        active_envs = [envs[i] for i in active_indices]
        dists = bmcts.search_batch(active_envs, add_root_noise=True, rng=rng)

        T = temperature_schedule(move_count)
        for idx, dist in zip(active_indices, dists):
            mover = envs[idx].current_player()
            obs = envs[idx].encode()
            action = _sample_action_from_dist(dist, T, rng)
            pre_value[idx].append((obs, dist, mover))
            histories[idx].append(action)
            envs[idx].step(action)

        move_count += 1

    # Backfill values per game
    games: list[SelfPlayGame] = []
    for i in range(n_games):
        env = envs[i]
        truncated = not env.is_terminal()
        if truncated:
            returns: tuple[float, float] = (0.0, 0.0)
        else:
            r0, r1 = env.returns()
            returns = (float(r0), float(r1))

        examples = [
            TrainingExample(
                observation=obs,
                policy_target=dist.astype(np.float32, copy=False),
                value_target=returns[mover],
            )
            for (obs, dist, mover) in pre_value[i]
        ]
        games.append(
            SelfPlayGame(
                examples=examples,
                move_history=histories[i],
                returns=returns,
                truncated=truncated,
            )
        )
    return games


__all__ = [
    "TrainingExample",
    "SelfPlayGame",
    "default_temperature",
    "play_one_game",
    "play_batched_games",
]
