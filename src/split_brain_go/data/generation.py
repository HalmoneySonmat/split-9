"""Phase 3 dataset generation — self-play games into (board, signal, text) tuples.

Runs the trained Go-Net + MCTS through ``n_games`` self-play games and
captures, for every move:

    * the encoded board state (the input that produced the move)
    * the action played
    * the network's policy/value output at that state
    * the synthesized English explanation (random template choice)

Activations are *not* stored — they're recomputed on-the-fly during
adapter training. Storing them would inflate disk by ~1.5 GB per 100k
examples and pin the activation-layer choice. Keeping just the board
keeps the dataset small (~5 KB / example) and lets us experiment with
which Go-Net layers feed the adapter.

Output is a list of ``Phase3Example`` dataclasses, picklable for
serialisation. ``save_dataset`` / ``load_dataset`` provide a one-line
disk interface.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..env.go_env import GoEnv
from ..gonet.mcts import MCTS
from ..gonet.network import GoNet
from .synthetic import synthesize


# ============================================================ data class


@dataclass
class Phase3Example:
    """One training example for Phase 3 adapter learning.

    Attributes:
        game_id: Sequential id within this generation run.
        move_number: 0-indexed move within its game.
        board: ``(8, 9, 9)`` float tensor — same encoding the Go-Net
            consumes. CPU tensor; move to device at training time.
        action: Action id played (0..81; 81 = pass).
        explanation: Synthesized English text — the adapter's learning
            target.
        selected_confidence: Softmax probability of the chosen action
            (kept for filtering / analysis, not training).
        value_before: Network value estimate at the state, from mover's
            POV.
        value_after: Network value estimate at the next state, from the
            same mover's POV (sign-flipped from the next-state network
            output). ``None`` only at the final move of an unfinished
            game.
    """

    game_id: int
    move_number: int
    board: torch.Tensor
    action: int
    explanation: str
    selected_confidence: float
    value_before: float
    value_after: float | None


# ============================================================ generator


def _network_value(network: GoNet, env: GoEnv, device: torch.device) -> tuple[
    torch.Tensor, float
]:
    """Run network at ``env``, return (policy_logits, scalar value).

    Returns CPU tensors ready to be passed to ``synthesize``.
    """
    board = env.encode().unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        policy_logits, value = network(board)
    return policy_logits[0].cpu(), float(value.item())


def generate_dataset(
    network: GoNet,
    n_games: int = 100,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    max_moves: int = 200,
    temperature: float = 0.5,
    add_root_noise: bool = True,
    device: str | torch.device | None = None,
    rng: np.random.Generator | None = None,
    progress_every: int = 10,
) -> list[Phase3Example]:
    """Generate a Phase 3 training dataset by self-play.

    The temperature is set lower than typical training self-play (0.5
    rather than 1.0) so the games show *informative* moves the LLM can
    explain — fully-random openings would yield meaningless explanations.

    Args:
        network: Trained Go-Net.
        n_games: How many full games to play.
        n_simulations: MCTS simulations per move.
        max_moves: Safety cap.
        temperature: Action sampling temperature.
        add_root_noise: Dirichlet noise at root (for game-to-game diversity).
        progress_every: Print a brief progress line every N games.

    Returns:
        Flat list of all ``Phase3Example`` instances across all games.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if device is None:
        device = next(network.parameters()).device
    device = torch.device(device)

    examples: list[Phase3Example] = []
    mcts = MCTS(
        network,
        n_simulations=n_simulations,
        c_puct=c_puct,
        device=device,
    )

    for game_id in range(n_games):
        env = GoEnv()
        env.reset()

        # Stage 1: play through, recording per-move snapshots (incl. PVs).
        per_move = []
        move_idx = 0
        while not env.is_terminal() and move_idx < max_moves:
            env_before = env.clone()
            policy_logits, value_before = _network_value(network, env, device)

            # Search with PV extraction. Sample action from visit distribution.
            visit_dist, pvs = mcts.search_with_pvs(
                env,
                k=3,
                max_depth=5,
                add_root_noise=add_root_noise,
                rng=rng,
            )
            if temperature <= 0:
                action = int(np.argmax(visit_dist))
            else:
                transformed = visit_dist ** (1.0 / temperature)
                total = transformed.sum()
                if total <= 0:
                    action = int(rng.choice(env.legal_actions()))
                else:
                    probs = transformed / total
                    action = int(rng.choice(len(visit_dist), p=probs))

            env.step(action)
            env_after = env.clone()

            per_move.append(
                {
                    "move_number": move_idx,
                    "env_before": env_before,
                    "env_after": env_after,
                    "policy_logits": policy_logits,
                    "value_before": value_before,
                    "action": int(action),
                    "board": env_before.encode(),
                    "principal_variations": pvs,
                }
            )
            move_idx += 1

        # Stage 2: backfill value_after, then synthesize explanations.
        for i, m in enumerate(per_move):
            value_after: float | None
            if i + 1 < len(per_move):
                # Next move is from the opponent's POV, so value flips sign
                # to be from THIS move's mover POV.
                value_after = -float(per_move[i + 1]["value_before"])
            elif env.is_terminal():
                # Final move of a finished game: terminal value from
                # current mover's POV.
                returns = env.returns()
                mover = m["env_before"].current_player()
                value_after = float(returns[mover]) if mover in (0, 1) else None
            else:
                value_after = None

            signal, explanation = synthesize(
                m["env_before"],
                m["action"],
                m["env_after"],
                m["policy_logits"],
                m["value_before"],
                value_after,
                principal_variations=m["principal_variations"],
                rng=rng,
            )

            examples.append(
                Phase3Example(
                    game_id=game_id,
                    move_number=m["move_number"],
                    board=m["board"],
                    action=m["action"],
                    explanation=explanation,
                    selected_confidence=signal.selected_confidence,
                    value_before=m["value_before"],
                    value_after=value_after,
                )
            )

        if (game_id + 1) % progress_every == 0:
            print(
                f"  generated {game_id + 1}/{n_games} games — "
                f"{len(examples)} examples so far"
            )

    return examples


# ============================================================ persistence


def save_dataset(examples: list[Phase3Example], path: str | Path) -> None:
    """Pickle a list of examples to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(path: str | Path) -> list[Phase3Example]:
    """Load a previously-pickled dataset."""
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or (data and not isinstance(data[0], Phase3Example)):
        raise ValueError(f"Pickle at {path} does not contain Phase3Example list")
    return data


__all__ = [
    "Phase3Example",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]
