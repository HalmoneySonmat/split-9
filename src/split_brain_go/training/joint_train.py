"""AlphaZero-style joint training loop for GoNet.

Components:
    * ``alphazero_loss``  — pure function, policy CE + value MSE.
    * ``train_step``      — one forward-backward-update on a batch.
    * ``eval_vs_random``  — quick mid-training winrate check.
    * ``train_gonet``     — the alternating self-play / training cycle loop.

Schedule (ADR-014, alternating):
    For each cycle:
        1. Generate ``games_per_cycle`` self-play games with the current net.
        2. Add their examples to the replay buffer.
        3. Take ``train_steps_per_cycle`` mini-batch SGD steps.
        4. Evaluate vs Random for ``eval_n_games``.
        5. Save a checkpoint, tracking the best metric so far.

The ``TrainConfig`` dataclass holds every knob. Defaults are PoC values —
fast enough for a smoke test (≈ 30s on CPU), too small for serious training.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..env.go_env import GoEnv
from ..eval.winrate import eval_vs_greedy, eval_vs_random
from ..gonet.mcts import MCTS
from ..gonet.network import GoNet, GoNetConfig
from ..utils.logging import TrainLogger, make_run_name
from ..utils.seed import set_global_seed
from .checkpoint import CheckpointManager
from .replay_buffer import ReplayBuffer
from .selfplay import play_batched_games, play_one_game


# ============================================================ config


@dataclass
class TrainConfig:
    """All knobs for ``train_gonet``. PoC defaults."""

    seed: int = 42

    # Model (built into GoNetConfig at runtime)
    n_blocks: int = 4
    channels: int = 64

    # Outer schedule
    n_cycles: int = 5
    games_per_cycle: int = 10
    train_steps_per_cycle: int = 100

    # Self-play
    n_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.25
    dirichlet_weight: float = 0.25
    max_moves: int = 200

    # Batched self-play (Phase 1.3b throughput upgrade).
    # Set use_batched=True to run multiple games concurrently on one GPU.
    # batch_n_games is the chunk size; ``games_per_cycle`` is split into
    # chunks of this size. With batch_n_games=8 the GPU forward batch is
    # 8 leaves at a time — roughly 4–6× faster on RTX 3070 Ti than single.
    use_batched: bool = False
    batch_n_games: int = 8

    # Optimization
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Buffer
    buffer_capacity: int = 50_000
    min_buffer: int = 256

    # Eval
    eval_n_games: int = 20
    eval_n_simulations: int = 50
    eval_max_moves: int = 200
    # Whether to also evaluate vs Greedy (the network's own argmax with no MCTS).
    # vs Greedy is the *MCTS lift* metric — sensitive to learning progress
    # in a way that vs Random saturates early. Cost: ~2x eval time.
    eval_against_greedy: bool = True

    # IO
    checkpoint_dir: str = "runs/checkpoints"
    log_every: int = 10

    # Logging (TrainLogger). ``log_to`` ∈ {csv, wandb, auto, noop}.
    # 'csv' (default) writes runs/<log_run_name>/metrics.csv on disk —
    # zero external dependency. 'wandb' uses Weights & Biases (account
    # required). 'auto' tries wandb first, falls back to csv. 'noop' is
    # silent (used in tests).
    log_to: str = "csv"
    log_run_name: str | None = None  # auto-generated if None
    log_dir: str = "runs"
    log_train_steps: bool = False  # if True, log per-step train metrics

    def gonet_config(self) -> GoNetConfig:
        return GoNetConfig(n_blocks=self.n_blocks, channels=self.channels)


# ============================================================ losses


def alphazero_loss(
    pred_policy_logits: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """AlphaZero loss = soft cross-entropy (policy) + MSE (value).

    Shapes:
        pred_policy_logits  (B, 82) raw logits
        pred_value          (B,)    in [-1, 1]
        target_policy       (B, 82) MCTS visit distribution
        target_value        (B,)    game outcome from mover's POV
    """
    log_probs = F.log_softmax(pred_policy_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(pred_value, target_value)
    return {
        "loss": policy_loss + value_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


# ======================================================== train step


def train_step(
    network: GoNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One forward + backward + optimizer step. Returns scalar metrics."""
    network.train()
    boards, policies, values = (t.to(device, non_blocking=True) for t in batch)

    pred_logits, pred_values = network(boards)
    losses = alphazero_loss(pred_logits, pred_values, policies, values)

    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {k: float(v.detach().item()) for k, v in losses.items()}


# Note: eval_vs_random and eval_vs_greedy are imported from ``eval.winrate``.
# They return PairwiseResult, not float. ``train_gonet`` extracts ``.a_winrate``.


# ============================================================= main


def train_gonet(cfg: TrainConfig) -> dict[str, float]:
    """Run the full alternating cycle loop.

    Returns final metrics: ``{"winrate": ..., "loss": ...}``.
    """
    set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = GoNet(cfg.gonet_config()).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
    ckpt_mgr = CheckpointManager(
        Path(cfg.checkpoint_dir), metric_name="winrate", higher_is_better=True
    )

    # TrainLogger: writes metrics each cycle. CSV by default, optional wandb.
    run_name = cfg.log_run_name or make_run_name("phase1")
    logger = TrainLogger(
        run_name=run_name,
        config=asdict(cfg),
        mode=cfg.log_to,
        run_dir=cfg.log_dir,
    )
    if logger.is_active and logger.run_dir is not None:
        print(f"Logging metrics to: {logger.run_dir}")
    elif logger.is_active:
        print(f"Logging metrics via wandb run: {run_name}")
    else:
        print("Metrics logging: noop")

    final_metrics: dict[str, float] = {"winrate": 0.0, "loss": float("inf")}

    for cycle in range(cfg.n_cycles):
        cycle_start = time.time()

        # 1) Self-play — batched (Phase 1.3b) or one-game-at-a-time (PoC).
        if cfg.use_batched:
            remaining = cfg.games_per_cycle
            while remaining > 0:
                chunk = min(remaining, cfg.batch_n_games)
                games = play_batched_games(
                    network,
                    n_games=chunk,
                    n_simulations=cfg.n_simulations,
                    c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_weight=cfg.dirichlet_weight,
                    max_moves=cfg.max_moves,
                    device=device,
                )
                for g in games:
                    buffer.add(g.examples)
                remaining -= chunk
        else:
            for _ in range(cfg.games_per_cycle):
                game = play_one_game(
                    network,
                    n_simulations=cfg.n_simulations,
                    c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_weight=cfg.dirichlet_weight,
                    max_moves=cfg.max_moves,
                    device=device,
                )
                buffer.add(game.examples)

        # 2) Train
        last_loss = float("inf")
        last_policy_loss = float("inf")
        last_value_loss = float("inf")
        if len(buffer) >= cfg.min_buffer:
            for step in range(cfg.train_steps_per_cycle):
                batch = buffer.sample(cfg.batch_size)
                metrics = train_step(
                    network, optimizer, batch, device, grad_clip=cfg.grad_clip
                )
                last_loss = metrics["loss"]
                last_policy_loss = metrics["policy_loss"]
                last_value_loss = metrics["value_loss"]

                if cfg.log_train_steps:
                    global_step = cycle * cfg.train_steps_per_cycle + step
                    logger.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/policy_loss": metrics["policy_loss"],
                            "train/value_loss": metrics["value_loss"],
                        },
                        step=global_step,
                    )

                if step % cfg.log_every == 0:
                    print(
                        f"  cycle={cycle} step={step:4d} "
                        f"loss={metrics['loss']:.4f} "
                        f"policy={metrics['policy_loss']:.4f} "
                        f"value={metrics['value_loss']:.4f}"
                    )

        # 3) Eval — vs Random (always) and vs Greedy (if enabled).
        result_random = eval_vs_random(
            network,
            n_games=cfg.eval_n_games,
            n_simulations=cfg.eval_n_simulations,
            max_moves=cfg.eval_max_moves,
        )
        winrate_random = result_random.a_winrate

        winrate_greedy: float | None = None
        if cfg.eval_against_greedy:
            result_greedy = eval_vs_greedy(
                network,
                n_games=cfg.eval_n_games,
                n_simulations=cfg.eval_n_simulations,
                max_moves=cfg.eval_max_moves,
            )
            winrate_greedy = result_greedy.a_winrate

        # 4) Checkpoint — track best by vs-Greedy if available, else vs-Random.
        primary_metric = (
            winrate_greedy if winrate_greedy is not None else winrate_random
        )
        ckpt_mgr.save(
            step=cycle,
            model=network,
            optimizer=optimizer,
            metric=primary_metric,
            extra={
                "loss": last_loss,
                "buffer_size": len(buffer),
                "winrate_vs_random": winrate_random,
                "winrate_vs_greedy": winrate_greedy,
            },
        )

        elapsed = time.time() - cycle_start
        greedy_str = (
            f", winrate vs greedy = {winrate_greedy:.2f}"
            if winrate_greedy is not None else ""
        )
        print(
            f"cycle {cycle}/{cfg.n_cycles - 1} done in {elapsed:.1f}s — "
            f"winrate vs random = {winrate_random:.2f}"
            f"{greedy_str}, "
            f"buffer size = {len(buffer)}"
        )

        # Per-cycle logging
        log_dict: dict[str, float] = {
            "cycle/loss": last_loss,
            "cycle/policy_loss": last_policy_loss,
            "cycle/value_loss": last_value_loss,
            "cycle/winrate_vs_random": winrate_random,
            "cycle/buffer_size": len(buffer),
            "cycle/elapsed_seconds": elapsed,
        }
        if winrate_greedy is not None:
            log_dict["cycle/winrate_vs_greedy"] = winrate_greedy
        logger.log(log_dict, step=cycle)

        final_metrics = {
            "winrate": winrate_random,
            "winrate_vs_greedy": winrate_greedy if winrate_greedy is not None else 0.0,
            "loss": last_loss,
        }

    logger.finish()
    return final_metrics


__all__ = [
    "TrainConfig",
    "alphazero_loss",
    "train_step",
    "eval_vs_random",
    "train_gonet",
]
