"""Replay buffer for self-play training data.

Holds up to ``capacity`` ``TrainingExample`` instances in FIFO order. The
buffer is in-memory; for the PoC scale (50k–500k examples × ~5 KB each)
this fits comfortably under 2 GB RAM. If we ever need disk overflow
(Phase 1.3b at full scale), see ``save``/``load`` for serialization.

Sampling is uniform-random *with* replacement over the current buffer
contents. This is the standard AlphaZero-style replay; recent and old
games contribute equally once they're in the buffer (eviction is the
only mechanism that downweights age).
"""

from __future__ import annotations

import pickle
import random
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from .selfplay import TrainingExample


class ReplayBuffer:
    """Bounded FIFO buffer of TrainingExamples.

    Capacity defaults to 500 000 (ADR-013). Adding examples beyond capacity
    silently drops the oldest. Thread-unsafe — single-process only.
    """

    def __init__(self, capacity: int = 500_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf: deque[TrainingExample] = deque(maxlen=capacity)

    # --------------------------------------------------------------- size

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen  # type: ignore[return-value]

    # ------------------------------------------------------------- mutate

    def add(self, examples: Iterable[TrainingExample]) -> None:
        """Append examples. Oldest are evicted if capacity is exceeded."""
        self._buf.extend(examples)

    def clear(self) -> None:
        self._buf.clear()

    # -------------------------------------------------------------- read

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform-random batch (with replacement).

        Returns:
            boards   (B, 8, 9, 9) float32 — observations stacked
            policies (B, 82)       float32 — MCTS distributions stacked
            values   (B,)          float32 — value targets

        All tensors live on CPU; the caller does ``.to(device)`` for GPU.
        Raises ``RuntimeError`` if the buffer is empty.
        """
        if len(self._buf) == 0:
            raise RuntimeError("Cannot sample from an empty buffer")

        # random.choices does with-replacement sampling and is fast.
        chosen = random.choices(self._buf, k=batch_size)
        boards = torch.stack([ex.observation for ex in chosen], dim=0)
        policies = torch.from_numpy(
            np.stack([ex.policy_target for ex in chosen], axis=0)
        )
        values = torch.tensor(
            [ex.value_target for ex in chosen], dtype=torch.float32
        )
        return boards, policies, values

    # ---------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Pickle the buffer contents to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(list(self._buf), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Replace buffer contents with the pickled list at ``path``."""
        with Path(path).open("rb") as f:
            items = pickle.load(f)
        self._buf.clear()
        self._buf.extend(items)


__all__ = ["ReplayBuffer"]
