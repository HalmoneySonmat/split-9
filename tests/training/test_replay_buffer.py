"""Unit tests for ReplayBuffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.training.replay_buffer import ReplayBuffer
from split_brain_go.training.selfplay import TrainingExample


def _make_example(seed: int) -> TrainingExample:
    g = torch.Generator().manual_seed(seed)
    return TrainingExample(
        observation=torch.randn(8, 9, 9, generator=g),
        policy_target=np.random.RandomState(seed).dirichlet([1.0] * 82).astype(
            np.float32
        ),
        value_target=float(seed % 3 - 1),  # cycles through -1, 0, +1
    )


# ----------------------------------------------------------- basics


def test_add_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.add([_make_example(i) for i in range(10)])
    assert len(buf) == 10


def test_capacity_eviction():
    buf = ReplayBuffer(capacity=5)
    buf.add([_make_example(i) for i in range(8)])
    # Capacity 5, added 8 → only the most recent 5 should remain.
    assert len(buf) == 5


# ------------------------------------------------------------- sample


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    buf.add([_make_example(i) for i in range(50)])
    boards, policies, values = buf.sample(batch_size=32)
    assert boards.shape == (32, 8, 9, 9)
    assert boards.dtype == torch.float32
    assert policies.shape == (32, 82)
    assert policies.dtype == torch.float32
    assert values.shape == (32,)
    assert values.dtype == torch.float32


def test_sample_empty_raises():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(RuntimeError):
        buf.sample(batch_size=4)


# ---------------------------------------------------------- persistence


def test_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=100)
    buf.add([_make_example(i) for i in range(20)])
    path = tmp_path / "buffer.pkl"
    buf.save(path)

    buf2 = ReplayBuffer(capacity=100)
    buf2.load(path)

    assert len(buf2) == 20
    # Sampling shouldn't crash and gives matching shapes
    boards, policies, values = buf2.sample(batch_size=4)
    assert boards.shape == (4, 8, 9, 9)
