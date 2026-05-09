"""Tests for Phase 3 dataset generation.

Uses a small random Go-Net + few simulations + few games so the test
runs in seconds. Verifies shape correctness, value backfill logic,
disk roundtrip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.data.generation import (
    Phase3Example,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from split_brain_go.gonet.network import GoNet, GoNetConfig


@pytest.fixture()
def random_net() -> GoNet:
    torch.manual_seed(0)
    return GoNet(GoNetConfig.poc())


# ----------------------------------------------------- generation


def test_generate_returns_examples(random_net):
    examples = generate_dataset(
        random_net,
        n_games=2,
        n_simulations=4,
        max_moves=20,
        progress_every=99,
        rng=np.random.default_rng(0),
    )
    assert len(examples) > 0
    for ex in examples:
        assert isinstance(ex, Phase3Example)


def test_example_fields_have_correct_types(random_net):
    examples = generate_dataset(
        random_net,
        n_games=1,
        n_simulations=4,
        max_moves=15,
        progress_every=99,
        rng=np.random.default_rng(0),
    )
    ex = examples[0]
    assert isinstance(ex.game_id, int)
    assert isinstance(ex.move_number, int)
    assert ex.board.shape == (8, 9, 9)
    assert ex.board.dtype == torch.float32
    assert isinstance(ex.action, int)
    assert isinstance(ex.explanation, str)
    assert len(ex.explanation) > 10
    assert -1.0 <= ex.value_before <= 1.0


def test_game_ids_increment(random_net):
    """Examples from the same game share game_id; new games bump it."""
    examples = generate_dataset(
        random_net,
        n_games=3,
        n_simulations=4,
        max_moves=15,
        progress_every=99,
        rng=np.random.default_rng(0),
    )
    seen = set()
    for ex in examples:
        seen.add(ex.game_id)
    assert seen == {0, 1, 2}


def test_move_numbers_within_game_are_sequential(random_net):
    examples = generate_dataset(
        random_net,
        n_games=1,
        n_simulations=4,
        max_moves=15,
        progress_every=99,
        rng=np.random.default_rng(0),
    )
    seen_moves = sorted(ex.move_number for ex in examples)
    assert seen_moves == list(range(len(seen_moves)))


def test_value_after_present_for_non_terminal_moves(random_net):
    """Every move except possibly the last has value_after set."""
    examples = generate_dataset(
        random_net,
        n_games=1,
        n_simulations=4,
        max_moves=20,
        progress_every=99,
        rng=np.random.default_rng(7),
    )
    # All but the last example in a game must have value_after as a float.
    by_game: dict[int, list[Phase3Example]] = {}
    for ex in examples:
        by_game.setdefault(ex.game_id, []).append(ex)
    for game_examples in by_game.values():
        for ex in game_examples[:-1]:
            assert ex.value_after is not None


# ----------------------------------------------------- persistence


def test_save_and_load_roundtrip(tmp_path, random_net):
    examples = generate_dataset(
        random_net,
        n_games=1,
        n_simulations=4,
        max_moves=10,
        progress_every=99,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "phase3_data.pkl"
    save_dataset(examples, path)
    assert path.is_file()

    loaded = load_dataset(path)
    assert len(loaded) == len(examples)

    # Same fields preserved
    for orig, copy in zip(examples, loaded):
        assert orig.game_id == copy.game_id
        assert orig.move_number == copy.move_number
        assert orig.action == copy.action
        assert orig.explanation == copy.explanation
        assert torch.equal(orig.board, copy.board)


def test_load_rejects_non_phase3_pickle(tmp_path):
    bogus = tmp_path / "bogus.pkl"
    import pickle
    with bogus.open("wb") as f:
        pickle.dump([1, 2, 3], f)
    with pytest.raises(ValueError):
        load_dataset(bogus)


# ----------------------------------------------------- explanation diversity


def test_explanations_are_diverse(random_net):
    """Multiple games should yield non-identical explanations (3 templates × signals)."""
    examples = generate_dataset(
        random_net,
        n_games=3,
        n_simulations=4,
        max_moves=20,
        progress_every=99,
        rng=np.random.default_rng(42),
    )
    unique = {ex.explanation for ex in examples}
    # At least more than one unique string across N moves
    assert len(unique) > 5, f"Only {len(unique)} unique explanations from {len(examples)} examples"
