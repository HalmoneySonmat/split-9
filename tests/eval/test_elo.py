"""Tests for ELO fitting.

We construct synthetic tournament results with a known *true* skill
ordering and verify that ``fit_elo`` recovers the order. Exact rating
values aren't checked — they depend on initialisation and step count;
order is what matters for evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from split_brain_go.eval.elo import (
    TournamentResult,
    bootstrap_elo_ci,
    fit_elo,
    format_ratings_table,
)


# -------------------------------------------------------- helpers


def _make_result(wins: list[list[float]], games: list[list[int]],
                 names: list[str]) -> TournamentResult:
    return TournamentResult(
        agent_names=names,
        wins=np.asarray(wins, dtype=np.float64),
        games=np.asarray(games, dtype=np.int64),
    )


# -------------------------------------------------------- ordering


def test_two_agents_strong_dominates():
    """A wins all games against B → A's rating must be higher."""
    wins = [[0, 100], [0, 0]]
    games = [[0, 100], [100, 0]]
    result = _make_result(wins, games, ["A", "B"])
    ratings = fit_elo(result, n_steps=2000)
    assert ratings["A"] > ratings["B"]
    # ELO anchor sets agent 0 (A) to 1500, so B should be much lower.
    assert ratings["A"] == 1500
    assert ratings["B"] < 1400


def test_three_agents_transitive_order():
    """A > B > C in skill: A beats B 70%, B beats C 70%, A beats C 85%."""
    wins = [
        [0, 70, 85],
        [30, 0, 70],
        [15, 30, 0],
    ]
    games = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]
    result = _make_result(wins, games, ["A", "B", "C"])
    ratings = fit_elo(result, n_steps=3000)
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_equal_strength_agents():
    """Two agents that draw 50/50 → similar ratings."""
    wins = [[0, 50], [50, 0]]
    games = [[0, 100], [100, 0]]
    result = _make_result(wins, games, ["X", "Y"])
    ratings = fit_elo(result, n_steps=2000)
    # Should be roughly equal (within tolerance)
    assert abs(ratings["X"] - ratings["Y"]) < 50


# -------------------------------------------------------- anchor


def test_anchor_holds():
    wins = [[0, 30], [70, 0]]
    games = [[0, 100], [100, 0]]
    result = _make_result(wins, games, ["A", "B"])
    ratings = fit_elo(result, anchor_idx=0, anchor_rating=1500.0, n_steps=2000)
    assert ratings["A"] == 1500.0  # snapped exactly
    # B should be higher (B beat A 70%)
    assert ratings["B"] > 1500


def test_anchor_alternative():
    wins = [[0, 30], [70, 0]]
    games = [[0, 100], [100, 0]]
    result = _make_result(wins, games, ["A", "B"])
    # Anchor B at 1500; A should be lower since A is weaker.
    ratings = fit_elo(result, anchor_idx=1, anchor_rating=1500.0, n_steps=2000)
    assert ratings["B"] == 1500.0
    assert ratings["A"] < 1500


# -------------------------------------------------------- edge


def test_single_agent():
    result = _make_result([[0]], [[0]], ["only"])
    ratings = fit_elo(result, anchor_rating=1500)
    assert ratings == {"only": 1500.0}


def test_zero_agents():
    result = _make_result([], [], [])
    assert fit_elo(result) == {}


# -------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_estimate():
    """Bootstrap CI should contain the point estimate (almost always)."""
    wins = [[0, 70, 85], [30, 0, 70], [15, 30, 0]]
    games = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]
    result = _make_result(wins, games, ["A", "B", "C"])
    ratings = fit_elo(result, n_steps=2000)
    cis = bootstrap_elo_ci(result, n_bootstrap=20, seed=0, n_steps=1000)
    for name in ["A", "B", "C"]:
        lo, hi = cis[name]
        # Wide tolerance because n_bootstrap=20 is small + rating noise.
        assert lo - 100 <= ratings[name] <= hi + 100, (
            f"{name}: rating={ratings[name]} outside CI [{lo}, {hi}]"
        )


# -------------------------------------------------------- formatting


def test_format_ratings_table():
    ratings = {"A": 1600, "B": 1500, "C": 1400}
    text = format_ratings_table(ratings)
    lines = text.split("\n")
    # First non-header line should be A (highest rating)
    assert "A" in lines[2]
    assert "C" in lines[-1]


def test_format_ratings_table_with_ci():
    ratings = {"A": 1600, "B": 1500}
    cis = {"A": (1550, 1650), "B": (1450, 1550)}
    text = format_ratings_table(ratings, cis)
    assert "[" in text and "]" in text
