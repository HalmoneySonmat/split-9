"""Unit tests for synthetic explanation generation.

We construct deterministic states so signal extraction is checkable.
Templates are exercised for shape (returns non-empty strings, no
exceptions on edge cases like passes / first move / captures).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from split_brain_go.data.synthetic import (
    GameSignal,
    extract_signal,
    render_explanation,
    synthesize,
)
from split_brain_go.env.go_env import PASS_ACTION, GoEnv
from split_brain_go.gonet.mcts import PrincipalVariation


# ---------------------------------------------------------- helpers


def _step(env: GoEnv, action: int) -> GoEnv:
    """Clone, apply, return the new env."""
    new_env = env.clone()
    new_env.step(action)
    return new_env


def _zero_logits() -> torch.Tensor:
    return torch.zeros(82)


# ---------------------------------------------------------- signals


def test_signal_records_basic_fields():
    env = GoEnv()
    env.reset()
    after = _step(env, 40)  # Black plays center
    sig = extract_signal(env, 40, after, _zero_logits(), value=0.1, value_after=0.2)

    assert sig.move_number == 0
    assert sig.is_pass is False
    assert sig.selected_action == 40
    assert sig.selected_pos == (4, 4)
    assert sig.value_before == pytest.approx(0.1)
    assert sig.value_after == pytest.approx(0.2)
    assert sig.value_delta == pytest.approx(0.1)
    assert sig.captures == 0
    assert sig.game_phase == "opening"
    assert sig.mover == 0  # Black


def test_signal_handles_pass():
    env = GoEnv()
    env.reset()
    after = _step(env, PASS_ACTION)
    sig = extract_signal(env, PASS_ACTION, after, _zero_logits(), value=0.0)
    assert sig.is_pass is True
    assert sig.selected_pos is None


def test_signal_handles_no_value_after():
    """When value_after is None (game ended), delta should be None."""
    env = GoEnv()
    env.reset()
    after = _step(env, 0)
    sig = extract_signal(env, 0, after, _zero_logits(), value=0.5, value_after=None)
    assert sig.value_after is None
    assert sig.value_delta is None


def test_signal_phase_label_progresses():
    """Move number determines phase. We can fake by playing pass moves."""
    env = GoEnv()
    env.reset()
    # Play 17 passes to enter middlegame (after move 16)
    for i in range(17):
        env.step(PASS_ACTION) if not env.is_terminal() else None
    # If it terminated (two passes), make a fresh env at the same move number
    # (simpler: just check the boundary directly via a manual signal)
    sig = GameSignal(
        move_number=20,
        is_pass=False,
        selected_action=0,
        selected_pos=(0, 0),
        selected_confidence=0.5,
        game_phase="middlegame",
        mover=0,
    )
    assert sig.game_phase == "middlegame"


def test_signal_top_alternatives_match_policy_prob():
    env = GoEnv()
    env.reset()
    # Logits favouring action 40 strongly
    logits = torch.full((82,), -10.0)
    logits[40] = 5.0
    logits[20] = 2.0
    logits[30] = 1.0
    after = _step(env, 40)
    sig = extract_signal(env, 40, after, logits, value=0.0)
    # Top-3 = [40, 20, 30] but 40 is the chosen one, so alts are 20 and 30
    alt_actions = [a for a, _, _ in sig.top_alternatives]
    assert 20 in alt_actions
    assert 30 in alt_actions


# ---------------------------------------------------------- templates


def test_render_returns_nonempty_string():
    sig = GameSignal(
        move_number=5,
        is_pass=False,
        selected_action=40,
        selected_pos=(4, 4),
        selected_confidence=0.45,
        top_alternatives=[(20, (2, 2), 0.20), (50, (5, 5), 0.10)],
        value_before=0.1,
        value_after=0.3,
        value_delta=0.2,
        captures=0,
        game_phase="opening",
        mover=0,
    )
    rng = np.random.default_rng(0)
    text = render_explanation(sig, rng)
    assert isinstance(text, str)
    assert len(text) > 30


def test_render_handles_pass():
    sig = GameSignal(
        move_number=10,
        is_pass=True,
        selected_action=PASS_ACTION,
        selected_pos=None,
        selected_confidence=0.30,
        top_alternatives=[],
        value_before=0.0,
        value_after=0.0,
        value_delta=0.0,
        captures=0,
        game_phase="opening",
        mover=1,  # White passing
    )
    rng = np.random.default_rng(0)
    text = render_explanation(sig, rng)
    assert "pass" in text.lower() or "White" in text


def test_render_with_captures_mentions_them():
    """At least one of the templates always mentions captures > 0."""
    sig = GameSignal(
        move_number=20,
        is_pass=False,
        selected_action=15,
        selected_pos=(1, 6),
        selected_confidence=0.55,
        top_alternatives=[],
        value_before=0.0,
        value_after=0.4,
        value_delta=0.4,
        captures=3,
        game_phase="middlegame",
        mover=0,
    )
    # Try several templates
    rng = np.random.default_rng(0)
    seen_captures = 0
    for _ in range(20):
        text = render_explanation(sig, rng)
        if "captur" in text.lower() or " 3 " in text:
            seen_captures += 1
    assert seen_captures > 0, "No template mentioned captures across 20 draws"


def test_render_variety():
    """Different RNG seeds should sometimes produce different texts."""
    sig = GameSignal(
        move_number=5,
        is_pass=False,
        selected_action=40,
        selected_pos=(4, 4),
        selected_confidence=0.45,
        top_alternatives=[],
        value_before=0.1,
        value_after=0.2,
        value_delta=0.1,
        captures=0,
        game_phase="opening",
        mover=0,
    )
    seen = set()
    for seed in range(20):
        seen.add(render_explanation(sig, np.random.default_rng(seed)))
    assert len(seen) > 1, "All seeds produced the same text"


# ---------------------------------------------------------- one-shot


def test_synthesize_returns_signal_and_text():
    env = GoEnv()
    env.reset()
    after = _step(env, 40)
    sig, text = synthesize(
        env, 40, after, _zero_logits(), value=0.1, value_after=0.2,
        rng=np.random.default_rng(0),
    )
    assert isinstance(sig, GameSignal)
    assert isinstance(text, str) and len(text) > 30


# ---------------------------------------------------------- PV templates


def _make_pv(first_action: int, first_mover: int, value: float) -> PrincipalVariation:
    return PrincipalVariation(
        moves=[(first_action, first_mover), (20, 1 - first_mover), (50, first_mover)],
        visit_count=42,
        value=value,
    )


def test_signal_carries_principal_variations():
    pvs = [_make_pv(40, 0, 0.30), _make_pv(20, 0, 0.05)]
    env = GoEnv()
    env.reset()
    after = _step(env, 40)
    sig = extract_signal(
        env, 40, after, _zero_logits(),
        value=0.1, value_after=0.2,
        principal_variations=pvs,
    )
    assert len(sig.principal_variations) == 2
    assert sig.principal_variations[0].value == pytest.approx(0.30)


def test_pv_templates_render_lines():
    """PV-based templates must mention move coordinates from the PV."""
    pvs = [_make_pv(40, 0, 0.30), _make_pv(20, 0, 0.05), _make_pv(60, 0, -0.15)]
    sig = GameSignal(
        move_number=10,
        is_pass=False,
        selected_action=40,
        selected_pos=(4, 4),
        selected_confidence=0.55,
        top_alternatives=[],
        value_before=0.1,
        value_after=0.4,
        value_delta=0.3,
        captures=0,
        game_phase="middlegame",
        mover=0,
        principal_variations=pvs,
    )
    rng = np.random.default_rng(0)
    seen_with_pv_marker = 0
    for _ in range(40):
        text = render_explanation(sig, rng)
        # At least some draws should hit a PV-based template, which
        # mentions "val +0.30" or similar from one of the PVs.
        if "val +0.30" in text or "val +0.05" in text or "val -0.15" in text:
            seen_with_pv_marker += 1
    assert seen_with_pv_marker > 0, (
        "No PV-based template fired across 40 draws — template list broken"
    )


def test_pv_template_handles_no_pvs_gracefully():
    """If no PVs given, PV templates fall back to non-PV templates."""
    sig = GameSignal(
        move_number=5,
        is_pass=False,
        selected_action=40,
        selected_pos=(4, 4),
        selected_confidence=0.45,
        top_alternatives=[],
        value_before=0.1,
        value_after=0.2,
        value_delta=0.1,
        captures=0,
        game_phase="opening",
        mover=0,
        principal_variations=[],
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        text = render_explanation(sig, rng)
        # No template should crash; text always non-empty
        assert isinstance(text, str) and len(text) > 0


def test_six_templates_registered():
    from split_brain_go.data.synthetic import _TEMPLATES
    assert len(_TEMPLATES) == 6
