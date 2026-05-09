"""Synthetic explanation generation from Go-Net outputs and game state.

Each (state, action, next-state) triple produces a structured ``GameSignal``
of objective facts (which move, top alternatives, value before/after,
captures, game phase). A template renderer then converts the signal into
an English sentence — the training target for the adapter (Phase 3).

We deliberately do NOT use any external human commentary. ADR-007 lays
out the reason: mixing human commentary would confound the faithfulness
measurement (Phase 4) by giving the LLM another way to sound plausible.
The synthetic templates are intentionally narrow but varied enough that
the LLM doesn't memorise a single sentence.

Multiple templates are provided to avoid monotony; the renderer picks
one at random per call (seedable via ``rng``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..env.go_env import BOARD_SIZE, PASS_ACTION, GoEnv
from ..gonet.mcts import PrincipalVariation

# Move-count thresholds for game phase labels (9x9 Go).
PHASE_OPENING_END = 16
PHASE_MIDDLE_END = 50


# ============================================================== signal


@dataclass
class GameSignal:
    """Objective per-move facts extracted from environment + network."""

    move_number: int
    is_pass: bool
    selected_action: int
    selected_pos: tuple[int, int] | None  # None for pass
    selected_confidence: float
    top_alternatives: list[tuple[int, tuple[int, int] | None, float]] = field(
        default_factory=list
    )
    value_before: float = 0.0
    value_after: float | None = None
    value_delta: float | None = None  # value_after − value_before, mover POV
    captures: int = 0
    game_phase: str = "opening"  # "opening" | "middlegame" | "endgame"
    mover: int = 0  # 0 = Black, 1 = White
    # MCTS principal variations — top-K lines explored, each up to ~5 plies.
    # First PV is usually the chosen move's line.
    principal_variations: list[PrincipalVariation] = field(default_factory=list)


# ============================================================ extraction


def _action_to_pos(action: int) -> tuple[int, int] | None:
    if action == PASS_ACTION:
        return None
    return action // BOARD_SIZE, action % BOARD_SIZE


def _phase_label(move_number: int) -> str:
    if move_number < PHASE_OPENING_END:
        return "opening"
    if move_number < PHASE_MIDDLE_END:
        return "middlegame"
    return "endgame"


def _count_stones(env: GoEnv, color: int) -> int:
    """Number of `color` (0=Black, 1=White) stones on the board."""
    obs = env.encode()  # (8, 9, 9): channel 0 is own, 1 is opp from mover POV
    # Reconstruct via raw OpenSpiel tensor for color absolute counts.
    import numpy as np
    from ..env.encoding import _board_planes_from_observation

    black, white = _board_planes_from_observation(env._state)
    return int((black if color == 0 else white).sum())


def extract_signal(
    env_before: GoEnv,
    action: int,
    env_after: GoEnv,
    policy_logits: torch.Tensor,
    value: float,
    value_after: float | None = None,
    top_k: int = 3,
    principal_variations: list[PrincipalVariation] | None = None,
) -> GameSignal:
    """Build a ``GameSignal`` from a single move's data.

    Args:
        env_before: ``GoEnv`` snapshot just before the move.
        action: Action id played (0..81).
        env_after: ``GoEnv`` after ``env_before.step(action)``.
        policy_logits: ``(N_ACTIONS,)`` logits from Go-Net at env_before.
        value: Scalar value at env_before, from mover's POV.
        value_after: Scalar value at env_after, from mover's POV (i.e. for
            the *same* player; you may need to negate the network output if
            the network reports value from the new current player's POV).
            None if the game ended.
        top_k: Number of top alternative actions to record.

    Returns:
        Filled ``GameSignal``.
    """
    # Softmax over policy, then mask illegal actions for honest top-k.
    probs = F.softmax(policy_logits, dim=-1).detach().cpu().numpy()
    legal = env_before.legal_actions()
    legal_mask = np.zeros_like(probs)
    legal_mask[legal] = 1.0
    masked = probs * legal_mask
    s = masked.sum()
    if s > 0:
        masked = masked / s

    # Top-k by masked prob; include the actually selected action even if
    # it's not top-k by prob (rare with temperature=0 sampling).
    sorted_idx = np.argsort(-masked)
    top_idxs = list(sorted_idx[:top_k])
    if action not in top_idxs:
        top_idxs = [action] + top_idxs[: top_k - 1]
    alternatives = [
        (int(a), _action_to_pos(int(a)), float(masked[int(a)]))
        for a in top_idxs
        if int(a) != action
    ][: top_k - 1]

    # Captures: drop in opponent's stone count after the move.
    mover = env_before.current_player()
    opp = 1 - mover if mover in (0, 1) else 0
    opp_before = _count_stones(env_before, opp)
    opp_after = _count_stones(env_after, opp)
    captures = max(0, opp_before - opp_after)

    delta = None
    if value_after is not None:
        delta = float(value_after) - float(value)

    return GameSignal(
        move_number=len(env_before.history()),
        is_pass=(action == PASS_ACTION),
        selected_action=int(action),
        selected_pos=_action_to_pos(int(action)),
        selected_confidence=float(masked[int(action)]),
        top_alternatives=alternatives,
        value_before=float(value),
        value_after=value_after if value_after is None else float(value_after),
        value_delta=delta,
        captures=captures,
        game_phase=_phase_label(len(env_before.history())),
        mover=int(mover) if mover in (0, 1) else 0,
        principal_variations=list(principal_variations or []),
    )


# ============================================================ templates


def _format_pos(pos: tuple[int, int] | None) -> str:
    if pos is None:
        return "pass"
    r, c = pos
    return f"({r}, {c})"


def _format_alts(alts: list) -> str:
    if not alts:
        return "no significant alternatives"
    parts = []
    for _a, pos, p in alts:
        parts.append(f"{_format_pos(pos)} [{p:.2f}]")
    return ", ".join(parts)


def _format_value_delta(delta: float | None) -> str:
    if delta is None:
        return "the game ended"
    if delta > 0.05:
        return f"the value rose by {delta:+.2f}"
    if delta < -0.05:
        return f"the value fell by {delta:+.2f}"
    return "the value barely moved"


# ---------------------------- word-variation pools (E)


_VERB_CHOSE = ("chose", "played", "selected", "placed at")
_VERB_CONSIDER = ("Considered", "Evaluated", "Examined", "Explored")
_VERB_IMAGINE = ("imagined", "envisioned", "pictured", "visualized")
_NOUN_VALUE = ("value", "score", "evaluation", "outlook")
_NOUN_ALTERNATIVE = ("alternatives", "candidates", "other lines", "variations")


def _pick(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


# ---------------------------- PV formatting


def _format_color_pos(action: int, mover: int) -> str:
    """`B(4, 4)` style. Pass shows as `B(pass)`."""
    pos = _action_to_pos(action)
    color = "B" if mover == 0 else "W"
    if pos is None:
        return f"{color}(pass)"
    return f"{color}({pos[0]}, {pos[1]})"


def _format_pv_line(pv: PrincipalVariation) -> str:
    """One PV: ``B(4, 4) W(2, 3) B(5, 6) → val +0.35``."""
    if not pv.moves:
        return f"(empty line) → val {pv.value:+.2f}"
    seq = " ".join(_format_color_pos(a, m) for a, m in pv.moves)
    return f"{seq} → val {pv.value:+.2f}"


def _format_pvs_block(pvs: list[PrincipalVariation], max_lines: int = 3) -> str:
    if not pvs:
        return "(no MCTS lines available)"
    selected = pvs[:max_lines]
    return "\n".join(f"  {_format_pv_line(pv)}" for pv in selected)


# ---------------------------- templates


# Each template is a callable: (GameSignal, rng) → str. We list six so the
# renderer picks uniformly — three "single-move" forms and three
# "PV-based" forms that narrate MCTS exploration. Word-pool randomisation
# inside each template avoids the LLM memorising a fixed surface form.

def _tpl_neutral(s: GameSignal, rng: np.random.Generator) -> str:
    color = "Black" if s.mover == 0 else "White"
    verb = _pick(rng, _VERB_CHOSE)
    val_word = _pick(rng, _NOUN_VALUE)
    alt_word = _pick(rng, _NOUN_ALTERNATIVE)
    parts = [
        f"Move {s.move_number}: {color} {verb} {_format_pos(s.selected_pos)} "
        f"with confidence {s.selected_confidence:.2f}.",
        f"Top {alt_word}: {_format_alts(s.top_alternatives)}.",
        f"{val_word.capitalize()} before the move was {s.value_before:+.2f}; "
        f"{_format_value_delta(s.value_delta)}.",
    ]
    if s.captures:
        parts.append(f"{s.captures} opponent stone(s) were captured.")
    parts.append(f"This is the {s.game_phase} of the game.")
    return " ".join(parts)


def _tpl_concise(s: GameSignal, rng: np.random.Generator) -> str:
    color = "B" if s.mover == 0 else "W"
    pos = _format_pos(s.selected_pos)
    cap = f", {s.captures} capture(s)" if s.captures else ""
    delta = "" if s.value_delta is None else f", Δv={s.value_delta:+.2f}"
    return (
        f"[{s.game_phase}/{s.move_number}] {color} {pos} "
        f"(p={s.selected_confidence:.2f}, v={s.value_before:+.2f}{delta}{cap})"
    )


def _tpl_narrative(s: GameSignal, rng: np.random.Generator) -> str:
    color = "Black" if s.mover == 0 else "White"
    verb = _pick(rng, _VERB_CHOSE)
    intro_phrase = {
        "opening": "In the opening,",
        "middlegame": "In the middlegame,",
        "endgame": "In the endgame,",
    }[s.game_phase]

    if s.is_pass:
        body = f"{color} passed."
    else:
        body = (
            f"{color} {verb} {_format_pos(s.selected_pos)} "
            f"(probability {s.selected_confidence:.2f})."
        )

    if s.top_alternatives:
        alts_text = (
            f" Other candidates considered were "
            f"{_format_alts(s.top_alternatives)}."
        )
    else:
        alts_text = ""

    if s.value_delta is not None:
        if s.value_delta > 0.05:
            outcome = (
                f" After the move, the position improved for {color} "
                f"(value moved from {s.value_before:+.2f} to "
                f"{s.value_before + s.value_delta:+.2f})."
            )
        elif s.value_delta < -0.05:
            outcome = (
                f" After the move, the position weakened for {color} "
                f"(value moved from {s.value_before:+.2f} to "
                f"{s.value_before + s.value_delta:+.2f})."
            )
        else:
            outcome = (
                f" After the move, the value barely changed "
                f"(stayed near {s.value_before:+.2f})."
            )
    else:
        outcome = ""

    captures_text = (
        f" {s.captures} opponent stone(s) were captured." if s.captures else ""
    )

    return f"{intro_phrase} {body}{alts_text}{outcome}{captures_text}"


# ----- PV-based templates (3) — narrate MCTS exploration -----------------


def _tpl_pv_lines(s: GameSignal, rng: np.random.Generator) -> str:
    """List top-3 PVs as lines, then state the choice.

    Example:
        Considered:
          B(4, 4) W(2, 3) B(5, 6) W(7, 1) B(0, 0) → val +0.35
          B(2, 3) W(4, 4) B(7, 7) → val +0.05
          B(8, 8) W(1, 1) → val -0.15
        Chose B(4, 4): highest expected outlook.
    """
    consider_word = _pick(rng, _VERB_CONSIDER)
    val_word = _pick(rng, _NOUN_VALUE)
    pos = _format_pos(s.selected_pos)
    color = "B" if s.mover == 0 else "W"
    chose_verb = _pick(rng, _VERB_CHOSE)
    body = f"{consider_word}:\n{_format_pvs_block(s.principal_variations)}"
    tail = (
        f"\n{chose_verb.capitalize()} {color}{pos}: highest expected {val_word}."
    )
    return body + tail


def _tpl_imagination(s: GameSignal, rng: np.random.Generator) -> str:
    """Narrative: 'I imagined PV1; alternatives were worse.'"""
    if not s.principal_variations:
        return _tpl_neutral(s, rng)
    color = "Black" if s.mover == 0 else "White"
    verb_imagine = _pick(rng, _VERB_IMAGINE)
    main = s.principal_variations[0]
    main_seq = " then ".join(
        _format_color_pos(a, m) for a, m in main.moves
    ) or "no move"
    parts = [
        f"{color} {verb_imagine}: {main_seq} (val {main.value:+.2f})."
    ]
    if len(s.principal_variations) > 1:
        alts = []
        for pv in s.principal_variations[1:3]:
            head = pv.moves[0] if pv.moves else None
            if head is None:
                continue
            alts.append(
                f"{_format_color_pos(*head)} (val {pv.value:+.2f})"
            )
        if alts:
            parts.append(
                f"Other lines starting with " + " or ".join(alts) + " were weaker."
            )
    parts.append(
        f"Decision: {_format_color_pos(s.selected_action, s.mover)}."
    )
    return " ".join(parts)


def _tpl_comparison(s: GameSignal, rng: np.random.Generator) -> str:
    """Brief comparative summary by PV value."""
    if not s.principal_variations:
        return _tpl_concise(s, rng)
    color = "B" if s.mover == 0 else "W"
    pos = _format_pos(s.selected_pos)
    rows = []
    for pv in s.principal_variations[:3]:
        head = pv.moves[0] if pv.moves else None
        if head is None:
            continue
        rows.append(f"{_format_color_pos(*head)} val {pv.value:+.2f}")
    rows_text = "; ".join(rows) if rows else "(no lines)"
    chose = _pick(rng, _VERB_CHOSE)
    return f"[{s.game_phase}/{s.move_number}] {rows_text}. {chose.capitalize()} {color}{pos}."


_TEMPLATES = [
    _tpl_neutral,
    _tpl_concise,
    _tpl_narrative,
    _tpl_pv_lines,
    _tpl_imagination,
    _tpl_comparison,
]


def render_explanation(
    signal: GameSignal, rng: np.random.Generator | None = None
) -> str:
    """Render a signal using a randomly-chosen template + word variation."""
    rng = rng if rng is not None else np.random.default_rng()
    tpl = _TEMPLATES[int(rng.integers(0, len(_TEMPLATES)))]
    return tpl(signal, rng)


# ============================================================ convenience


def synthesize(
    env_before: GoEnv,
    action: int,
    env_after: GoEnv,
    policy_logits: torch.Tensor,
    value: float,
    value_after: float | None = None,
    principal_variations: list[PrincipalVariation] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GameSignal, str]:
    """One-shot: extract signal then render. Returns ``(signal, text)``."""
    signal = extract_signal(
        env_before,
        action,
        env_after,
        policy_logits,
        value,
        value_after,
        principal_variations=principal_variations,
    )
    return signal, render_explanation(signal, rng=rng)


__all__ = [
    "GameSignal",
    "extract_signal",
    "render_explanation",
    "synthesize",
]
