"""ELO rating from a pairwise result matrix.

Given a set of K agents and the outcome of pairwise games between every
pair, recover one ELO rating per agent. Formally a logistic-regression
fit: the probability that A beats B is modelled as

    P(A beats B) = 1 / (1 + 10^((R_B − R_A) / 400))

We take negative-log-likelihood of the observed game outcomes and minimise
over (R_1, …, R_K) by gradient descent (PyTorch). Anchor: ``R_anchor = 1500``
on the first agent so the ELO scale is fixed.

Use case for this project:
    Phase 1.3b's full training will produce a sequence of checkpoints
    (cycle 0, cycle 50, cycle 100, …). Running ``run_tournament`` with each
    checkpoint as an agent + computing ELO gives the *time series of
    learning progress* that vs Random can't sensitively reveal.

Bootstrap CI:
    Game-level resampling: each "trial" reshuffles the games via bootstrap,
    refits ELO, and we report the per-agent 95% percentile interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .winrate import Agent, eval_pairwise


# ============================================================ data


@dataclass
class TournamentResult:
    """Per-pair game outcomes for K agents.

    ``wins[i, j]`` is the number of times agent i beat agent j.
    Draws are split: each draw contributes 0.5 to ``wins[i, j]`` and 0.5
    to ``wins[j, i]``. ``games[i, j]`` is the total games played.
    """

    agent_names: list[str]
    wins: np.ndarray   # (K, K) float; wins[i,j] = #games i beat j (+ 0.5*draws)
    games: np.ndarray  # (K, K) int; symmetric, games[i,j] == games[j,i]

    @property
    def n_agents(self) -> int:
        return len(self.agent_names)


# ============================================================ runner


def run_tournament(
    agents: dict[str, Agent],
    n_games_per_pair: int = 20,
    max_moves: int = 200,
    rng: np.random.Generator | None = None,
) -> TournamentResult:
    """Round-robin: every pair (i, j), i < j, plays ``n_games_per_pair`` games.

    Color is alternated by ``eval_pairwise`` already, so we don't need to
    swap who plays Black externally.
    """
    rng = rng if rng is not None else np.random.default_rng()
    names = list(agents.keys())
    K = len(names)
    wins = np.zeros((K, K), dtype=np.float64)
    games = np.zeros((K, K), dtype=np.int64)

    for i in range(K):
        for j in range(i + 1, K):
            result = eval_pairwise(
                agents[names[i]], agents[names[j]],
                n_games=n_games_per_pair, max_moves=max_moves, rng=rng,
            )
            # eval_pairwise returns from agent_a (= names[i])'s POV.
            wins[i, j] = result.a_wins + 0.5 * result.draws
            wins[j, i] = result.b_wins + 0.5 * result.draws
            counted = result.a_wins + result.b_wins + result.draws
            games[i, j] = counted
            games[j, i] = counted

    return TournamentResult(agent_names=names, wins=wins, games=games)


# ============================================================ ELO fit


def fit_elo(
    result: TournamentResult,
    anchor_idx: int = 0,
    anchor_rating: float = 1500.0,
    lr: float = 50.0,
    n_steps: int = 2000,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Estimate ELO ratings via gradient descent on the log-likelihood.

    The Bradley–Terry / Elo model says:
        P(i beats j) = sigmoid((R_i - R_j) * ln(10) / 400)

    The likelihood is shift-invariant — adding a constant to every
    rating gives identical probabilities. We anchor *by construction*:
    ratings are computed as

        R_k = anchor_rating + free_k - free[anchor_idx]

    so R[anchor_idx] = anchor_rating regardless of the free parameter
    values. This avoids the quadratic-penalty trick which is numerically
    unstable (penalty + high lr → NaN).

    Optimizer is Adam (adaptive step), since loss curvature differs by
    orders of magnitude across pairs (50/100 wins vs 100/100 wins).

    Args:
        result: Tournament outcomes.
        anchor_idx: Which agent's rating is fixed.
        anchor_rating: Value of the anchored rating.
        lr: Adam step size. 10 is robust for ELO scale.
        n_steps: Iteration count. K ≤ 20 converges in ~1000–2000 steps.

    Returns:
        ``{agent_name: rating}`` with anchor exactly equal.
    """
    K = result.n_agents
    if K == 0:
        return {}
    if K == 1:
        return {result.agent_names[0]: anchor_rating}

    device = torch.device(device)
    rows, cols = np.triu_indices(K, k=1)
    wins_ij = torch.tensor(
        result.wins[rows, cols], dtype=torch.float64, device=device
    )
    games_ij = torch.tensor(
        result.games[rows, cols], dtype=torch.float64, device=device
    )

    free = torch.zeros(K, dtype=torch.float64, device=device, requires_grad=True)
    optim = torch.optim.Adam([free], lr=lr)
    LN10_OVER_400 = float(np.log(10) / 400.0)

    for _ in range(n_steps):
        optim.zero_grad()
        # Anchor by construction: R[anchor_idx] always equals anchor_rating.
        ratings = anchor_rating + free - free[anchor_idx]
        diff = ratings[rows] - ratings[cols]
        p_i_wins = torch.sigmoid(LN10_OVER_400 * diff)
        eps = 1e-9
        nll = -(
            wins_ij * torch.log(p_i_wins + eps)
            + (games_ij - wins_ij) * torch.log(1 - p_i_wins + eps)
        ).sum()
        nll.backward()
        optim.step()

    with torch.no_grad():
        final_t = anchor_rating + free - free[anchor_idx]
        final = final_t.cpu().numpy().tolist()
    # Snap anchor exactly to defeat any floating-point drift.
    final[anchor_idx] = anchor_rating
    return {name: float(r) for name, r in zip(result.agent_names, final)}


# ============================================================ bootstrap


def bootstrap_elo_ci(
    result: TournamentResult,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int | None = None,
    **fit_kwargs,
) -> dict[str, tuple[float, float]]:
    """Bootstrap each pair's wins to estimate ELO uncertainty.

    Resampling is *over individual games per pair*: for pair (i, j) with
    ``games[i,j] = N`` games where i won W_ij of them, we draw a Binomial
    sample with the same N games and observed win frequency.

    Returns a dict of (low, high) per agent.
    """
    rng = np.random.default_rng(seed)
    K = result.n_agents
    samples: list[np.ndarray] = []

    rows, cols = np.triu_indices(K, k=1)
    for _ in range(n_bootstrap):
        boot_wins = result.wins.copy()
        for r, c in zip(rows, cols):
            n = int(result.games[r, c])
            if n == 0:
                continue
            p_i = result.wins[r, c] / n
            new_wins_i = float(rng.binomial(n, p_i))
            boot_wins[r, c] = new_wins_i
            boot_wins[c, r] = n - new_wins_i
        boot_result = TournamentResult(
            agent_names=result.agent_names,
            wins=boot_wins,
            games=result.games.copy(),
        )
        ratings = fit_elo(boot_result, **fit_kwargs)
        samples.append(np.array([ratings[name] for name in result.agent_names]))

    arr = np.stack(samples, axis=0)  # (n_bootstrap, K)
    alpha = (1 - confidence) / 2
    lo = np.quantile(arr, alpha, axis=0)
    hi = np.quantile(arr, 1 - alpha, axis=0)
    return {
        name: (float(lo[i]), float(hi[i]))
        for i, name in enumerate(result.agent_names)
    }


# ============================================================ pretty


def format_ratings_table(
    ratings: dict[str, float],
    cis: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Render a sorted text table. Useful for prints."""
    sorted_names = sorted(ratings.keys(), key=lambda n: -ratings[n])
    lines = [f"{'agent':<24}{'rating':>10}"]
    if cis is not None:
        lines[0] += f"{'95% CI':>22}"
    lines.append("-" * len(lines[0]))
    for n in sorted_names:
        line = f"{n:<24}{ratings[n]:>10.1f}"
        if cis is not None and n in cis:
            lo, hi = cis[n]
            line += f"  [{lo:>7.1f}, {hi:>7.1f}]"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "TournamentResult",
    "run_tournament",
    "fit_elo",
    "bootstrap_elo_ci",
    "format_ratings_table",
]
