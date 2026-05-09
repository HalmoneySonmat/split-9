"""PUCT Monte-Carlo Tree Search (AlphaGo Zero style).

This is the *PoC* implementation: single-game, single-thread, one network
forward per simulation. Phase 1.3b will replace this with a batched variant
that evaluates many leaves in one forward pass.

Sign conventions (the part most people get wrong):
    * Each node stores ``value_sum`` from the *current_player's* perspective.
    * When a parent picks a child via UCB, it flips the child's mean Q
      (because the child's perspective is the opponent's).
    * Backup walks leaf → root, flipping the propagated value at each step.
    * Terminal leaf: ``terminal_value`` is the value seen by the *current
      player at the terminal node*, which has no legal move. Equivalent
      formulation: negate the result for the player who just moved.

Tree layout:
    Node holds a clone of GoEnv. Children are created lazily — only when
    a simulation actually descends through that action. Priors are stored
    as a dict keyed by action id, so we don't carry around 82-element
    arrays for nodes deep in the tree where most actions are unvisited.

Public surface:
    MCTS(network, n_simulations=...).search(env)        -> visit distribution
    MCTS(...).select_action(env, temperature=...)        -> (action, dist)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..env.go_env import GoEnv

if TYPE_CHECKING:
    from .network import GoNet


N_ACTIONS = 82
PASS_ACTION = 81


@dataclass
class PrincipalVariation:
    """One main line from the MCTS tree.

    Attributes:
        moves: List of ``(action, mover)`` tuples, in play order.
            ``mover`` is 0 (Black) or 1 (White) — who played each move.
        visit_count: Visits at the *root* for this line's first move.
            Higher = MCTS preferred this line.
        value: Estimated game value at the PV's leaf, *from the root's
            current player's perspective*. Positive = good for root
            player; negative = bad.
    """

    moves: list[tuple[int, int]]
    visit_count: int
    value: float


# ====================================================================== Node


@dataclass
class Node:
    """A state node in the MCTS tree.

    Lazy: only ``state``, ``parent`` are required at construction. Priors and
    children dicts are populated by ``MCTS._expand`` on first visit.
    """

    state: GoEnv
    parent: Node | None = None

    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0  # only meaningful when is_terminal

    visit_count: int = 0
    value_sum: float = 0.0

    # Populated by _expand
    prior: dict[int, float] = field(default_factory=dict)
    children: dict[int, Node] = field(default_factory=dict)

    @property
    def Q(self) -> float:
        """Mean value at this node, from this node's current_player POV."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ====================================================================== MCTS


class MCTS:
    """PUCT MCTS searcher.

    Args:
        network: A trained or random ``GoNet`` instance. Must accept input of
            shape ``(B, 8, 9, 9)`` and return ``(policy_logits, value)``.
        n_simulations: Tree-rollouts per call to ``search``. ADR-012 sets
            PoC=100, full=200, eval=400.
        c_puct: PUCT exploration constant. ADR-011 fixes 1.5.
        dirichlet_alpha: Concentration parameter for the root noise.
            ADR-011 sets 0.25 (standard for boards under 19×19).
        dirichlet_weight: Mixture weight for the root noise (0..1).
        device: Where to run network forwards. The network itself can live
            on a different device; we move the input tensor.
    """

    def __init__(
        self,
        network: "GoNet",
        n_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.25,
        dirichlet_weight: float = 0.25,
        device: str | torch.device = "cpu",
    ) -> None:
        self.network = network
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.device = torch.device(device)

    # ----------------------------------------------------------- public API

    def search(
        self,
        env: GoEnv,
        add_root_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run ``n_simulations`` from a copy of ``env``.

        Returns an ``(82,)`` array of *visit-count proportions* over actions.
        """
        root = self._search_internal(env, add_root_noise, rng)
        return self._visit_distribution(root)

    def search_with_pvs(
        self,
        env: GoEnv,
        k: int = 3,
        max_depth: int = 5,
        add_root_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list["PrincipalVariation"]]:
        """Run search, return ``(visit_dist, top_k_principal_variations)``.

        Each principal variation is a greedy walk of most-visited children
        from one of the top-K root actions, up to ``max_depth`` plies.
        """
        root = self._search_internal(env, add_root_noise, rng)
        dist = self._visit_distribution(root)
        pvs = self._extract_top_k_pvs(root, k=k, max_depth=max_depth)
        return dist, pvs

    def _search_internal(
        self,
        env: GoEnv,
        add_root_noise: bool,
        rng: np.random.Generator | None,
    ) -> Node:
        """Common search core. Returns the populated root node."""
        if env.is_terminal():
            raise ValueError("Cannot run MCTS on a terminal state")
        rng = rng if rng is not None else np.random.default_rng()

        root = Node(state=env.clone())
        self._expand(root)
        if add_root_noise:
            self._add_root_noise(root, rng)

        for _ in range(self.n_simulations):
            self._simulate(root)

        return root

    def _extract_top_k_pvs(
        self, root: Node, k: int = 3, max_depth: int = 5
    ) -> list["PrincipalVariation"]:
        """Top-K principal variations from a populated MCTS tree.

        For each of the K most-visited root children, greedily walk the
        most-visited descendant chain up to ``max_depth`` plies.
        """
        if not root.children:
            return []

        sorted_root_actions = sorted(
            root.children.keys(),
            key=lambda a: -root.children[a].visit_count,
        )[:k]

        pvs: list[PrincipalVariation] = []
        for first_action in sorted_root_actions:
            moves: list[tuple[int, int]] = []
            node = root
            action = first_action
            for _depth in range(max_depth):
                if action not in node.children:
                    break
                child = node.children[action]
                mover = node.state.current_player()
                if mover not in (0, 1):
                    mover = 0
                moves.append((int(action), int(mover)))
                if child.is_terminal or not child.is_expanded or not child.children:
                    node = child
                    break
                # Next: most-visited descendant
                node = child
                action = max(
                    child.children.keys(),
                    key=lambda a: child.children[a].visit_count,
                )

            # Compute value at PV's end, from root's perspective.
            end_node = node
            if end_node.is_terminal:
                raw_value = end_node.terminal_value
            else:
                raw_value = end_node.Q
            # Sign convention: end_node's value is from end_node's mover POV.
            # Each move flips POV, so flip ((-1)^len(moves)).
            sign = (-1) ** len(moves)
            value_root_pov = float(raw_value * sign)

            pvs.append(
                PrincipalVariation(
                    moves=moves,
                    visit_count=int(root.children[first_action].visit_count),
                    value=value_root_pov,
                )
            )
        return pvs

    def select_action(
        self,
        env: GoEnv,
        temperature: float = 1.0,
        add_root_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        """Run a search and pick an action.

        Args:
            temperature: 0 means argmax; >0 sample proportional to
                ``visit ** (1/T)``.

        Returns:
            ``(action, visit_distribution)``. The distribution is the raw
            search output (before the temperature transform), which is what
            should be stored as the policy training target.
        """
        rng = rng if rng is not None else np.random.default_rng()
        dist = self.search(env, add_root_noise=add_root_noise, rng=rng)

        if temperature <= 0:
            action = int(np.argmax(dist))
        else:
            transformed = dist ** (1.0 / temperature)
            total = transformed.sum()
            if total <= 0:
                # Fallback: uniform over legal
                legal = env.legal_actions()
                action = int(rng.choice(legal))
            else:
                probs = transformed / total
                action = int(rng.choice(N_ACTIONS, p=probs))
        return action, dist

    # ------------------------------------------------------------- internals

    def _simulate(self, root: Node) -> None:
        """One PUCT simulation: select → expand or terminal-eval → backup."""
        path: list[Node] = [root]
        node = root

        # Selection: descend until we find a leaf (unexpanded) or a terminal.
        while node.is_expanded and not node.is_terminal:
            action = self._select_action_in_node(node)
            if action not in node.children:
                # Lazy child creation
                child_state = node.state.clone()
                child_state.step(action)
                child = Node(state=child_state, parent=node)
                if child_state.is_terminal():
                    child.is_terminal = True
                    # Player who just moved was *node's* current_player.
                    mover = node.state.current_player()
                    returns = child_state.returns()
                    # value at child (from "next-to-move" POV) = -returns[mover]
                    child.terminal_value = -float(returns[mover])
                node.children[action] = child
            node = node.children[action]
            path.append(node)

        # Evaluation
        if node.is_terminal:
            leaf_value = node.terminal_value
        else:
            leaf_value = self._expand(node)

        # Backup: walk leaf → root, flipping sign each step.
        v = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v

    def _select_action_in_node(self, node: Node) -> int:
        """PUCT formula. Caller has guaranteed node.is_expanded."""
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score = -math.inf
        best_action = -1
        c_puct = self.c_puct

        for action, prior in node.prior.items():
            if action in node.children:
                child = node.children[action]
                # Child Q is from child's POV; we want it from node's POV.
                q_for_node = -child.Q if child.visit_count > 0 else 0.0
                n_child = child.visit_count
            else:
                q_for_node = 0.0
                n_child = 0
            u = c_puct * prior * sqrt_total / (1 + n_child)
            score = q_for_node + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expand(self, node: Node) -> float:
        """Run network on the node's state, fill priors, return value.

        Returns:
            The network's value estimate, from *node.current_player*'s POV.

        We save and restore the network's training mode around the forward
        so MCTS is safe to call from any point in a training loop.
        """
        obs = node.state.encode().unsqueeze(0).to(self.device)
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                policy_logits, value = self.network(obs)
        finally:
            self.network.train(was_training)

        probs = F.softmax(policy_logits[0], dim=-1).cpu().numpy()  # (82,)
        legal = node.state.legal_actions()

        # Mask + renormalize. If everything got masked to 0 (numerically
        # impossible for softmax but safe-guarded), fall back to uniform.
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_mask[legal] = 1.0
        masked = probs * legal_mask
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            masked = legal_mask / max(1, legal_mask.sum())

        node.prior = {a: float(masked[a]) for a in legal}
        node.is_expanded = True
        return float(value.item())

    def _add_root_noise(self, root: Node, rng: np.random.Generator) -> None:
        """Mix Dirichlet noise into the root prior to encourage exploration."""
        legal = list(root.prior.keys())
        if not legal:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(legal))
        w = self.dirichlet_weight
        for action, eps in zip(legal, noise):
            root.prior[action] = (1.0 - w) * root.prior[action] + w * float(eps)

    def _visit_distribution(self, root: Node) -> np.ndarray:
        """Convert root child visit counts into an (82,) distribution."""
        dist = np.zeros(N_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            dist[action] = child.visit_count
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist


__all__ = [
    "MCTS",
    "Node",
    "PrincipalVariation",
    "N_ACTIONS",
    "PASS_ACTION",
]
