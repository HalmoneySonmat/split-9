"""Batched PUCT Monte-Carlo Tree Search.

Runs N independent MCTS searches in lockstep, batching network forwards
across the N leaves reached at each simulation step. The point is throughput:
a batch=1 forward on RTX 3070 Ti uses ~5–10% of the GPU, while batch=8 uses
60–90%. Same simulation count, ~5× wall-clock speedup.

Algorithmic equivalence to single-MCTS:
    BatchedMCTS(N=1) is identical (in expectation, sans RNG state) to MCTS.
    Each tree is independent — there is no information sharing between games.

What is *batched*:
    Only the *network forward* of leaves needing evaluation. CPU-side
    selection / backup remains per-tree (Python loops). On a single GPU
    this is the right granularity — the GPU forward is the bottleneck.

What is *not* in this module:
    * Virtual loss (would parallelise sims within a single tree). Not needed
      here because we parallelise across trees instead.
    * Tree reuse across moves. Each call to ``search_batch`` builds fresh
      trees. AlphaZero typically does not reuse trees either.
    * Async / coroutine machinery. Lockstep is simpler and adequate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..env.go_env import GoEnv
from .mcts import N_ACTIONS, Node

if TYPE_CHECKING:
    from .network import GoNet


class BatchedMCTS:
    """Lockstep PUCT MCTS over multiple independent games.

    Args:
        network: Shared GoNet. One forward per simulation step over the batch
            of leaves needing evaluation.
        n_simulations: Per-tree simulations. Same as single MCTS.
        c_puct, dirichlet_alpha, dirichlet_weight: PUCT parameters.
        device: Where to run forwards.
    """

    def __init__(
        self,
        network: "GoNet",
        n_simulations: int = 200,
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

    def search_batch(
        self,
        envs: list[GoEnv],
        add_root_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Run search for each environment, return their visit distributions.

        Args:
            envs: List of GoEnv instances. None may be terminal.
            add_root_noise: Whether to mix Dirichlet noise into root priors.
            rng: NumPy Generator. Same RNG used for all games' root noise
                and for batch-internal randomness; pass distinct seeds across
                separate ``search_batch`` calls if you want independent runs.

        Returns:
            List of (82,) ``np.float32`` distributions, one per input env.
        """
        for e in envs:
            if e.is_terminal():
                raise ValueError("Cannot search a terminal state")
        rng = rng if rng is not None else np.random.default_rng()

        roots = [Node(state=e.clone()) for e in envs]

        # Initial expansion of every root in one batch.
        self._batch_expand(roots)
        if add_root_noise:
            for r in roots:
                self._add_root_noise(r, rng)

        # Main loop. Each iteration advances every tree by one simulation.
        for _ in range(self.n_simulations):
            self._one_simulation_step(roots)

        return [self._visit_distribution(r) for r in roots]

    # ----------------------------------------------------- single sim step

    def _one_simulation_step(self, roots: list[Node]) -> None:
        """Advance every tree by exactly one simulation, sharing one forward."""
        # Phase 1: descend each tree to a leaf (CPU).
        paths: list[list[Node]] = []
        leaves: list[Node] = []
        for root in roots:
            path = [root]
            node = root
            while node.is_expanded and not node.is_terminal:
                action = self._select_action_in_node(node)
                if action not in node.children:
                    child_state = node.state.clone()
                    child_state.step(action)
                    child = Node(state=child_state, parent=node)
                    if child_state.is_terminal():
                        child.is_terminal = True
                        mover = node.state.current_player()
                        returns = child_state.returns()
                        child.terminal_value = -float(returns[mover])
                    node.children[action] = child
                node = node.children[action]
                path.append(node)
            paths.append(path)
            leaves.append(node)

        # Phase 2: batch-evaluate non-terminal leaves with one forward.
        non_terminal_idx = [i for i, leaf in enumerate(leaves) if not leaf.is_terminal]
        if non_terminal_idx:
            obs_batch = torch.stack(
                [leaves[i].state.encode() for i in non_terminal_idx], dim=0
            ).to(self.device)
            was_training = self.network.training
            self.network.eval()
            try:
                with torch.no_grad():
                    policy_logits, values = self.network(obs_batch)
            finally:
                self.network.train(was_training)
            policies = F.softmax(policy_logits, dim=-1).cpu().numpy()
            values_arr = values.cpu().numpy()
        else:
            policies = np.zeros((0, N_ACTIONS), dtype=np.float32)
            values_arr = np.zeros((0,), dtype=np.float32)

        # Phase 3: per-tree expansion (for newly-eval'd leaves) + backup.
        eval_cursor = 0
        for i, (path, leaf) in enumerate(zip(paths, leaves)):
            if leaf.is_terminal:
                leaf_value = leaf.terminal_value
            else:
                policy = policies[eval_cursor]
                value = float(values_arr[eval_cursor])
                eval_cursor += 1
                self._populate_prior(leaf, policy)
                leaf_value = value

            # Backup: leaf → root, alternating sign.
            v = leaf_value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += v
                v = -v

    # ------------------------------------------------------------ helpers

    def _select_action_in_node(self, node: Node) -> int:
        """PUCT formula. Same as single MCTS."""
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score = -math.inf
        best_action = -1
        c_puct = self.c_puct
        for action, prior in node.prior.items():
            if action in node.children:
                child = node.children[action]
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

    def _batch_expand(self, nodes: list[Node]) -> None:
        """Run network on every node's state in one forward and populate
        priors. Used for the initial root expansion in ``search_batch``."""
        if not nodes:
            return
        obs_batch = torch.stack([n.state.encode() for n in nodes], dim=0).to(
            self.device
        )
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                policy_logits, _ = self.network(obs_batch)
        finally:
            self.network.train(was_training)
        policies = F.softmax(policy_logits, dim=-1).cpu().numpy()
        for node, policy in zip(nodes, policies):
            self._populate_prior(node, policy)

    def _populate_prior(self, node: Node, policy_probs: np.ndarray) -> None:
        """Mask illegal actions, renormalize, store as ``node.prior``."""
        legal = node.state.legal_actions()
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_mask[legal] = 1.0
        masked = policy_probs * legal_mask
        total = masked.sum()
        if total > 0:
            masked = masked / total
        else:
            masked = legal_mask / max(1, legal_mask.sum())
        node.prior = {a: float(masked[a]) for a in legal}
        node.is_expanded = True

    def _add_root_noise(self, root: Node, rng: np.random.Generator) -> None:
        legal = list(root.prior.keys())
        if not legal:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(legal))
        w = self.dirichlet_weight
        for action, eps in zip(legal, noise):
            root.prior[action] = (1.0 - w) * root.prior[action] + w * float(eps)

    def _visit_distribution(self, root: Node) -> np.ndarray:
        dist = np.zeros(N_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            dist[action] = child.visit_count
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist


__all__ = ["BatchedMCTS"]
