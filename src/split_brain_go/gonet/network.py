"""GoNet — AlphaGo Zero-style policy/value network for 9x9 Go.

Architecture (per ADR-009 / ADR-010):
    input (B, 8, 9, 9)
       │
       ├─ Stem: Conv 3x3 → BN → ReLU
       │
       ├─ N × ResidualBlock  (Conv 3x3 → BN → ReLU → Conv 3x3 → BN → +residual → ReLU)
       │     ↑ activations for layers 0..N-1 are captured here when requested
       │
       ├─ PolicyHead: Conv 1x1 (→2ch) → BN → ReLU → flatten → Linear → (B, 82) logits
       │
       └─ ValueHead:  Conv 1x1 (→1ch) → BN → ReLU → flatten → Linear → ReLU → Linear → tanh → (B,)

Two configurations from ADR-009/010:
    PoC:    n_blocks=4, channels=64,  ≈ 0.5M parameters
    Full:   n_blocks=6, channels=128, ≈ 2.5M parameters

The ``forward_with_acts`` method exposes activations at any subset of residual
block outputs, plus the stem output (layer id ``-1``). This is the interface
that Phase 2's adapter will consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Constants — kept here so MCTS/encoder can import without circular deps.
BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1  # 82, includes pass


@dataclass(frozen=True)
class GoNetConfig:
    """Hyperparameters that fully define a GoNet instance."""

    input_channels: int = 8
    n_blocks: int = 4
    channels: int = 64
    value_hidden: int = 64
    board_size: int = BOARD_SIZE
    n_actions: int = N_ACTIONS

    @classmethod
    def poc(cls) -> "GoNetConfig":
        return cls(n_blocks=4, channels=64)

    @classmethod
    def default(cls) -> "GoNetConfig":
        return cls(n_blocks=6, channels=128)


# ============================================================ building blocks


class ResidualBlock(nn.Module):
    """Two 3x3 convs with batch norm, joined by a skip connection.

    The structure is the de-facto standard from AlphaGo Zero / ResNet:
    final ReLU is applied *after* the residual addition, which keeps
    activations non-negative and gradient flow clean.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyHead(nn.Module):
    """Maps trunk features to per-action logits over 82 actions (81 cells + pass)."""

    def __init__(self, channels: int, board_size: int, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        return self.fc(x)  # raw logits; softmax happens in caller / MCTS


class ValueHead(nn.Module):
    """Maps trunk features to a scalar value in [-1, 1] for the side to move."""

    def __init__(self, channels: int, board_size: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)  # (B,) not (B, 1)


# ===================================================================== GoNet


# Layer id for the stem (post-relu) — a magic constant so callers can request
# the input to the first residual block without an off-by-one negotiation.
STEM_LAYER_ID: int = -1


class GoNet(nn.Module):
    """Policy + value network over 9x9 Go boards.

    Args:
        config: Architectural hyperparameters. Use ``GoNetConfig.poc()`` for
            the small variant, ``GoNetConfig.default()`` for the full run.

    The two ``forward`` paths:
        forward(board) -> (policy_logits, value)
            Standard inference path. No activation capture.

        forward_with_acts(board, layers) -> (policy_logits, value, acts)
            Returns the same outputs plus a ``dict[int, Tensor]`` of
            activations at requested layer ids. Layer ids: ``-1`` = stem
            output (after ReLU); ``0..n_blocks-1`` = residual block outputs.

    Shape contract:
        Input  : (B, input_channels, 9, 9)
        Policy : (B, 82) raw logits
        Value  : (B,) in [-1, 1]
        Each captured act: (B, channels, 9, 9)
    """

    def __init__(self, config: GoNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or GoNetConfig.poc()
        c = self.config.channels

        self.stem_conv = nn.Conv2d(
            self.config.input_channels, c, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(c)

        self.blocks = nn.ModuleList(
            [ResidualBlock(c) for _ in range(self.config.n_blocks)]
        )

        self.policy_head = PolicyHead(
            channels=c,
            board_size=self.config.board_size,
            n_actions=self.config.n_actions,
        )
        self.value_head = ValueHead(
            channels=c,
            board_size=self.config.board_size,
            hidden=self.config.value_hidden,
        )

    # ----------------------------------------------------------------- forward

    def _trunk(self, board: Tensor) -> Tensor:
        """Stem + all residual blocks, no captures."""
        x = F.relu(self.stem_bn(self.stem_conv(board)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, board: Tensor) -> tuple[Tensor, Tensor]:
        x = self._trunk(board)
        return self.policy_head(x), self.value_head(x)

    def forward_with_acts(
        self,
        board: Tensor,
        layers: list[int],
    ) -> tuple[Tensor, Tensor, dict[int, Tensor]]:
        """Forward with activation capture at requested layer ids.

        Captured tensors are ``.clone()``-ed so callers can mutate or hold
        references without affecting subsequent forward passes.
        """
        if not layers:
            policy, value = self.forward(board)
            return policy, value, {}

        layer_set = set(layers)
        max_block_id = self.config.n_blocks - 1
        for lid in layer_set:
            if lid != STEM_LAYER_ID and not (0 <= lid <= max_block_id):
                raise ValueError(
                    f"Invalid layer id {lid}; must be {STEM_LAYER_ID} (stem) "
                    f"or in [0, {max_block_id}]"
                )

        acts: dict[int, Tensor] = {}

        x = F.relu(self.stem_bn(self.stem_conv(board)))
        if STEM_LAYER_ID in layer_set:
            acts[STEM_LAYER_ID] = x.clone()

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_set:
                acts[i] = x.clone()

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value, acts

    # ------------------------------------------------------------ introspection

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
