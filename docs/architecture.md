# Architecture — Interface specification (Phase 0)

This document fixes the *signatures* of the core modules so that Phases 1–3 can be implemented in any order without integration churn. It does **not** specify implementations. Where a detail is intentionally undecided, this is called out.

## 1. System diagram

```
         ┌─────────┐                           ┌─────────────────┐
state ──▶│ GoEnv   │── encode ──▶ board ──────▶│     GoNet       │
         └─────────┘                            ├──┬──────────────┤
                                                │  │ residual     │
                                                │  │ blocks       │
                                                │  └──────────────┤
                                                │     ↓ acts at   │
                                                │     selected    │
                                                │     layers      │
                                                │     ↓           │
                                                │  policy / value │
                                                └────┬─┬──────────┘
                                                     │ │
                                                     │ └────▶ chosen action
                                                     ▼
                                              ┌────────────┐
                                  acts ──────▶│ Adapter    │
                                              │ (Resampler │
                                              │  + Gated   │
                                              │   X-Attn)  │
                                              └────────────┘
                                                     │
                                                     ▼ act_tokens
                                              ┌────────────┐
                              prompt + ids ──▶│  LLM       │── text out
                                              │ (frozen,   │
                                              │ instrumented)│
                                              └────────────┘
```

## 2. Module signatures

### 2.1 `env.go_env`

```python
from dataclasses import dataclass

@dataclass
class StepInfo:
    legal: list[int]
    move_number: int

class GoEnv:
    def __init__(self, board_size: int = 9, komi: float = 7.5): ...

    def reset(self) -> tuple[torch.Tensor, StepInfo]:
        """Return (encoded_board, info) at the start of a new game."""

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, StepInfo]:
        """Apply action. Returns (encoded_board, reward, done, info).
        Reward is 0 except on terminal: +1/-1 from current player's POV."""

    def encode(self) -> torch.Tensor:
        """Current state as (C, B, B) tensor. C is fixed at construction."""

    def legal_actions(self) -> list[int]: ...

    def is_terminal(self) -> bool: ...

    def returns(self) -> tuple[float, float]: ...
```

**C (number of input channels)** — undecided in Phase 0. Likely 8–17 (own stones, opp stones, recent-N moves, turn indicator, illegal-mask). Decided in Phase 1.2 ADR.

### 2.2 `gonet.network`

```python
class GoNet(torch.nn.Module):
    def __init__(self, input_channels: int, n_blocks: int, channels: int): ...

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """board: (B, C, 9, 9) -> (policy_logits: (B, 82), value: (B,))."""

    def forward_with_acts(
        self,
        board: torch.Tensor,
        layers: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """Same as forward but also returns activations at requested layer ids.
        Activation tensor shape: (B, channels, 9, 9)."""
```

### 2.3 `gonet.mcts`

```python
class MCTS:
    def __init__(self, net: GoNet, n_simulations: int = 200, c_puct: float = 1.5): ...

    def search(self, env: GoEnv) -> tuple[int, np.ndarray]:
        """Return (best_action, visit_distribution_over_82_actions)."""
```

### 2.4 `adapter.xattn`

```python
class GoActivationAdapter(torch.nn.Module):
    def __init__(self, layer_channels: dict[int, int], n_tokens: int, d_model: int): ...

    def forward(self, acts: dict[int, torch.Tensor]) -> torch.Tensor:
        """acts: {layer_id: (B, C_l, 9, 9)} -> (B, n_tokens, d_model)."""
```

### 2.5 `llm.instrumented`

```python
class InstrumentedLLM(torch.nn.Module):
    """Wraps a frozen HF causal-LM and exposes injection points for cross-attn."""

    def __init__(self, model_id: str, inject_layers: list[int]): ...

    def forward(
        self,
        input_ids: torch.Tensor,
        act_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns next-token logits. If act_tokens is None, behaves as the
        frozen base model."""

    def generate(
        self,
        input_ids: torch.Tensor,
        act_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.Tensor: ...
```

### 2.6 `data.synthetic`

```python
@dataclass
class SyntheticExample:
    board: torch.Tensor              # (C, 9, 9)
    chosen_action: int
    activations: dict[int, torch.Tensor]
    explanation: str                 # template-generated English

def generate_examples(
    gonet: GoNet,
    env_factory: Callable[[], GoEnv],
    n_games: int,
    layers: list[int],
) -> Iterator[SyntheticExample]: ...
```

### 2.7 `eval` metrics

```python
def winrate(model_a, model_b, n_games: int = 200) -> tuple[float, tuple[float, float]]: ...

def patching_consistency(
    joint, dataset, layer: int,
    n_patches: int = 200, tau_d: float = 0.05, tau_e: float = 0.2,
) -> float: ...

def counterfactual_consistency(
    joint, dataset, top_k: int = 3,
) -> tuple[float, float]:  # (CFC_diff, CFC_align)
    ...

def information_ablation_score(
    joint, dataset, mask_ratio: float, seed_count: int = 5,
) -> float: ...
```

## 3. Cross-cutting conventions

- **Tensor names**. `board` for raw input, `act_l<id>` for activations at layer id, `tok_act` for adapter output tokens, `tok_text` for LLM input ids.
- **Devices**. All modules accept tensors on whatever device the caller provides; nothing internal calls `.cuda()`. Device pinning happens in scripts/.
- **dtypes**. bf16 for forward, fp32 for the optimizer state. Mixed precision via `torch.amp`.
- **Determinism**. Each script accepts `--seed`. Set once via `set_global_seed(seed)` in `src/split_brain_go/__init__.py`.
- **Config**. Hydra. Scripts take `--config-name=<name>` and override individual fields with `+key=value`.

## 4. What is **not** fixed yet

These are the open architectural questions whose answers will be added as future ADRs:

1. **C** — number of GoEnv input channels (Phase 1.2 / ADR-008).
2. **Selected GoNet layers for activation export** — which residual blocks feed the adapter (Phase 2.2 / ADR-009).
3. **N (adapter token count)** — output sequence length of the Perceiver Resampler (Phase 2.3 / ADR-010).
4. **LLM injection layers** — at which layers cross-attention blocks are inserted (Phase 2.3 / ADR-011).
5. **Synthetic template family** — the closed vocabulary of objective signals (Phase 3.1 / ADR-012).

We'll add each ADR at the moment the decision is made, not retroactively.
