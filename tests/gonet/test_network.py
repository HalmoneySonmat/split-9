"""Unit tests for GoNet — shapes, parameter counts, gradients, activation hooks."""

from __future__ import annotations

import pytest
import torch

from split_brain_go.gonet.network import (
    BOARD_SIZE,
    N_ACTIONS,
    STEM_LAYER_ID,
    GoNet,
    GoNetConfig,
)

# ---------------------------------------------------------------- shapes


@pytest.fixture()
def poc_net() -> GoNet:
    return GoNet(GoNetConfig.poc())


def test_forward_shapes(poc_net):
    B = 3
    board = torch.zeros(B, 8, BOARD_SIZE, BOARD_SIZE)
    policy, value = poc_net(board)
    assert policy.shape == (B, N_ACTIONS)
    assert value.shape == (B,)


def test_value_in_tanh_range(poc_net):
    poc_net.eval()
    board = torch.randn(5, 8, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        _, value = poc_net(board)
    assert value.min().item() >= -1.0 - 1e-5
    assert value.max().item() <= 1.0 + 1e-5


def test_policy_logits_unbounded(poc_net):
    """Logits are raw; not constrained. Sanity check: not all zeros."""
    poc_net.eval()
    board = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        policy, _ = poc_net(board)
    assert not torch.allclose(policy, torch.zeros_like(policy))


# ------------------------------------------------------- parameter counts


def test_poc_param_count_in_range():
    net = GoNet(GoNetConfig.poc())
    n = net.num_parameters()
    # Loose bounds — order of magnitude check.
    assert 200_000 <= n <= 1_000_000, f"PoC params={n} outside [200k, 1M]"


def test_default_param_count_in_range():
    net = GoNet(GoNetConfig.default())
    n = net.num_parameters()
    assert 1_500_000 <= n <= 5_000_000, f"Default params={n} outside [1.5M, 5M]"


# ------------------------------------------------------------- gradients


def test_forward_backward_runs(poc_net):
    """Both heads contribute to a loss and their gradients flow."""
    poc_net.train()
    board = torch.randn(4, 8, BOARD_SIZE, BOARD_SIZE)
    policy_target = torch.randint(0, N_ACTIONS, (4,))
    value_target = torch.randn(4)

    policy, value = poc_net(board)
    loss = (
        torch.nn.functional.cross_entropy(policy, policy_target)
        + torch.nn.functional.mse_loss(value, value_target)
    )
    loss.backward()

    # All parameters got gradients
    for name, p in poc_net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero grad for {name}"


# ----------------------------------------------------- forward_with_acts


def test_forward_with_acts_returns_correct_keys(poc_net):
    poc_net.eval()
    board = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE)
    requested = [STEM_LAYER_ID, 0, 2]  # stem, block 0, block 2
    with torch.no_grad():
        policy, value, acts = poc_net.forward_with_acts(board, requested)
    assert policy.shape == (2, N_ACTIONS)
    assert value.shape == (2,)
    assert set(acts.keys()) == set(requested)
    for tensor in acts.values():
        assert tensor.shape == (2, poc_net.config.channels, BOARD_SIZE, BOARD_SIZE)


def test_forward_with_acts_empty_list_matches_forward(poc_net):
    poc_net.eval()
    board = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        p1, v1 = poc_net(board)
        p2, v2, acts = poc_net.forward_with_acts(board, [])
    assert torch.allclose(p1, p2)
    assert torch.allclose(v1, v2)
    assert acts == {}


def test_forward_with_acts_invalid_id_raises(poc_net):
    board = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    with pytest.raises(ValueError):
        poc_net.forward_with_acts(board, layers=[99])


def test_acts_are_independent_clones(poc_net):
    """Mutating a captured activation must not corrupt later forwards."""
    poc_net.eval()
    board = torch.randn(1, 8, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        _, _, acts = poc_net.forward_with_acts(board, [0])
        captured = acts[0]
        captured.zero_()  # mutate the clone

        # Subsequent forward must produce unchanged output
        policy_after, value_after = poc_net(board)
        policy_before, value_before, _ = poc_net.forward_with_acts(board, [])

    assert torch.allclose(policy_before, policy_after)
    assert torch.allclose(value_before, value_after)


# --------------------------------------------------------- config swap


def test_poc_and_default_configs_distinct():
    poc = GoNet(GoNetConfig.poc())
    full = GoNet(GoNetConfig.default())
    assert poc.num_parameters() < full.num_parameters()
    assert poc.config.n_blocks < full.config.n_blocks
    assert poc.config.channels < full.config.channels


def test_input_channel_count_is_8():
    net = GoNet(GoNetConfig.poc())
    # 8 came from ADR-008. If somebody changes the encoder out from under us,
    # this test fires.
    board = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    net(board)  # should not raise

    with pytest.raises(RuntimeError):
        net(torch.zeros(1, 4, BOARD_SIZE, BOARD_SIZE))  # OpenSpiel native ≠ ours
