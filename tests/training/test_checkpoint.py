"""Round-trip + best-tracking tests for CheckpointManager."""

from __future__ import annotations

import torch
from torch import nn

from split_brain_go.training.checkpoint import CheckpointManager


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_save_load_roundtrip(tmp_path):
    mgr = CheckpointManager(tmp_path, metric_name="winrate", higher_is_better=True)
    model_a = _tiny_model()
    optim_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    # Take one step so optimizer has state
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    loss = nn.functional.cross_entropy(model_a(x), y)
    loss.backward()
    optim_a.step()

    path = mgr.save(step=42, model=model_a, optimizer=optim_a, metric=0.65)

    # New model + optimizer; load and compare outputs
    model_b = _tiny_model()
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    meta = mgr.load(path, model_b, optim_b)

    assert meta.step == 42
    assert meta.metric == 0.65
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b)


def test_best_tracking_higher_is_better(tmp_path):
    mgr = CheckpointManager(tmp_path, metric_name="winrate", higher_is_better=True)
    m = _tiny_model()
    mgr.save(step=1, model=m, metric=0.40)
    mgr.save(step=2, model=m, metric=0.55)  # new best
    mgr.save(step=3, model=m, metric=0.50)  # worse, ignored

    assert (tmp_path / "best.pt").exists()
    import json

    best_meta = json.loads((tmp_path / "best.json").read_text())
    assert best_meta["metric"] == 0.55
    assert best_meta["step"] == 2


def test_keep_last_n(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last_n=2)
    m = _tiny_model()
    for s in [10, 20, 30, 40]:
        mgr.save(step=s, model=m, metric=float(s))
    surviving = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert surviving == ["step_000030.pt", "step_000040.pt"]
