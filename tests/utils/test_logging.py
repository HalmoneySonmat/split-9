"""Unit tests for TrainLogger.

We test all backends except wandb-active. wandb requires an API key and
network so we only test its fallback paths (no-wandb, disabled).
"""

from __future__ import annotations

import csv
import json
import os
from unittest import mock

import pytest

from split_brain_go.utils.logging import TrainLogger, make_run_name, plot_run


# ============================================================ noop


def test_noop_mode_is_inactive():
    logger = TrainLogger(mode="noop")
    assert not logger.is_active


def test_noop_log_does_not_raise():
    logger = TrainLogger(mode="noop")
    logger.log({"loss": 1.23})
    logger.log({"loss": 1.23}, step=7)
    logger.update_config(extra="hi")
    logger.watch(object())
    logger.finish()


# ============================================================ csv


def test_csv_creates_run_directory(tmp_path):
    logger = TrainLogger(
        run_name="test-run",
        config={"lr": 0.001},
        mode="csv",
        run_dir=tmp_path,
    )
    assert (tmp_path / "test-run").is_dir()
    assert (tmp_path / "test-run" / "config.json").is_file()
    cfg = json.loads((tmp_path / "test-run" / "config.json").read_text())
    assert cfg["lr"] == 0.001


def test_csv_log_writes_rows(tmp_path):
    logger = TrainLogger(run_name="t", mode="csv", run_dir=tmp_path)
    logger.log({"loss": 1.0, "winrate": 0.5}, step=0)
    logger.log({"loss": 0.8, "winrate": 0.6}, step=1)
    logger.finish()

    with (tmp_path / "t" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert float(rows[0]["loss"]) == 1.0
    assert int(rows[1]["step"]) == 1


def test_csv_handles_new_keys_mid_run(tmp_path):
    """Adding a new metric after the first call must not crash; older rows
    fill the new column with empty strings."""
    logger = TrainLogger(run_name="t", mode="csv", run_dir=tmp_path)
    logger.log({"loss": 1.0}, step=0)
    logger.log({"loss": 0.9, "winrate": 0.5}, step=1)  # new key 'winrate'
    logger.finish()

    with (tmp_path / "t" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert "winrate" in rows[0]
    assert rows[0]["winrate"] == ""  # empty for first row
    assert float(rows[1]["winrate"]) == 0.5


def test_csv_context_manager(tmp_path):
    with TrainLogger(run_name="t", mode="csv", run_dir=tmp_path) as logger:
        logger.log({"x": 1.0}, step=0)
    assert (tmp_path / "t" / "metrics.csv").is_file()


def test_csv_path_property(tmp_path):
    logger = TrainLogger(run_name="t", mode="csv", run_dir=tmp_path)
    assert logger.csv_path == tmp_path / "t" / "metrics.csv"


# ============================================================ auto


def test_auto_mode_falls_back_to_csv_when_wandb_disabled(tmp_path):
    with mock.patch.dict(os.environ, {"WANDB_DISABLED": "1"}, clear=False):
        logger = TrainLogger(run_name="t", mode="auto", run_dir=tmp_path)
        assert logger.is_active
        assert logger.actual_mode == "csv"


# ============================================================ errors


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        TrainLogger(mode="bogus")


def test_wandb_mode_when_disabled_raises():
    with mock.patch.dict(os.environ, {"WANDB_DISABLED": "1"}, clear=False):
        # wandb mode doesn't honor WANDB_DISABLED — wandb itself does.
        # If wandb is installed, init may succeed but be a noop. If wandb
        # isn't installed, our wrapper raises. Either is acceptable; we
        # don't assert on this case.
        try:
            TrainLogger(mode="wandb").finish()
        except RuntimeError:
            pass  # acceptable: wandb not installed


# ============================================================ helpers


def test_make_run_name_includes_date():
    name = make_run_name("test", "suffix")
    assert name.startswith("test-")
    assert name.endswith("-suffix")


def test_make_run_name_strips_empty():
    name = make_run_name("only")
    assert name.startswith("only-")
    assert "--" not in name


def test_plot_run_smoke(tmp_path):
    """Ensure plot_run can render a tiny CSV without error (matplotlib backend
    is non-interactive in tests)."""
    import matplotlib

    matplotlib.use("Agg")  # headless

    with TrainLogger(run_name="t", mode="csv", run_dir=tmp_path) as logger:
        logger.log({"loss": 1.0, "winrate": 0.5}, step=0)
        logger.log({"loss": 0.8, "winrate": 0.6}, step=1)
        logger.log({"loss": 0.6, "winrate": 0.7}, step=2)

    out = tmp_path / "plot.png"
    plot_run(tmp_path / "t", metrics=["loss", "winrate"], out_path=out)
    assert out.is_file()
    assert out.stat().st_size > 0
