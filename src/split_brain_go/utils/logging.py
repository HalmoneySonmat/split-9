"""Training logger with CSV (default) and optional wandb backends.

Why this exists. Long-running training (4–6 weeks for Phase 1.3b) needs
metric persistence; otherwise debugging a divergence after the fact is
impossible. wandb is the standard, but it's overkill for a one-person
research project — and requires a cloud account. We default to CSV on
disk (zero setup) and offer wandb as an opt-in.

Backends:
    * ``csv`` (default): writes ``runs/<run_name>/metrics.csv`` and
      ``config.json``. Plot any time with pandas + matplotlib (see
      ``plot_run`` helper). No external dependencies.
    * ``wandb``: cloud dashboard. Requires ``pip install wandb`` +
      ``wandb login``. Falls back to noop if wandb not available.
    * ``noop``: silent. Useful for tests.

Schema flexibility. CSV is rewritten when a new metric key first appears,
so callers may add or drop keys mid-run without crashing. The cost (full
rewrite) is negligible for typical cycle frequencies.

Crash safety. CSV is flushed after every ``log`` call. If training crashes,
the CSV through the last successful cycle is on disk.

Usage:
    >>> with TrainLogger(run_name="phase1-baseline",
    ...                  config={"lr": 1e-3}) as logger:
    ...     for cycle in range(N):
    ...         logger.log({"loss": loss, "winrate": wr}, step=cycle)
    >>> # Later:
    >>> from split_brain_go.utils.logging import plot_run
    >>> plot_run("runs/phase1-baseline", metrics=["loss", "winrate"])
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
from typing import Any


def _is_disabled_env() -> bool:
    return os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}


# ============================================================ Logger


class TrainLogger:
    """Backend-agnostic training logger.

    Default backend is CSV. wandb is optional and best-effort: in
    ``mode="wandb"`` it raises if wandb isn't installed, but in
    ``mode="auto"`` it falls back to CSV silently.
    """

    def __init__(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "csv",
        run_dir: str | Path = "runs",
        project: str = "split-brain-go",
        tags: list[str] | None = None,
    ) -> None:
        if mode not in {"csv", "wandb", "auto", "noop"}:
            raise ValueError(f"Invalid mode {mode!r}; use csv/wandb/auto/noop")

        self.mode = mode
        self.run_name = run_name or make_run_name("run")
        self.config = dict(config or {})
        self._project = project
        self._tags = list(tags or [])
        self._wandb = None
        self._wandb_run = None

        # CSV state
        self._rows: list[dict[str, Any]] = []
        self._csv_columns: list[str] = []
        self._run_dir: Path | None = None
        self._csv_path: Path | None = None
        self._active = False

        if mode == "noop":
            return

        # Determine actual backend
        actual = mode
        if mode == "auto":
            if _is_disabled_env() or not _wandb_available():
                actual = "csv"
            else:
                actual = "wandb"

        if actual == "csv":
            self._init_csv(run_dir)
        elif actual == "wandb":
            ok = self._init_wandb(project, run_name, config, tags)
            if not ok and mode == "auto":
                # wandb init failed; degrade to CSV
                self._init_csv(run_dir)
            elif not ok:
                raise RuntimeError("wandb requested but init failed")

        self.actual_mode = actual

    # ------------------------------------------------------------ backends

    def _init_csv(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir) / self.run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._run_dir / "metrics.csv"
        cfg_path = self._run_dir / "config.json"
        cfg_path.write_text(json.dumps(self.config, indent=2, default=str))
        self._active = True

    def _init_wandb(
        self,
        project: str,
        run_name: str | None,
        config: dict | None,
        tags: list | None,
    ) -> bool:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            return False
        try:
            run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or [],
                reinit="finish_previous",
            )
        except Exception:
            return False
        self._wandb = wandb
        self._wandb_run = run
        self._active = True
        return True

    # --------------------------------------------------------------- public

    @property
    def is_active(self) -> bool:
        return self._active

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._active:
            return
        row = dict(metrics)
        if step is not None:
            row["step"] = step

        if self._wandb_run is not None:
            self._wandb_run.log(row)
            return

        # CSV path
        self._rows.append(row)
        self._write_csv()

    def update_config(self, **kwargs: Any) -> None:
        if not self._active:
            return
        self.config.update(kwargs)
        if self._wandb_run is not None:
            self._wandb_run.config.update(kwargs, allow_val_change=True)
        elif self._run_dir is not None:
            (self._run_dir / "config.json").write_text(
                json.dumps(self.config, indent=2, default=str)
            )

    def watch(self, model: Any, log_freq: int = 100) -> None:
        """Track gradients + parameter histograms (wandb only; noop on CSV)."""
        if self._wandb is None or self._wandb_run is None:
            return
        self._wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        # CSV: nothing to close (we flush on every write).
        self._active = False

    @property
    def csv_path(self) -> Path | None:
        return self._csv_path

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    # -------------------------------------------------------------- private

    def _write_csv(self) -> None:
        """Rewrite the full CSV. Cheap for typical cycle counts (≤ 10k rows)."""
        # Union of all keys ever logged, sorted for stability with 'step' first.
        all_keys: set[str] = set()
        for r in self._rows:
            all_keys.update(r.keys())
        ordered = ["step"] if "step" in all_keys else []
        ordered += sorted(k for k in all_keys if k != "step")
        self._csv_columns = ordered
        with self._csv_path.open("w", newline="") as f:  # type: ignore[union-attr]
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in ordered})

    # ------------------------------------------------------ context manager

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


# ============================================================ helpers


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        return False


def make_run_name(prefix: str, *parts: str) -> str:
    """Compose a filesystem-friendly run name: ``prefix-2026-05-07-suffix``."""
    today = datetime.date.today().isoformat()
    bits = [prefix, today, *parts]
    return "-".join(b for b in bits if b)


def plot_run(
    run_dir: str | Path,
    metrics: list[str] | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Plot one or more metrics from a CSV-mode run.

    Args:
        run_dir: Path to ``runs/<run_name>/``.
        metrics: Which columns to plot. Defaults to every numeric column
            other than 'step'.
        out_path: If given, save to this path; else display interactively.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "metrics.csv")

    if metrics is None:
        metrics = [c for c in df.columns if c != "step" and df[c].dtype.kind in "fi"]

    x = df["step"] if "step" in df.columns else df.index
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ax.plot(x, df[m], marker=".", linewidth=1)
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()


__all__ = ["TrainLogger", "make_run_name", "plot_run"]
