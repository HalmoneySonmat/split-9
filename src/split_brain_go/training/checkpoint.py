"""Checkpoint management.

Wraps the boilerplate of saving / loading a (model, optimizer, step, metrics)
tuple to a single safetensors-friendly dict. Tracks "best so far" via a
user-supplied metric. Disk overflow is the caller's responsibility; we only
track files we wrote.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointMeta:
    step: int
    metric: float
    metric_name: str
    extra: dict[str, Any]


class CheckpointManager:
    """Save / load checkpoints under a directory, track 'best' file.

    Layout under root_dir:
        step_000100.pt        # model + optimizer state
        step_000100.json      # metadata (step, metric, extra)
        ...
        best.pt -> step_NNN.pt   # symlink (or copy on Windows)
        best.json -> step_NNN.json

    Higher metric is better by default; set ``higher_is_better=False`` to
    invert (e.g. for loss).
    """

    def __init__(
        self,
        root_dir: str | Path,
        metric_name: str = "winrate",
        higher_is_better: bool = True,
        keep_last_n: int = 5,
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.keep_last_n = keep_last_n
        self._best_metric: float | None = None

    # ------------------------------------------------------------------ I/O

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        metric: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        meta = CheckpointMeta(
            step=step, metric=metric, metric_name=self.metric_name, extra=extra or {}
        )
        ckpt_path = self.root / f"step_{step:06d}.pt"
        meta_path = self.root / f"step_{step:06d}.json"

        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(payload, ckpt_path)
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

        self._maybe_update_best(metric, ckpt_path, meta_path)
        self._enforce_keep_last_n()
        return ckpt_path

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointMeta:
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            d = json.loads(meta_path.read_text())
            return CheckpointMeta(**d)
        return CheckpointMeta(
            step=int(payload.get("step", 0)),
            metric=0.0,
            metric_name=self.metric_name,
            extra={},
        )

    # -------------------------------------------------------------- helpers

    def _is_better(self, new: float) -> bool:
        if self._best_metric is None:
            return True
        return new > self._best_metric if self.higher_is_better else new < self._best_metric

    def _maybe_update_best(self, metric: float, ckpt: Path, meta: Path) -> None:
        if not self._is_better(metric):
            return
        self._best_metric = metric
        # Use copy rather than symlink — works cross-platform (incl. Windows).
        shutil.copy2(ckpt, self.root / "best.pt")
        shutil.copy2(meta, self.root / "best.json")

    def _enforce_keep_last_n(self) -> None:
        steps = sorted(self.root.glob("step_*.pt"))
        if len(steps) <= self.keep_last_n:
            return
        for old in steps[: -self.keep_last_n]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
