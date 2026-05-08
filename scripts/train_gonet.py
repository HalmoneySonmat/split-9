"""Entry point: train GoNet via self-play + alternating learning cycles.

Usage:
    python scripts/train_gonet.py                  # uses defaults (PoC)
    python scripts/train_gonet.py --config configs/train/selfplay.yaml

The YAML file is a flat mapping of TrainConfig field names to values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from split_brain_go.training.joint_train import TrainConfig, train_gonet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file overriding TrainConfig defaults.",
    )
    args = parser.parse_args()

    if args.config is not None:
        with args.config.open() as f:
            data = yaml.safe_load(f) or {}
        cfg = TrainConfig(**data)
    else:
        cfg = TrainConfig()

    print("Training with config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k:24s} = {v}")
    print()

    final = train_gonet(cfg)
    print()
    print(f"Final winrate vs random: {final['winrate']:.3f}")
    print(f"Final loss:              {final['loss']:.4f}")


if __name__ == "__main__":
    main()
