"""Generate Phase 3 training dataset from a trained Go-Net.

Plays self-play games with the trained network, captures every move's
state + MCTS principal variations, and saves the resulting list of
``Phase3Example`` objects to disk.

Usage:
    python scripts/generate_phase3_data.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --n-games 100 \\
        --n-simulations 200 \\
        --output runs/phase3_data.pkl

For a quick PoC dataset (≈ 30 minutes on RTX 3070 Ti):
    python scripts/generate_phase3_data.py \\
        --gonet-ckpt runs/checkpoints/best.pt \\
        --n-games 30 \\
        --n-simulations 100 \\
        --output runs/phase3_data_small.pkl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from split_brain_go.data.generation import generate_dataset, save_dataset
from split_brain_go.gonet.network import GoNet, GoNetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gonet-ckpt",
        type=Path,
        required=True,
        help="Path to a trained Go-Net checkpoint (e.g. runs/checkpoints/best.pt).",
    )
    p.add_argument(
        "--gonet-config",
        choices=["poc", "default"],
        default="default",
        help="GoNetConfig variant matching the checkpoint.",
    )
    p.add_argument("--n-games", type=int, default=100)
    p.add_argument("--n-simulations", type=int, default=200)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase3_data.pkl"),
    )
    p.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="How many sample explanations to print after generation.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.gonet_ckpt.is_file():
        raise FileNotFoundError(f"Go-Net checkpoint not found: {args.gonet_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = (
        GoNetConfig.poc() if args.gonet_config == "poc" else GoNetConfig.default()
    )
    net = GoNet(cfg).to(device)
    ckpt = torch.load(args.gonet_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"Loaded Go-Net from {args.gonet_ckpt}")
    print(
        f"Generating {args.n_games} games × {args.n_simulations} simulations "
        f"(temperature={args.temperature})"
    )

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    examples = generate_dataset(
        net,
        n_games=args.n_games,
        n_simulations=args.n_simulations,
        max_moves=args.max_moves,
        temperature=args.temperature,
        rng=rng,
        progress_every=max(1, args.n_games // 20),
    )
    elapsed = time.time() - t0
    avg_per_game = elapsed / max(1, args.n_games)
    avg_per_example = elapsed / max(1, len(examples))
    print(
        f"\nGenerated {len(examples)} examples in {elapsed:.1f}s "
        f"({avg_per_game:.1f}s/game, {avg_per_example:.2f}s/example)"
    )

    save_dataset(examples, args.output)
    print(f"Saved → {args.output}")

    if args.show_samples > 0 and examples:
        n_samp = min(args.show_samples, len(examples))
        idxs = [int(i * (len(examples) - 1) / max(1, n_samp - 1)) for i in range(n_samp)]
        print("\n--- Sample explanations ---")
        for i in idxs:
            ex = examples[i]
            print(f"\n[example {i} | game {ex.game_id} | move {ex.move_number}]")
            print(ex.explanation)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
