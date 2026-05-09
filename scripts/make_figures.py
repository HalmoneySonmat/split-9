"""Render the README figures from measured data.

Outputs PNGs to ``docs/figures/`` (creates if missing). All numbers
are hard-coded from the measurement runs — re-run the probes and edit
this file if data changes.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- styling
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# Colors
C_BASE = "#2b6cb0"      # blue
C_TRAIN = "#2f855a"     # green
C_RANDOM = "#a0aec0"    # gray
C_PRIOR = "#dd6b20"     # orange
C_BOARD = "#9f7aea"     # purple


# ============================================================ Figure 1
# Baseline comparison bar chart


def fig_baselines() -> None:
    labels = ["Output-only", "Random Adapter", "Trained Adapter"]
    losses = [2.78, 2.78, 0.50]
    ppls = [16.14, 16.14, 1.65]
    colors = [C_BASE, C_RANDOM, C_TRAIN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(labels))

    bars = ax1.bar(x, losses, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10)
    ax1.set_ylabel("validation loss")
    ax1.set_title("Baselines: validation loss")
    for bar, v in zip(bars, losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.07,
                 f"{v:.2f}", ha="center", fontsize=10)
    ax1.set_ylim(0, max(losses) * 1.15)

    bars = ax2.bar(x, ppls, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=10)
    ax2.set_ylabel("validation perplexity")
    ax2.set_title("Baselines: perplexity")
    for bar, v in zip(bars, ppls):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                 f"{v:.2f}", ha="center", fontsize=10)
    ax2.set_ylim(0, max(ppls) * 1.15)

    fig.suptitle("Trained adapter vs no-adapter / random-adapter baselines",
                 fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "fig1_baselines.png")
    plt.close(fig)


# ============================================================ Figure 2
# IAS sweep curve


def fig_ias() -> None:
    ratios = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    losses = np.array([0.503, 0.507, 0.521, 0.556, 0.607])
    base = losses[0]
    delta_pct = (losses - base) / base * 100

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(ratios, losses, "o-", color=C_BASE, linewidth=2, markersize=8,
             label="loss")
    ax1.set_xlabel("channel mask ratio (r)")
    ax1.set_ylabel("validation loss", color=C_BASE)
    ax1.tick_params(axis="y", labelcolor=C_BASE)
    ax1.set_xticks(ratios)
    for r, l in zip(ratios, losses):
        ax1.annotate(f"{l:.3f}", (r, l), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=C_BASE)

    ax2 = ax1.twinx()
    ax2.plot(ratios, delta_pct, "s--", color=C_PRIOR, linewidth=1.4,
             markersize=6, alpha=0.7, label="Δloss vs base (%)")
    ax2.set_ylabel("Δloss vs base (%)", color=C_PRIOR)
    ax2.tick_params(axis="y", labelcolor=C_PRIOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for r, d in zip(ratios, delta_pct):
        ax2.annotate(f"{d:+.1f}%", (r, d), textcoords="offset points",
                     xytext=(0, -18), ha="center", fontsize=9, color=C_PRIOR)

    ax1.set_title("IAS sweep — convex curve indicates redundant channels")
    fig.savefig(OUT_DIR / "fig2_ias_sweep.png")
    plt.close(fig)


# ============================================================ Figure 3
# APC matched vs mismatched bar


def fig_apc() -> None:
    labels = ["matched", "mismatched"]
    losses = [0.503, 0.594]
    per_seed_matched = [0.503, 0.503, 0.503]
    per_seed_mismatched = [0.5956, 0.5963, 0.5899]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, losses, color=[C_TRAIN, C_PRIOR],
                  edgecolor="black", linewidth=0.5, width=0.55)

    # overlay per-seed scatter
    for xi, seeds in zip(x, [per_seed_matched, per_seed_mismatched]):
        ax.scatter([xi] * len(seeds), seeds, color="black",
                   zorder=3, s=24, alpha=0.7, label="per-seed" if xi == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("validation loss")
    ax.set_title("APC — board-mismatched activations hurt loss "
                 f"(APC = {(losses[1]-losses[0])/losses[0]:+.3f})")
    for bar, v in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", fontsize=10)

    # arrow showing the gap
    ax.annotate("", xy=(1, losses[1] - 0.005), xytext=(1, losses[0] + 0.005),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4})
    ax.text(1.1, (losses[0] + losses[1]) / 2,
            f"+{(losses[1]-losses[0])*100:.1f}% loss",
            va="center", fontsize=10)

    ax.set_ylim(0.45, 0.65)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT_DIR / "fig3_apc.png")
    plt.close(fig)


# ============================================================ Figure 4
# Loss decomposition stacked bar


def fig_loss_decomposition() -> None:
    """Show how the trained-adapter loss reduction breaks down."""
    output_only = 2.78
    fully_masked = 0.61
    matched = 0.50

    domain_prior = output_only - fully_masked    # 2.17 -> 95% of total reduction
    board_signal = fully_masked - matched        # 0.11 -> 5% of total reduction
    total_reduction = output_only - matched      # 2.28

    fig, ax = plt.subplots(figsize=(8, 4.4))

    # left bar: output-only loss
    ax.bar(0, output_only, color=C_BASE, edgecolor="black", linewidth=0.5,
           width=0.55, label="Output-only loss (2.78)")

    # right stacked bar: matched + components above it
    ax.bar(1, matched, color=C_TRAIN, edgecolor="black", linewidth=0.5,
           width=0.55, label="Trained matched loss (0.50)")
    ax.bar(1, board_signal, bottom=matched, color=C_BOARD,
           edgecolor="black", linewidth=0.5, width=0.55,
           label=f"per-board signal (+{board_signal:.2f}, ~5%)")
    ax.bar(1, domain_prior, bottom=matched + board_signal, color=C_PRIOR,
           edgecolor="black", linewidth=0.5, width=0.55,
           label=f"learned domain prior (+{domain_prior:.2f}, ~95%)")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no adapter", "trained adapter\n(decomposed)"])
    ax.set_ylabel("validation loss")
    ax.set_title("Where does the 2.78 → 0.50 loss reduction come from?")
    ax.set_ylim(0, output_only * 1.1)

    # arrows / annotations
    ax.text(0, output_only + 0.05, f"{output_only:.2f}",
            ha="center", fontsize=11)
    ax.text(1, matched / 2, "matched\n0.50", ha="center",
            color="white", fontsize=10, fontweight="bold")
    ax.text(1, matched + board_signal / 2,
            f"board\n+{board_signal:.2f}",
            ha="center", color="white", fontsize=9)
    ax.text(1, matched + board_signal + domain_prior / 2,
            f"prior\n+{domain_prior:.2f}",
            ha="center", color="white", fontsize=10, fontweight="bold")

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(OUT_DIR / "fig4_loss_decomposition.png")
    plt.close(fig)


# ============================================================ Figure 5
# Mode-collapse histogram from observed sample outputs


def fig_mode_collapse() -> None:
    """Histogram of output coordinates across the 10 sampled boards.

    Source: runs/samples.txt — extracted manually from observed outputs.
    """
    coords = {
        "(2,1)": 4,
        "(8,8)": 3,
        "pass":  3,
        "(8,0)": 1,
        "(8,1)": 1,
        "(1,1)": 1,
        # plus the actual ground-truth coordinates, none of which were emitted
        "actor's actual coord\n(0/10 emitted)": 0,
    }
    labels = list(coords.keys())
    counts = list(coords.values())
    colors = [C_BASE if c > 0 else C_PRIOR for c in counts]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(labels)), counts, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("# samples emitting this coord")
    ax.set_title("Mode collapse: trained outputs concentrate on a few "
                 "frequent coordinates\n(N = 10 sampled boards, "
                 "0 / 10 match the actual move)")
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, c + 0.08,
                    str(c), ha="center", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.4)
    fig.savefig(OUT_DIR / "fig5_mode_collapse.png")
    plt.close(fig)


# ============================================================ main


def main() -> int:
    print(f"Rendering figures to {OUT_DIR.resolve()}/ ...")
    fig_baselines()
    print("  fig1_baselines.png")
    fig_ias()
    print("  fig2_ias_sweep.png")
    fig_apc()
    print("  fig3_apc.png")
    fig_loss_decomposition()
    print("  fig4_loss_decomposition.png")
    fig_mode_collapse()
    print("  fig5_mode_collapse.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
