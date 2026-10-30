# plot_training.py
#
# Training curves from a run's log.jsonl: results against the perfect player, move accuracy on
# oracle-labelled positions, and training losses, all against the number of self-play games.
# --baseline adds an evaluation of the untrained network as the point at 0 games.

import argparse
import json

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def line(ax, x, y, label, colour, fmt="{:.0f}%"):
    ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=5, label=label)
    ax.annotate(fmt.format(y[-1]), (x[-1], y[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--baseline", help="JSON eval record of the untrained network")
    ap.add_argument("--subtitle", default="", help="note printed above the panels")
    ap.add_argument("--out", default="docs/training_curves.png")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.log)]
    evals = [r for r in rows if "vs_perfect_D" in r]
    if args.baseline:
        evals.insert(0, json.load(open(args.baseline)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)

    ax = axes[0]
    g = [r["games"] for r in evals]
    n = [r["vs_perfect_W"] + r["vs_perfect_D"] + r["vs_perfect_L"] for r in evals]
    line(ax, g, [100 * r["vs_perfect_D"] / k for r, k in zip(evals, n)], "Draws (best possible)", SERIES[0])
    line(ax, g, [100 * r["vs_perfect_L"] / k for r, k in zip(evals, n)], "Losses", SERIES[1])
    ax.set_ylim(0, 105)
    style(ax, "Matches against the perfect player", "% of games")

    ax = axes[1]
    line(ax, g, [100 * r["mcts_opt"] for r in evals], "MCTS move", SERIES[0])
    line(ax, g, [100 * r["policy_opt"] for r in evals], "Raw policy move", SERIES[1])
    line(ax, g, [100 * r["value_acc"] for r in evals], "Value head W/D/L", SERIES[2])
    ax.set_ylim(0, 105)
    style(ax, "Accuracy vs the exact solution", "% correct")

    ax = axes[2]
    g_all = [r["games"] for r in rows]
    line(ax, g_all, [r["loss_p"] for r in rows], "Policy loss", SERIES[0], fmt="{:.2f}")
    line(ax, g_all, [r["loss_v"] for r in rows], "Value loss", SERIES[1], fmt="{:.2f}")
    ax.set_ylim(bottom=0)
    style(ax, "Training loss", "cross-entropy")

    for ax, loc in zip(axes, ["upper right", "lower left", "lower left"]):
        ax.set_xlabel("self-play games", color=TEXT_MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc=loc)
    if args.subtitle:
        fig.text(0.005, 0.985, args.subtitle, ha="left", va="top", fontsize=10, color=TEXT_MUTED)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
