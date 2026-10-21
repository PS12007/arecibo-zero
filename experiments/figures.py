"""All figures for the report and README (results/*.json[l] -> docs/fig/*.png)."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

from common import ROOT, load  # noqa: E402

FIG = f"{ROOT}/docs/fig"
os.makedirs(FIG, exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#a9c8ef", "#2a78d6", "#0d3d7a"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def radix_label(base, lsb):
    if base == 1:
        return "unary"
    return f"base {base}" + (" (LSB first)" if lsb else "")


def mean_table(recs, arch, metric="add_heldout"):
    t = defaultdict(list)
    for r in recs:
        if r["arch"] != arch:
            continue
        d = r["design"]
        t[(d["base"], d["lsb_first"], r["budget"])].append(r["scores"][metric])
    return {k: float(np.mean(v)) for k, v in t.items()}, {k: len(v) for k, v in t.items()}


def heat(ax, rows, budgets, tab, title):
    M = np.full((len(rows), len(budgets)), np.nan)
    for i, (b, l) in enumerate(rows):
        for j, B in enumerate(budgets):
            M[i, j] = tab.get((b, l, B), np.nan)
    im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(budgets)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{100*M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(budgets)), [f"$2^{{{int(np.log2(B))}}}$" for B in budgets])
    ax.set_yticks(range(len(rows)), [radix_label(*r) for r in rows])
    ax.set_xlabel("message length (bits)")
    ax.set_title(title, loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_radix():
    recs = load(f"{ROOT}/results/radix.jsonl")
    if not recs:
        return
    rows = [(1, False)] + [(b, l) for b in (2, 3, 4, 6, 10, 16) for l in (False, True)]
    budgets = sorted({r["budget"] for r in recs if r["arch"] == "tf"})
    tab, _ = mean_table(recs, "tf")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [5, 2]})
    im = heat(axes[0], rows, budgets, tab, "Transformer receivers (mean of 2 seeds)")
    tab_l, _ = mean_table(recs, "lstm")
    lb = sorted({r["budget"] for r in recs if r["arch"] == "lstm"})
    heat(axes[1], rows, lb, tab_l, "LSTM receivers")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("held-out sums answered correctly (%)")
    cb.set_ticks([0, .5, 1], labels=["0", "50", "100"])
    fig.suptitle("Which way of writing numbers does a mind with no priors understand?", x=0.05, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(f"{FIG}/radix.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_radix()
    print("figures ->", FIG)
