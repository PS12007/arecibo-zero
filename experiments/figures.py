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
        if r["arch"] != arch or (arch == "tf" and r.get("tag") != "ens"):
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
    lb = [2**14, 2**16]
    heat(axes[1], rows, lb, tab_l, "LSTM receivers")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("held-out sums answered correctly (%)")
    cb.set_ticks([0, .5, 1], labels=["0", "50", "100"])
    fig.suptitle("Which way of writing numbers does a mind with no priors understand?", x=0.05, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(f"{FIG}/radix.png")
    plt.close(fig)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_content():
    recs = [r for r in load(f"{ROOT}/results/content.jsonl") if r["arch"] == "tf"]
    if not recs:
        return
    names = ["add", "add+count", "add+cmp", "add+mul", "all"]
    budgets = sorted({r["budget"] for r in recs})
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(budgets)
    for j, B in enumerate(budgets):
        for i, n in enumerate(names):
            v = [r["scores"]["add_heldout"] for r in recs if r["design"]["name"] == n and r["budget"] == B]
            if not v:
                continue
            x = i + (j - (len(budgets) - 1) / 2) * w
            ax.bar(x, np.mean(v), w * 0.9, color=SERIES[0], alpha=0.35 + 0.65 * j / max(1, len(budgets) - 1),
                   label=f"$2^{{{int(np.log2(B))}}}$ bits" if i == 0 else None)
            ax.plot([x] * len(v), v, "o", ms=2.5, color=INK, alpha=0.6)
    ax.set_xticks(range(len(names)), ["sums only", "+ counting", "+ comparisons", "+ products", "everything"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out sums correct")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("Same bit budget, different curriculum (base 10, LSB first, transformer)", loc="left", color=INK)
    style(ax)
    fig.savefig(f"{FIG}/content.png")
    plt.close(fig)


def fig_channel():
    recs = [r for r in load(f"{ROOT}/results/channel.jsonl") if r.get("steps", 8000) == 8000]
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for c, rep in zip(SERIES, (1, 3)):
        ns = sorted({r["noise"] for r in recs if r["rep"] == rep})
        m = [np.mean([r["scores"]["add_heldout"] for r in recs if r["rep"] == rep and r["noise"] == n]) for n in ns]
        ax.plot(ns, m, "-o", color=c, lw=2, ms=5, label="no repetition" if rep == 1 else "each bit sent 3x")
    ax.set_xlabel("bit-flip probability")
    ax.set_ylabel("held-out sums correct")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Raw noisy bits, fixed total budget ($2^{16}$ bits)", loc="left", color=INK)
    style(ax)
    fig.savefig(f"{FIG}/channel.png")
    plt.close(fig)


def fig_evolve():
    from evolve import genome_to_design
    names = {"tf": "transformers", "lstm": "LSTMs", "lstm-u": "LSTMs, unary allowed", "mixed": "mixed"}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    any_, n = False, 0
    for c, name in zip(SERIES, names):
        p = f"{ROOT}/results/evolve_{name}.json"
        if not os.path.exists(p):
            continue
        any_ = True
        h = json.load(open(p))["history"]
        n = max(n, len(h))
        best = [g[0]["fitness"] for g in h]
        med = [float(np.median([x["fitness"] for x in g])) for g in h]
        ax.plot(range(len(h)), best, "-o", color=c, lw=2, ms=5, label=names[name])
        ax.plot(range(len(h)), med, "--", color=c, lw=1.2, alpha=0.7)
        d = genome_to_design(h[-1][0]["genome"])
        lab = radix_label(d.base, d.lsb_first and d.base > 1)
        ax.annotate(lab, (len(h) - 1, best[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    if not any_:
        plt.close(fig)
        return
    ax.set_xticks(range(n))
    ax.set_xlim(-0.2, n - 0.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("held-out sums correct")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left", title="receivers (solid: best, dashed: median)",
              title_fontsize=8)
    ax.set_title("Evolving the message ($2^{14}$ bits)", loc="left", color=INK)
    style(ax)
    fig.savefig(f"{FIG}/evolve.png")
    plt.close(fig)

def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_numberline():
    """First principal component of the hidden state after each numeral, against
    the numeral's true value: the receiver's "number line" (seed 0 receivers)."""
    keys = ["b3-msb", "b4-lsb", "b6-msb", "b10-msb", "b10-lsb", "b16-lsb"]
    paths = [f"{ROOT}/results/numberline_{k}-s0.npz" for k in keys]
    if not all(os.path.exists(p) for p in paths):
        return
    dec = {}
    if os.path.exists(f"{ROOT}/results/decipher.json"):
        dec = json.load(open(f"{ROOT}/results/decipher.json"))
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), sharex=True)
    for ax, k, p in zip(axes.flat, keys, paths):
        z = np.load(p)
        pc, v = z["pc"], z["value"]
        ok = (v >= 0) & (v < 128)          # products above 127 are too rare to plot
        pc, v = pc[ok], v[ok]
        pc = (pc - pc.mean()) / (pc.std() + 1e-9)
        ax.plot(v, pc, "o", ms=2, color=SERIES[0], alpha=0.15, mec="none")
        vs = np.unique(v)
        med = [np.median(pc[v == u]) for u in vs]
        ax.plot(vs, med, "-", color=INK, lw=1.4)
        ax.axvline(63.5, color=INK2, lw=0.8, ls=":")
        b, o = k.split("-")
        rho = spearman(v, pc)
        full = dec.get(f"{k}-s0", {})
        read = (full.get("succ_exact") and full.get("comparison_correct")
                and all(full.get("operators_correct", {}).values())) if full else None
        tag = "" if read is None else ("  (deciphered)" if read else "  (not deciphered)")
        ax.set_title(f"base {b[1:]}, {o.upper()} first  ρ = {rho:+.2f}{tag}", loc="left", fontsize=8.5, color=INK)
        style(ax)
    fig.supxlabel("true value of the numeral (dotted line: operands are always below 64, sums reach 127)", fontsize=9,
                  color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("hidden-state PC1 (z)")
    fig.suptitle("PC1 of the hidden state after each numeral: it marks operand vs answer (step at 64), not size", x=0.05, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/numberline.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_radix()
    fig_content()
    fig_channel()
    fig_evolve()
    fig_numberline()
    print("figures ->", FIG)
