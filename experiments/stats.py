"""Paired statistics for the headline claims (CPU only, from results/*.jsonl).

LSB vs MSB: every (base, budget, seed) cell where both orders were trained
gives one paired difference.  Reports wins, an exact two-sided sign test and
a bootstrap 95% interval for the mean difference (resampling cells)."""
import json
from math import comb

import numpy as np

from common import ROOT, load


def sign_test(wins, n):
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def boot_ci(x, rng, n=10000):
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def lsb_vs_msb(recs, arch, tag):
    cell = {}
    for r in recs:
        if r["arch"] != arch or (tag and r.get("tag") != tag) or r["design"]["base"] == 1:
            continue
        d = r["design"]
        cell[(d["base"], r["budget"], r["seed"], d["lsb_first"])] = r["scores"]["add_heldout"]
    diffs, per_base = [], {}
    for (b, B, s, lsb), v in cell.items():
        if lsb and (b, B, s, False) in cell:
            dlt = v - cell[(b, B, s, False)]
            diffs.append(dlt)
            per_base.setdefault(b, []).append(dlt)
    return diffs, per_base


def main():
    rng = np.random.default_rng(0)
    recs = load(f"{ROOT}/results/radix.jsonl")
    out = {}
    for arch, tag in (("tf", "ens"), ("lstm", None)):
        diffs, per_base = lsb_vs_msb(recs, arch, tag)
        if not diffs:
            continue
        d = np.array(diffs)
        live = d[np.abs(d) > 0.02]          # drop ties at the floor
        wins = int((live > 0).sum())
        out[arch] = {"pairs": len(d), "non_tied": len(live), "lsb_wins": wins,
                     "sign_p": sign_test(wins, len(live)),
                     "mean_diff": float(d.mean()), "ci95": boot_ci(d, rng),
                     "per_base_mean": {int(b): float(np.mean(v)) for b, v in sorted(per_base.items())}}
    # exam resolution: binomial standard error at the observed accuracies
    n_q = 400
    out["exam_questions"] = n_q
    out["exam_se_at_50pct"] = float(np.sqrt(0.25 / n_q))
    # seed spread: |seed0 - seed1| over every transformer radix cell
    by = {}
    for r in recs:
        if r["arch"] == "tf" and r.get("tag") == "ens":
            d = r["design"]
            by.setdefault((d["base"], d["lsb_first"], r["budget"]), []).append(r["scores"]["add_heldout"])
    gaps = [abs(v[0] - v[1]) for v in by.values() if len(v) == 2]
    out["seed_gap"] = {"cells": len(gaps), "median": float(np.median(gaps)), "max": float(np.max(gaps))}
    json.dump(out, open(f"{ROOT}/results/stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
