"""Collect everything the web demo needs into docs/data.json."""
import json
import os
from collections import defaultdict

import numpy as np

from common import ROOT, Design, Codec, build_message, load
import sys
sys.path.insert(0, ROOT)
from arecibo.channel import BitChannel  # noqa: E402


def mean_sd(v):
    return [float(np.mean(v)), float(np.std(v)), len(v)]


out = {}

# --- sample transmissions in several designs -------------------------------
samples = {}
for key, d in {"unary": Design(base=1), "b2": Design(base=2), "b2-lsb": Design(base=2, lsb_first=True),
               "b10": Design(base=10), "b10-lsb": Design(base=10, lsb_first=True),
               "b16": Design(base=16),
               "mixed": Design(base=10, lsb_first=True, mix={"add": .4, "count": .2, "cmp": .2, "mul": .2})}.items():
    msg, codec, info = build_message(d, 2**12, seed=3)
    samples[key] = {"vocab": codec.vocab, "tokens": msg[:160].tolist(), "label": d.label(),
                    "bits_per_symbol": codec.bits_per_token}
out["samples"] = samples

# Arecibo-sized raster (1679 = 23 x 73 bits) of a real bit-level transmission
d = Design(base=10, lsb_first=True)
codec = Codec(d)
ch = BitChannel(codec, rep=1)
msg, codec, _ = build_message(d, 2**12, seed=3)
out["raster"] = {"w": 23, "h": 73, "bits": ch.encode(msg)[:1679].tolist(), "symbol_width": ch.w}

# --- radix -------------------------------------------------------------------
recs = load(f"{ROOT}/results/radix.jsonl")
if recs:
    t = defaultdict(list)
    for r in recs:
        if r["arch"] == "tf" and r.get("tag") != "ens":
            continue  # earlier single-model runs: kept only as an engine cross-check
        dd = r["design"]
        t[(r["arch"], dd["base"], dd["lsb_first"], r["budget"])].append(r["scores"]["add_heldout"])
    out["radix"] = [{"arch": a, "base": b, "lsb": l, "budget": B, "acc": mean_sd(v)}
                    for (a, b, l, B), v in sorted(t.items())]

# --- content -------------------------------------------------------------------
recs = load(f"{ROOT}/results/content.jsonl")
if recs:
    t = defaultdict(list)
    for r in recs:
        for k, v in r["scores"].items():
            t[(r["arch"], r["design"]["name"], r["budget"], k)].append(v)
    out["content"] = [{"arch": a, "mix": n, "budget": B, "metric": k, "acc": mean_sd(v)}
                      for (a, n, B, k), v in sorted(t.items())]

# --- channel -------------------------------------------------------------------
recs = load(f"{ROOT}/results/channel.jsonl")
if recs:
    t = defaultdict(list)
    for r in recs:
        t[(r["noise"], r["rep"], r["budget"])].append(r["scores"]["add_heldout"])
    out["channel"] = [{"noise": n, "rep": rp, "budget": B, "acc": mean_sd(v)} for (n, rp, B), v in sorted(t.items())]

# --- decipher ------------------------------------------------------------------
p = f"{ROOT}/results/decipher.json"
if os.path.exists(p):
    out["decipher"] = json.load(open(p))

# --- evolution -----------------------------------------------------------------
ev = {}
for name in ("tf", "lstm", "mixed"):
    p = f"{ROOT}/results/evolve_{name}.json"
    if os.path.exists(p):
        h = json.load(open(p))
        ev[name] = {"budget": h["budget"],
                    "best_per_gen": [g[0]["fitness"] for g in h["history"]],
                    "mean_per_gen": [float(np.mean([x["fitness"] for x in g])) for g in h["history"]],
                    "champion": h["history"][-1][0]}
if ev:
    out["evolve"] = ev
p = f"{ROOT}/results/transfer.json"
if os.path.exists(p):
    out["transfer"] = json.load(open(p))

os.makedirs(f"{ROOT}/docs", exist_ok=True)
json.dump(out, open(f"{ROOT}/docs/data.json", "w"))
print("wrote docs/data.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
