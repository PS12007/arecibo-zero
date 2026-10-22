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
        if r["arch"] == "lstm" and r["budget"] not in (2**14, 2**16):
            continue  # early smoke tests
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
        if r.get("steps", 8000) != 8000:
            continue   # the long-training follow-up is reported separately
        t[(r["noise"], r["rep"], r["budget"])].append(r["scores"]["add_heldout"])
    long_ = [r["scores"]["add_heldout"] for r in recs if r.get("steps", 8000) != 8000]
    if long_:
        out["channel_long"] = {"steps": max(r.get("steps", 8000) for r in recs), "acc": mean_sd(long_)}
    out["channel"] = [{"noise": n, "rep": rp, "budget": B, "acc": mean_sd(v)} for (n, rp, B), v in sorted(t.items())]

# --- decipher ------------------------------------------------------------------
p = f"{ROOT}/results/decipher.json"
if os.path.exists(p):
    out["decipher"] = json.load(open(p))
    for r in out["decipher"].values():
        if r.get("loss") != r.get("loss"):   # NaN: receiver loaded from cache, loss not recorded
            r["loss"] = None

# --- evolution -----------------------------------------------------------------
ev = {}
for name in ("tf", "lstm", "lstm-u", "mixed"):
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

FINDINGS = {
    "radix": "<b>Write the least significant digit first.</b> It wins in every base: at 2<sup>15</sup> bits, base 4 LSB-first gets 87% of unseen sums and MSB-first 27%. "
             "Base 4 needs the fewest bits to reach 50% (about 16k), ahead of base 10 (19k). Binary, the Arecibo choice, is the worst positional system: written MSB-first it never reaches 50%. "
             "<b>The best numerals depend on the mind:</b> LSTMs given unary tally marks answer 51–83% of unseen sums (three seeds) after hearing just 41 examples, while every positional system leaves them at chance.",
    "content": "<b>Nothing teaches addition except sums.</b> Accuracy is predicted almost exactly by how many sums fit in the message. Counting, the traditional opener, gives no head start. "
               "Products, which look just like sums, cost a further 8–11 points. The other skills are learned on their own (89% of comparisons and 83% of products at 2<sup>16</sup> bits), but they do not transfer.",
    "channel": "<b>Framing is everything.</b> Given the same 2<sup>16</sup> bits as raw 5-bit codewords instead of symbols, no receiver learned to add (about 1% even on sums it heard, with no noise and no repetition), "
               "against 91% when symbol boundaries are given. With every condition at the floor, noise and repetition coding cannot be compared. With 4× the training the receivers nearly memorise the bit stream (training loss down tenfold) but still answer only 4–5% of unseen sums: more training buys recitation, not understanding.",
    "evolve": "<b>Tuning a message to one mind over-fits it.</b> The search on transformers converged on base 4, LSB first; the search on LSTMs, once allowed tally marks, "
              "switched to them within two generations. Sent to fresh receivers of five kinds, each champion did well only on its own kind. "
              "Plain unary sums, which no search proposed, were the only design every kind understood: 43% on average, against 16% for the transformer champion, "
              "10% for decimal and 3% for Arecibo-style binary.",
    "decipher": "<b>Five of twelve shuffled messages were read back completely</b>: base, digit values, digit order, +, × and the comparison with its three outcomes. Both base-4 receivers are among them, and in each case the recovered + makes 100% of the decoded sums true. "
                "Digits and the separator were found exactly in every base up to 10. Receivers that understood poorly (base 3, MSB first: 27–30% of unseen sums) could not be read. "
                "Base 16 failed in the probe, not in the mind: its receivers score 81%, but only digits 0–7 ever lead a numeral and the role step missed the other nine.",
}
out["findings"] = FINDINGS

os.makedirs(f"{ROOT}/docs", exist_ok=True)
json.dump(out, open(f"{ROOT}/docs/data.json", "w"), allow_nan=False)
print("wrote docs/data.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
