"""Q4: the receiver gets raw bits (unknown symbol width), possibly corrupted.
Is repetition coding worth its bits, or is arithmetic its own error-correcting
code?  Budget counts every transmitted bit, including the repeats."""
import json
import os
import sys
import time

from common import ROOT, Design, Codec, build_message, train_ensemble, Member, device, load
sys.path.insert(0, ROOT)
from arecibo.channel import BitChannel, bit_exam  # noqa: E402

quick = "--quick" in sys.argv
base = int(os.environ.get("BASE", "10"))
lsb = os.environ.get("LSB", "1") == "1"
path = f"{ROOT}/results/channel{'_quick' if quick else ''}.jsonl"
noises = [0.0, 0.03] if quick else [0.0, 0.01, 0.03, 0.1]
reps = [1, 3]
budgets = [2**16]
seeds = [0] if quick else [0, 1]
CTX = 512
STEPS = int(os.environ.get("STEPS", "8000"))
if os.environ.get("LONG"):          # follow-up: clean channel, 4x the training
    noises, reps = [0.0], [1]

done = {(r["noise"], r["rep"], r["budget"], r["seed"], r["arch"], r.get("steps", 8000)) for r in load(path)}
dev = device()
jobs = [(budget, noise, rep, seed) for budget in budgets for noise in noises for rep in reps
        for seed in seeds if (noise, rep, budget, seed, "tf", STEPS) not in done]
K = 4
for c0 in range(0, len(jobs), K):
    chunk = jobs[c0:c0 + K]
    t0 = time.time()
    built = []
    for budget, noise, rep, seed in chunk:
        d = Design(base=base, lsb_first=lsb)
        codec = Codec(d)
        ch = BitChannel(codec, rep=rep, noise=noise, seed=seed)
        # the symbol-level message must fit budget / rep transmitted bits
        sym_budget = budget / rep * codec.bits_per_token / ch.w
        toks, codec, info = build_message(d, sym_budget, seed=seed)
        built.append((ch, toks, info, ch.encode(toks)))
    ens, losses = train_ensemble([b[3] for b in built], 2, [j[3] for j in chunk], ctx=CTX,
                                 steps=STEPS, dev=dev)
    for k, ((budget, noise, rep, seed), (ch, toks, info, bits)) in enumerate(zip(chunk, built)):
        sc = bit_exam(Member(ens, k), ch, toks, CTX, dev=dev)
        rec = dict(noise=noise, rep=rep, budget=budget, seed=seed, arch="tf", base=base, lsb=lsb,
                   bits=len(bits), steps=STEPS, statements=info["statements"], loss=float(losses[k]), scores=sc,
                   secs=(time.time() - t0) / len(chunk), engine="ensemble")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
    del ens
