"""Shared runner: build a message, train a receiver on it, examine it, and
append one JSON line per run (resumable)."""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arecibo.message import Design, build_message  # noqa: E402
from arecibo.receivers import train_receiver, exam, device  # noqa: E402
from arecibo.message import Codec  # noqa: E402
from arecibo.ensemble import train_ensemble, Member  # noqa: E402

ENS_K = int(os.environ.get("ENS_K", "8"))


def ctx_for(design):
    return 256 if design.base == 1 else 128


def key_of(rec):
    return json.dumps([rec["design"], rec["budget"], rec["arch"], rec["seed"], rec.get("tag", "")],
                      sort_keys=True)


def load(path):
    """Read a results file plus any shard files written next to it."""
    import glob
    out = []
    for p in [path] + sorted(glob.glob(path + ".shard*")):
        if os.path.exists(p):
            with open(p) as f:
                out += [json.loads(l) for l in f if l.strip()]
    return out


def run_grid(path, jobs, steps=8000):
    """jobs: list of (Design, budget_bits, arch, seed[, tag])."""
    done = {key_of(r) for r in load(path)}
    dev = device()
    with open(os.path.join(ROOT, "runs", f"pid_{os.getpid()}"), "w") as f:
        f.write(" ".join(sys.argv))
    todo = []
    for j in jobs:
        d, budget, arch, seed = j[:4]
        tag = j[4] if len(j) > 4 else ""
        rec = {"design": d.to_dict(), "budget": budget, "arch": arch, "seed": seed, "tag": tag}
        if key_of(rec) not in done:
            todo.append((d, budget, arch, seed, tag, rec))
    # SHARD=i/n splits the remaining jobs across parallel processes
    shard = os.environ.get("SHARD")
    if shard:
        i, n = map(int, shard.split("/"))
        todo = todo[i::n]
        path = f"{path}.shard{i}"
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    # transformer jobs: train ENS_K receivers at once (same ctx), others one by one
    ens_jobs = [j for j in todo if j[2] == "tf"]
    single = [j for j in todo if j[2] != "tf"]
    groups = {}
    for j in ens_jobs:
        groups.setdefault(ctx_for(j[0]), []).append(j)
    chunks = [(c, g[i:i + ENS_K]) for c, g in groups.items() for i in range(0, len(g), ENS_K)]
    n_done = 0
    for ctx, chunk in chunks:
        t0 = time.time()
        built = [build_message(d, budget, seed=seed) for d, budget, arch, seed, tag, rec in chunk]
        V = max(c.V for _, c, _ in built)
        ens, losses = train_ensemble([m for m, _, _ in built], V, [j[3] for j in chunk], ctx=ctx,
                                     steps=steps, dev=dev)
        for k, ((d, budget, arch, seed, tag, rec), (msg, codec, info)) in enumerate(zip(chunk, built)):
            scores = exam(Member(ens, k), codec, msg, ctx=ctx, dev=dev)
            rec.update(label=d.label(), info=info, loss=float(losses[k]), steps=steps, ctx=ctx,
                       scores=scores, secs=(time.time() - t0) / len(chunk), engine="ensemble")
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done += 1
            print(f"[{n_done}/{len(todo)}] {d.label():28s} {budget:>7} {arch:7s} s{seed} "
                  f"loss {losses[k]:.3f} {scores}", flush=True)
        del ens
    todo = single
    for i, (d, budget, arch, seed, tag, rec) in enumerate(todo):
        t0 = time.time()
        msg, codec, info = build_message(d, budget, seed=seed)
        ctx = ctx_for(d)
        model, loss = train_receiver(msg, codec.V, arch=arch, ctx=ctx, steps=steps, seed=seed, dev=dev)
        scores = exam(model, codec, msg, ctx=ctx, dev=dev)
        rec.update(label=d.label(), info=info, loss=loss, steps=steps, ctx=ctx, scores=scores,
                   secs=time.time() - t0)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i+1}/{len(todo)}] {d.label():28s} {budget:>7} {arch:7s} s{seed} "
              f"loss {loss:.3f} {scores} {rec['secs']:.0f}s", flush=True)
