"""Q5: evolve the message.  A small genetic search over designs maximises the
mean held-out addition accuracy of a *population of receivers*.  Running the
search with different populations (transformers only / LSTMs only / mixed)
and cross-testing the winners asks whether a message tuned to one kind of
mind is understood by another.

usage: python evolve.py POP   (POP in tf, lstm, lstm-u, mixed, transfer)"""
import json
import os
import sys

import numpy as np

from common import Design, run_grid, load, key_of, ROOT

BUDGET = int(os.environ.get("BUDGET", str(2**14)))
PATH = f"{ROOT}/results/evolve.jsonl"
POPS = {"tf": [("tf", 0), ("tf", 1)], "lstm": [("lstm", 0), ("lstm", 1)],
        "mixed": [("tf", 0), ("lstm", 0)], "lstm-u": [("lstm", 0), ("lstm", 1)]}
BASES = [2, 3, 4, 6, 10, 16]
# lstm-u: the LSTM search again, but tally marks (base 1) are allowed. §3 says
# unary is by far the best numeral for LSTMs; can the search find it unaided?
BASES_FOR = {"lstm-u": [1] + BASES}
KINDS = ["add", "count", "cmp", "mul"]


def genome_to_design(g):
    mix = {k: round(float(w), 2) for k, w in zip(KINDS, g["w"]) if w > 0.02}
    mix.setdefault("add", 0.3)
    return Design(base=int(g["base"]), lsb_first=bool(g["lsb"]), mix=mix)


def random_genome(rng, bases=BASES):
    return {"base": int(rng.choice(bases)), "lsb": int(rng.integers(2)),
            "w": [float(rng.uniform(0.4, 1.0))] + [float(rng.choice([0, rng.uniform(0, .5)])) for _ in range(3)]}


def mutate(g, rng, bases=BASES):
    g = json.loads(json.dumps(g))
    r = rng.random()
    if r < 0.3:
        i = bases.index(g["base"])
        g["base"] = bases[int(np.clip(i + rng.choice([-1, 1]), 0, len(bases) - 1))]
    elif r < 0.45:
        g["lsb"] = 1 - g["lsb"]
    else:
        k = int(rng.integers(4))
        g["w"][k] = float(np.clip(g["w"][k] + rng.normal(0, 0.2), 0 if k else 0.3, 1.0))
        if rng.random() < 0.3:
            g["w"][k] = 0.0 if k else g["w"][k]
    return g


def fitness(designs, pop):
    jobs = [(d, BUDGET, a, s, "evolve") for d in designs for a, s in pop]
    run_grid(PATH, jobs)
    recs = {key_of(r): r for r in load(PATH)}
    out = []
    for d in designs:
        vals = []
        for a, s in pop:
            k = key_of({"design": d.to_dict(), "budget": BUDGET, "arch": a, "seed": s, "tag": "evolve"})
            vals.append(recs[k]["scores"]["add_heldout"])
        out.append(float(np.mean(vals)))
    return out


def search(name, gens=4, size=8, elite=3):
    pop = POPS[name]
    rng = np.random.default_rng({"tf": 1, "lstm": 2, "mixed": 3, "lstm-u": 4}[name])
    bases = BASES_FOR.get(name, BASES)
    # seed the population with two classic human designs
    genomes = [{"base": 10, "lsb": 0, "w": [1, 0, 0, 0]}, {"base": 2, "lsb": 0, "w": [1, 0.3, 0, 0]}]
    genomes += [random_genome(rng, bases) for _ in range(size - len(genomes))]
    history = []
    for gen in range(gens + 1):
        fit = fitness([genome_to_design(g) for g in genomes], pop)
        order = np.argsort(fit)[::-1]
        history.append([{"genome": genomes[i], "fitness": fit[i]} for i in order])
        print(f"== {name} gen {gen}: best {fit[order[0]]:.3f} {genome_to_design(genomes[order[0]]).label()}",
              flush=True)
        if gen == gens:
            break
        parents = [genomes[i] for i in order[:elite]]
        children = [mutate(parents[int(rng.integers(elite))], rng, bases) for _ in range(size - elite)]
        genomes = parents + children
    with open(f"{ROOT}/results/evolve_{name}.json", "w") as f:
        json.dump({"budget": BUDGET, "pop": pop, "history": history}, f, indent=1)


def transfer():
    """Cross-test each search's champion on fresh receivers of every kind."""
    test = [(a, s) for a in ["tf", "tf-s", "tf-deep", "lstm", "lstm-s"] for s in (5, 6)]
    champs = {}
    for name in POPS:
        p = f"{ROOT}/results/evolve_{name}.json"
        if os.path.exists(p):
            h = json.load(open(p))["history"][-1][0]
            champs[name] = genome_to_design(h["genome"])
    champs["human-b10"] = Design(base=10)
    champs["human-b2"] = Design(base=2, mix={"add": 1, "count": 0.3})
    jobs = [(d, BUDGET, a, s, "transfer") for d in champs.values() for a, s in test]
    run_grid(PATH, jobs)
    recs = {key_of(r): r for r in load(PATH)}
    table = {}
    for name, d in champs.items():
        row = {}
        for a in sorted({a for a, _ in test}):
            v = [recs[key_of({"design": d.to_dict(), "budget": BUDGET, "arch": a, "seed": s, "tag": "transfer"})]
                 ["scores"]["add_heldout"] for aa, s in test if aa == a]
            row[a] = float(np.mean(v))
        table[name] = {"design": d.to_dict(), "label": d.label(), "scores": row}
        print(name, d.label(), row, flush=True)
    json.dump(table, open(f"{ROOT}/results/transfer.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1]
    transfer() if what == "transfer" else search(what)
