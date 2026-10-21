"""Q6: blind decipherment.  We receive a message in a design we do not know
(random radix, digit order and symbol labels), train a receiver on it, and
try to recover the whole design by reading the receiver.

Steps that read the *mind*: digit/operator roles (from the receiver's
next-symbol expectations), digit values (from the receiver's number line,
PC1 of end-of-numeral hidden states), digit order (whichever reading of the
numerals best matches the receiver's number line).  The final step reads
the *message*: with numerals decoded, each operator is matched to the
arithmetic relation that makes the most statements true."""
import itertools
import json
import os
import sys

import numpy as np
import torch

from common import ROOT, Design, build_message, train_receiver, device
sys.path.insert(0, ROOT)
from arecibo.decipher import (relabel, hidden_states, discover_roles,  # noqa: E402
                              recover_values, numeral_runs, spearman,
                              successor_probe, values_from_successor)

quick = "--quick" in sys.argv
BUDGET = 2**17
CTX = 128
dev = device()
out_path = f"{ROOT}/results/decipher{'_quick' if quick else ''}.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

cases = [(3, 0), (4, 1), (6, 0), (10, 1), (10, 0), (16, 1)]
if quick:
    cases = cases[3:4]
MIX = {"add": 0.5, "count": 0.2, "cmp": 0.15, "mul": 0.15}


def read_numerals(msg, value, lsb):
    runs = numeral_runs(msg, value >= 0)
    b = int((value >= 0).sum())
    vals = []
    for s, e in runs:
        ds = value[msg[s:e]]
        if lsb:
            ds = ds[::-1]
        v = 0
        for d in ds:
            v = v * b + int(d)
        vals.append(v)
    return runs, np.array(vals)


for base, lsb in cases:
    for seed in ([0] if quick else [0, 1]):
        key = f"b{base}-{'lsb' if lsb else 'msb'}-s{seed}"
        if key in results:
            continue
        d = Design(base=base, lsb_first=bool(lsb), mix=MIX)
        true_msg, codec, info = build_message(d, BUDGET, seed=seed)
        V = codec.V
        msg, perm = relabel(true_msg, V, seed=77 + seed)    # what we receive
        ck = f"{ROOT}/runs/decipher_{key}.pt"   # cache receivers so the analysis can be re-run
        if os.path.exists(ck):
            from arecibo.receivers import ARCHS
            model = ARCHS["tf"](V, CTX).to(dev)
            model.load_state_dict(torch.load(ck))
            model.eval()
            loss = float("nan")
        else:
            model, loss = train_receiver(msg, V, arch="tf", ctx=CTX, steps=8000, seed=seed, dev=dev)
            torch.save(model.state_dict(), ck)
        H, P = hidden_states(model, msg, CTX, dev=dev)

        # 1. roles
        is_digit, M, freq = discover_roles(msg, P, V)
        true_digit = np.zeros(V, bool)
        true_digit[perm[:base]] = True
        role_acc = float((is_digit == true_digit).mean())

        # 2. values from the receiver's number line
        order, pc, runs = recover_values(msg, H, is_digit)
        value = -np.ones(V, int)
        for v, sym in enumerate(order):
            value[sym] = v
        inv = np.argsort(perm)   # shown id -> true id
        true_vals = [int(inv[s]) for s in order]
        value_exact = bool(true_vals == sorted(true_vals) and is_digit.sum() == base)
        rho = spearman(np.arange(len(order)), true_vals) if len(order) > 1 else 0.0

        # 2b. values by asking the receiver to count
        sep, succ = successor_probe(model, msg, is_digit, CTX, dev=dev)
        chain, lsb_succ, agree = values_from_successor(succ)
        succ_vals = [int(inv[sch]) for sch in chain] if chain else []
        succ_exact = bool(succ_vals == list(range(base)))
        succ_frac = float(np.mean([a == b for a, b in zip(succ_vals, range(base))])) if chain else 0.0
        if chain and len(chain) == is_digit.sum():
            value = -np.ones(V, int)
            for v, sym in enumerate(chain):
                value[sym] = v

        # 3. digit order: which reading matches the mind's number line?
        fits = {}
        for rd in (0, 1):
            rr, vals = read_numerals(msg, value, rd)
            pcs = pc[: len(vals)]
            fits[rd] = spearman(pcs, vals[: len(pcs)])
        lsb_geom = int(fits[1] > fits[0])
        lsb_hat = lsb_succ if lsb_succ is not None else lsb_geom

        # 4. operators: decode statements and match relations
        rr, vals = read_numerals(msg, value, lsb_hat)
        ops = [s for s in range(V) if not is_digit[s]]
        # statements a OP b EQ c ; -> use positions of consecutive runs
        start_of = {s: i for i, (s, e) in enumerate(rr)}
        rel_score = {}
        for op, eq in itertools.permutations(ops, 2):
            hits = {"+": 0, "*": 0}
            n = 0
            for i in range(len(rr) - 2):
                (s1, e1), (s2, e2), (s3, e3) = rr[i], rr[i + 1], rr[i + 2]
                if e2 < len(msg) and msg[e1] == op and s2 == e1 + 1 and msg[e2] == eq and s3 == e2 + 1:
                    n += 1
                    a, b_, c = vals[i], vals[i + 1], vals[i + 2]
                    hits["+"] += a + b_ == c
                    hits["*"] += a * b_ == c
            if n >= 20:
                for r, h in hits.items():
                    rel_score[(op, eq, r)] = (h / n, n)
        found = {}
        for r in ("+", "*"):
            cands = [(v[0], k) for k, v in rel_score.items() if k[2] == r]
            if cands:
                best = max(cands)
                found[r] = {"op": int(best[1][0]), "eq": int(best[1][1]), "truth_rate": float(best[0])}
        # comparison: "a OP b EQ R" where the single symbol R depends only on sign(a-b)
        eq_sym = found.get("+", {}).get("eq")
        cmp_found = None
        if eq_sym is not None:
            for op in ops:
                if op in (found.get("+", {}).get("op"), found.get("*", {}).get("op")):
                    continue
                trip = []
                for i in range(len(rr) - 1):
                    (s1, e1), (s2, e2) = rr[i], rr[i + 1]
                    if (e2 + 1 < len(msg) and msg[e1] == op and s2 == e1 + 1 and msg[e2] == eq_sym
                            and not is_digit[msg[e2 + 1]]):
                        trip.append((int(np.sign(vals[i] - vals[i + 1])), int(msg[e2 + 1])))
                if len(trip) < 20:
                    continue
                table = {}
                for sg, R in trip:
                    table.setdefault(sg, {}).setdefault(R, 0)
                    table[sg][R] += 1
                mapping = {sg: max(c, key=c.get) for sg, c in table.items()}
                cons = sum(table[sg][mapping[sg]] for sg in table) / len(trip)
                if cons > 0.95 and len(set(mapping.values())) == len(mapping) and                         (cmp_found is None or cons > cmp_found["consistency"]):
                    cmp_found = {"op": int(op), "map": {str(k): int(v) for k, v in mapping.items()},
                                 "consistency": float(cons)}
        cmp_correct = bool(cmp_found and cmp_found["op"] == int(perm[codec.o("?")]) and
                           cmp_found["map"].get("-1") == int(perm[codec.o("<")]) and
                           cmp_found["map"].get("1") == int(perm[codec.o(">")]) and
                           cmp_found["map"].get("0") == int(perm[codec.o("=")]))
        op_correct = {r: (found.get(r, {}).get("op") == int(perm[codec.o(r)])
                          and found.get(r, {}).get("eq") == int(perm[codec.o("=")])) for r in ("+", "*")}

        results[key] = dict(base=base, lsb=lsb, seed=seed, loss=loss, V=V,
                            role_acc=role_acc, n_digits_found=int(is_digit.sum()),
                            value_order_true=true_vals, value_exact=value_exact, value_spearman=rho,
                            succ_chain_values=succ_vals, succ_agreement=agree, succ_exact=succ_exact, succ_frac=succ_frac,
                            separator_correct=bool(sep == int(perm[codec.o(",")])),
                            lsb_from_counting=lsb_succ, lsb_from_geometry=lsb_geom,
                            order_fit=fits, lsb_hat=lsb_hat, lsb_correct=lsb_hat == lsb,
                            operators=found, operators_correct=op_correct,
                            comparison=cmp_found, comparison_correct=cmp_correct)
        print(key, json.dumps(results[key]), flush=True)
        json.dump(results, open(out_path, "w"), indent=1)
        if not os.path.exists(f"{ROOT}/results/numberline_{key}.npz") and seed == 0:
            rr, vals = read_numerals(true_msg, np.where(np.arange(V) < (base if base > 1 else 1),
                                                         np.arange(V), -1), lsb)
            np.savez_compressed(f"{ROOT}/results/numberline_{key}.npz",
                                pc=pc[:3000].astype(np.float32), value=vals[:3000])
