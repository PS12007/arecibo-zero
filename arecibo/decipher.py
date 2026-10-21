"""Blind decipherment: recover an unknown message design by reading the mind
of a receiver trained on it.

We get a message whose symbols have been randomly relabelled.  We train a
receiver on it (no knowledge needed), then, using only the message and the
receiver's internals:

1. Role discovery.  Seeds: symbols the receiver expects can repeat
   immediately ("11"); operators almost never do.  Expansion: any symbol the
   receiver expects in the same places as the seeds (correlated
   predicted-probability profiles over the message) is also a digit.
2. Value recovery.  The receiver's hidden state at the end of a numeral run
   should encode the number's magnitude.  The first principal component of
   those states, over all numerals in the message, gives an ordering; we read
   each digit's value from where single-digit numerals land on that axis.
3. Operator recovery.  Among operator symbols, '=' is the one after which the
   receiver is most certain (lowest predictive entropy over the following
   run), the statement end ';' is the most frequent, and so on.

All steps are scored against the hidden ground truth.
"""
from __future__ import annotations

import numpy as np
import torch

from .receivers import device


def relabel(msg: np.ndarray, V: int, seed: int):
    perm = np.random.default_rng(seed).permutation(V)  # true id -> shown id
    return perm[msg], perm


@torch.no_grad()
def hidden_states(model, msg: np.ndarray, ctx: int, dev=None, stride=None):
    """Hidden state and next-symbol distribution at every position (sliding
    windows, keep the second half of each window for full left context)."""
    dev = dev or device()
    stride = stride or ctx // 2
    n = len(msg)
    H = None
    P = None
    data = torch.as_tensor(msg, device=dev)
    starts = list(range(0, max(1, n - ctx), stride))
    for s0 in range(0, len(starts), 64):
        batch = starts[s0:s0 + 64]
        x = torch.stack([data[s:s + ctx] for s in batch])
        logits, h = model(x, return_h=True)
        p = logits.softmax(-1)
        if H is None:
            H = torch.zeros(n, h.shape[-1], device=dev)
            P = torch.zeros(n, p.shape[-1], device=dev)
        for i, s in enumerate(batch):
            lo = 0 if s == 0 else ctx - stride
            H[s + lo:s + ctx] = h[i, lo:]
            P[s + lo:s + ctx] = p[i, lo:]
    return H.cpu().numpy(), P.cpu().numpy()


def discover_roles(msg, P, V, corr_thresh=0.5):
    """Split symbols into 'digit-like' and 'operator-like' without labels.
    Digit symbols are the largest group whose members are typically followed
    by each other; we use the mean predicted mass that the next symbol is in
    the same group, and iterate a 2-means style assignment."""
    # mean predictive distribution after each symbol
    M = np.zeros((V, V))
    for v in range(V):
        idx = np.flatnonzero(msg[:-1] == v)
        idx = idx[idx < len(P)]
        if len(idx):
            M[v] = P[idx].mean(0)
    # Cue 1 (seeds): digits can follow themselves ("11"); operators almost
    # never do.  Seeds = symbols whose expected repeat probability is at least
    # a tenth of the highest one.
    freq = np.bincount(msg, minlength=V) / len(msg)
    present = freq > 0
    rep = np.diag(M)
    seeds = (rep >= 0.1 * rep.max()) & present
    # Cue 2 (expansion): digits are the symbols the receiver expects in the
    # same places.  Correlate each symbol's predicted-probability profile over
    # the message with the seeds' mean profile.
    prof = P[:, :V]
    target = prof[:, seeds].mean(1)
    corr = np.array([np.corrcoef(prof[:, v], target)[0, 1] if present[v] and prof[:, v].std() > 0
                     else -1 for v in range(V)])
    group = (seeds | (corr > corr_thresh)) & present
    return group, M, freq


def numeral_runs(msg, is_digit):
    runs = []
    i = 0
    n = len(msg)
    while i < n:
        if is_digit[msg[i]]:
            j = i
            while j < n and is_digit[msg[j]]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def recover_values(msg, H, is_digit, max_runs=6000):
    """Order digit symbols by where single-digit numerals fall on the first
    principal component of end-of-numeral hidden states."""
    runs = numeral_runs(msg, is_digit)
    runs = [r for r in runs if r[1] - 1 < len(H)][:max_runs]
    X = np.stack([H[e - 1] for _, e in runs])
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc = X @ vt[0]
    lengths = np.array([e - s for s, e in runs])
    # orient the axis: longer numerals are bigger
    if np.corrcoef(pc, lengths)[0, 1] < 0:
        pc = -pc
    digits = np.flatnonzero(is_digit)
    score = {}
    for d in digits:
        sel = [k for k, (s, e) in enumerate(runs) if e - s == 1 and msg[s] == d]
        if not sel:  # fall back: runs whose first symbol is d
            sel = [k for k, (s, e) in enumerate(runs) if msg[s] == d]
        score[int(d)] = float(np.mean(pc[sel])) if sel else 0.0
    order = sorted(score, key=score.get)
    return order, pc, runs


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def find_separator(msg, is_digit):
    """The counting separator: the operator that most often sits between two
    numerals that are themselves followed by the same operator (a, b, c ...)."""
    runs = numeral_runs(msg, is_digit)
    best, cnt = None, {}
    for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
        if s2 == e1 + 1 and e2 < len(msg) and msg[e1] == msg[e2]:
            cnt[int(msg[e1])] = cnt.get(int(msg[e1]), 0) + 1
    if cnt:
        best = max(cnt, key=cnt.get)
    return best, runs


@torch.no_grad()
def successor_probe(model, msg, is_digit, ctx, dev=None, n_ctx=8, max_new=6):
    """Ask the receiver to count: splice "x SEP" into places where the message
    counts and find the 1- or 2-digit numeral (followed by SEP) it considers
    most likely next, summing log-probabilities over several contexts.  Returns the separator,
    succ: digit -> continuation (tuple of symbols, majority over contexts)."""
    dev = dev or device()
    sep, runs = find_separator(msg, is_digit)
    # starts of counting runs: numeral followed by sep and another numeral
    starts = [s for (s, e), (s2, e2) in zip(runs, runs[1:])
              if e < len(msg) and msg[e] == sep and s2 == e + 1 and s > ctx]
    rng = np.random.default_rng(0)
    starts = [starts[i] for i in rng.choice(len(starts), min(n_ctx, len(starts)), replace=False)]
    digits = [int(x) for x in np.flatnonzero(is_digit)]
    cands = [(y,) for y in digits] + [(y, z) for y in digits for z in digits]
    succ = {}
    for x in digits:
        score = np.zeros(len(cands))
        for s in starts:
            pre = msg[s - ctx + 8:s].tolist() + [x, sep]
            seqs = [(pre + list(c) + [sep])[-ctx:] for c in cands]
            L = max(len(q) for q in seqs)
            seqs = [q[:1] * (L - len(q)) + q for q in seqs]   # left-pad (positions shift; fine)
            t = torch.tensor(seqs, device=dev)
            lp = []
            for i in range(0, len(t), 128):
                lp.append(model(t[i:i + 128]).log_softmax(-1))
            lp = torch.cat(lp)
            for ci, c in enumerate(cands):
                n = len(c) + 1
                tgt = t[ci, -n:]
                score[ci] += lp[ci, -n - 1:-1].gather(1, tgt[:, None]).sum().item()
        succ[x] = cands[int(np.argmax(score))]
    return sep, succ


def values_from_successor(succ):
    """Turn the receiver's counting answers into digit values and digit order.
    Following the first symbol of each answer gives an ordering of the digits;
    we try every digit as zero and both digit orders, and keep the assignment
    under which the most answers are correct successors.  (Robust to a few
    wrong answers.)  Returns (chain of symbols in value order, lsb, agreement)."""
    digits = list(succ)
    b = len(digits)
    first = {x: s[0] for x, s in succ.items() if len(s)}
    best = (None, None, -1)
    for z in digits:
        chain = [z]
        while len(chain) < b and first.get(chain[-1]) not in (None, *chain):
            chain.append(first[chain[-1]])
        if len(chain) < b:
            continue
        val = {sym: v for v, sym in enumerate(chain)}
        for lsb in (0, 1):
            def enc(n):
                ds = []
                while True:
                    ds.append(chain[n % b])
                    n //= b
                    if n == 0:
                        break
                return tuple(ds if lsb else ds[::-1])
            agree = sum(succ[x] == enc(val[x] + 1) for x in digits)
            if agree > best[2]:
                best = (chain, lsb, agree)
    return best
