"""Raw-bit channel: the receiver sees only a stream of 0/1, with no symbol
boundaries, optional bit flips and optional repetition coding.

Every symbol of the design gets a random fixed-width codeword (the receiver
is never told the width).  Repetition r writes each bit r times.  Noise flips
each transmitted bit independently with probability p.  The exam asks clean
questions and reads the answer with the sender's codebook (majority vote over
repeats)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .message import Codec, split_pairs
from .receivers import device


class BitChannel:
    def __init__(self, codec: Codec, rep=1, noise=0.0, seed=0):
        self.codec = codec
        self.w = math.ceil(math.log2(codec.V))
        rng = np.random.default_rng(500 + seed)
        codes = rng.permutation(2 ** self.w)[: codec.V]
        self.table = np.array([[(c >> (self.w - 1 - i)) & 1 for i in range(self.w)] for c in codes],
                              dtype=np.int64)
        self.rep = rep
        self.noise = noise
        self.seed = seed

    def encode(self, toks, noisy=True):
        bits = self.table[np.asarray(toks)].reshape(-1)
        bits = np.repeat(bits, self.rep)
        if noisy and self.noise > 0:
            rng = np.random.default_rng(900 + self.seed)
            flip = rng.random(len(bits)) < self.noise
            bits = bits ^ flip
        return bits.astype(np.int64)

    def decode(self, bits):
        """Majority-vote repeats, then map codewords back (None if invalid)."""
        bits = np.asarray(bits)
        r, w = self.rep, self.w
        n = len(bits) // (r * w)
        if n == 0:
            return []
        b = bits[: n * r * w].reshape(n * w, r).sum(1) * 2 > r
        words = b.reshape(n, w)
        out = []
        for word in words:
            m = np.flatnonzero((self.table == word).all(1))
            out.append(int(m[0]) if len(m) else None)
        return out


@torch.no_grad()
def generate_fixed(model, rows, n_new, ctx, dev=None):
    dev = dev or device()
    x = torch.tensor(rows, device=dev)
    for _ in range(n_new):
        nxt = model(x[:, -ctx:])[:, -1].argmax(-1)
        x = torch.cat([x, nxt[:, None]], 1)
    return x[:, -n_new:].cpu().numpy()


def bit_exam(model, ch: BitChannel, clean_msg_tokens, ctx, split_seed=0, n_max=300, dev=None):
    """Held-out and seen addition questions asked over the channel (clean)."""
    codec = ch.codec
    rng = np.random.default_rng(1)
    train_pairs, exam_pairs = split_pairs(split_seed)
    end = codec.o(";")
    toks = np.asarray(clean_msg_tokens)
    ends = np.flatnonzero(toks == end)
    ends = ends[ends > 64]
    res = {}
    for key, pairs in (("add_heldout", exam_pairs), ("add_seen", train_pairs)):
        qs = [pairs[i] for i in rng.choice(len(pairs), min(n_max, len(pairs)), replace=False)]
        groups = {}
        for a, b in qs:
            if rng.random() < .5:
                a, b = b, a
            full = codec.add(a, b) + [end]
            k = len(codec.num(a)) + 1 + len(codec.num(b)) + 1
            groups.setdefault(len(full) - k, []).append((full[:k], full[k:]))
        correct = []
        for n_ans, items in groups.items():
            rows, targets = [], []
            for p, t in items:
                e = int(ends[rng.integers(len(ends))])
                pre = ch.encode(toks[max(0, e + 1 - 64): e + 1].tolist() + p, noisy=False)
                rows.append(pre.tolist())
                targets.append(t)
            L = min(len(r) for r in rows)
            rows = [r[-L:] for r in rows]
            n_new = n_ans * ch.w * ch.rep
            for i in range(0, len(rows), 200):
                out = generate_fixed(model, rows[i:i + 200], n_new, ctx, dev)
                for o, t in zip(out, targets[i:i + 200]):
                    correct.append(ch.decode(o) == t)
        res[key] = float(np.mean(correct))
    return res
