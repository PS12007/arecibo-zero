"""Message generator: the sender's side of first contact.

A *design* fixes how numbers are written (radix, digit order) and what the
message talks about (a mix of counting runs, sums, products, comparisons).
The *concepts* are fixed across designs so that every design teaches the same
mathematics and can be compared at an equal bit budget.

Exam pairs (unordered operand pairs held out from the message) never appear
in any addition statement, in either order, so a receiver can only answer
them by having extracted the rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

OPS = ["+", "*", "=", "?", "<", ">", ",", ";"]

ADD_RANGE = 64          # addition operands in [0, 64)
MUL_RANGE = 16          # multiplication operands in [0, 16)
CMP_RANGE = 64
COUNT_RANGE = 128
COUNT_RUN = 6
EXAM_FRAC = 0.25        # fraction of unordered add pairs held out


@dataclass
class Design:
    base: int = 10                  # 1 = unary
    lsb_first: bool = False         # write least significant digit first
    mix: dict = field(default_factory=lambda: {"add": 1.0})
    delimiters: bool = True         # ';' between statements
    name: str = ""

    def label(self):
        if self.name:
            return self.name
        r = "unary" if self.base == 1 else f"b{self.base}{'-lsb' if self.lsb_first else ''}"
        m = "+".join(f"{k}{v:g}" for k, v in sorted(self.mix.items()))
        return f"{r}|{m}"

    def to_dict(self):
        return asdict(self)


class Codec:
    """Maps statements to token ids for one design."""

    def __init__(self, design: Design):
        self.d = design
        nd = 1 if design.base == 1 else design.base
        self.digits = [f"d{i}" for i in range(nd)]
        self.vocab = self.digits + OPS
        self.idx = {s: i for i, s in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.bits_per_token = math.log2(self.V)

    def num(self, n: int) -> list[int]:
        b = self.d.base
        if b == 1:
            return [0] * n          # unary: zero is the empty string
        if n == 0:
            ds = [0]
        else:
            ds = []
            while n:
                ds.append(n % b)
                n //= b
            ds = ds[::-1]           # MSB first
        if self.d.lsb_first:
            ds = ds[::-1]
        return ds                   # digit symbol i has id i

    def decode_num(self, toks: list[int]) -> int | None:
        b = self.d.base
        if any(t >= len(self.digits) for t in toks):
            return None
        if b == 1:
            return len(toks)
        if not toks:
            return None
        ds = toks[::-1] if self.d.lsb_first else toks
        v = 0
        for t in ds:
            v = v * b + t
        return v

    def o(self, s):
        return self.idx[s]

    # statement encoders ------------------------------------------------
    def add(self, a, b):
        return self.num(a) + [self.o("+")] + self.num(b) + [self.o("=")] + self.num(a + b)

    def mul(self, a, b):
        return self.num(a) + [self.o("*")] + self.num(b) + [self.o("=")] + self.num(a * b)

    def cmp(self, a, b):
        r = "<" if a < b else (">" if a > b else "=")
        return self.num(a) + [self.o("?")] + self.num(b) + [self.o("=")] + [self.o(r)]

    def count(self, start, k=COUNT_RUN):
        out = []
        for i in range(k):
            if i:
                out.append(self.o(","))
            out += self.num(start + i)
        return out

    def end(self):
        return [self.o(";")] if self.d.delimiters else []


def split_pairs(seed=0):
    """Unordered addition pairs -> (train set, exam set)."""
    rng = np.random.default_rng(10_000 + seed)
    pairs = [(a, b) for a in range(ADD_RANGE) for b in range(a, ADD_RANGE)]
    rng.shuffle(pairs)
    k = int(len(pairs) * EXAM_FRAC)
    exam = set(pairs[:k])
    train = [p for p in pairs[k:]]
    return train, sorted(exam)


def build_message(design: Design, bit_budget: int, seed=0, split_seed=0):
    """Return (token array, codec, info). Statements are sampled i.i.d. from
    the design's mix until the bit budget is used up."""
    codec = Codec(design)
    rng = np.random.default_rng(seed)
    train_pairs, _ = split_pairs(split_seed)
    kinds = list(design.mix.keys())
    w = np.array([design.mix[k] for k in kinds], float)
    w = w / w.sum()
    max_tokens = int(bit_budget / codec.bits_per_token)
    toks: list[int] = []
    counts = {k: 0 for k in kinds}
    while True:
        k = kinds[rng.choice(len(kinds), p=w)]
        if k == "add":
            a, b = train_pairs[rng.integers(len(train_pairs))]
            if rng.random() < 0.5:
                a, b = b, a
            s = codec.add(a, b)
        elif k == "mul":
            s = codec.mul(int(rng.integers(MUL_RANGE)), int(rng.integers(MUL_RANGE)))
        elif k == "cmp":
            s = codec.cmp(int(rng.integers(CMP_RANGE)), int(rng.integers(CMP_RANGE)))
        elif k == "count":
            s = codec.count(int(rng.integers(COUNT_RANGE - COUNT_RUN)))
        else:
            raise ValueError(k)
        s = s + codec.end()
        if len(toks) + len(s) > max_tokens:
            break
        toks += s
        counts[k] += 1
    info = {"tokens": len(toks), "bits": len(toks) * codec.bits_per_token,
            "V": codec.V, "statements": counts}
    return np.array(toks, dtype=np.int64), codec, info
