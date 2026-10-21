"""Prior-free receivers: randomly initialised sequence models that learn only
from the message (next-symbol prediction), then sit an exam."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .message import Codec, split_pairs, ADD_RANGE, MUL_RANGE


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, V, ctx, d=128, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x, return_h=False):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        h = self.ln(h)
        return (self.head(h), h) if return_h else self.head(h)


class LSTM(nn.Module):
    def __init__(self, V, ctx, d=256, layers=2):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.rnn = nn.LSTM(d, d, layers, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x, return_h=False):
        h, _ = self.rnn(self.tok(x))
        return (self.head(h), h) if return_h else self.head(h)


ARCHS = {
    "tf-s": lambda V, ctx: Transformer(V, ctx, d=64, layers=2, heads=4),
    "tf": lambda V, ctx: Transformer(V, ctx, d=128, layers=2, heads=4),
    "tf-deep": lambda V, ctx: Transformer(V, ctx, d=128, layers=4, heads=4),
    "lstm": lambda V, ctx: LSTM(V, ctx, d=256, layers=2),
    "lstm-s": lambda V, ctx: LSTM(V, ctx, d=128, layers=1),
}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_receiver(msg: np.ndarray, V: int, arch="tf", ctx=128, steps=3000, bs=64,
                   lr=1e-3, wd=0.1, seed=0, dev=None, log_every=0):
    dev = dev or device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ARCHS[arch](V, ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    data = torch.as_tensor(msg, device=dev)
    n = len(msg) - ctx - 1
    if n < 1:  # message shorter than one window: pad by repetition
        reps = math.ceil((ctx + 2) / len(msg))
        data = data.repeat(reps)
        n = len(data) - ctx - 1
    gen = torch.Generator(device=dev).manual_seed(seed)
    warm = 200
    t0 = time.time()
    for step in range(steps):
        lr_t = lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        ix = torch.randint(0, n, (bs,), device=dev, generator=gen)
        w = data[ix[:, None] + torch.arange(ctx + 1, device=dev)]
        logits = model(w[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    return model, loss.item()


# --------------------------------------------------------------------------
# Exam
# --------------------------------------------------------------------------

@torch.no_grad()
def complete(model, prompts: list[list[int]], context: np.ndarray, codec: Codec, ctx: int,
             max_new: int, dev=None):
    """Greedy-complete each prompt until ';' (or max_new). Each prompt is
    preceded by a real slice of the message ending at a statement boundary, so
    the receiver is asked questions 'in the language of the message'."""
    dev = dev or device()
    end = codec.o(";")
    B = len(prompts)
    rng = np.random.default_rng(0)
    ends = np.flatnonzero(context == end)
    ends = ends[ends > ctx]
    rows = []
    for p in prompts:
        e = int(ends[rng.integers(len(ends))]) if len(ends) else len(context) - 1
        pre = context[max(0, e + 1 - ctx * 2): e + 1].tolist()
        rows.append(pre + p)
    L = max(len(r) for r in rows)
    # left-pad with message prefix so all rows have equal length
    x = torch.tensor([context[: L - len(r)].tolist() + r if L > len(r) else r for r in rows],
                     device=dev)
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_new):
        logits = model(x[:, -ctx:])[:, -1]
        nxt = logits.argmax(-1)
        for i in range(B):
            if not done[i]:
                if nxt[i].item() == end:
                    done[i] = True
                else:
                    out[i].append(nxt[i].item())
        if done.all():
            break
        x = torch.cat([x, nxt[:, None]], 1)
    return out


def exam(model, codec: Codec, msg: np.ndarray, ctx=128, split_seed=0, n_max=400, dev=None,
         sections=("add_heldout", "add_seen", "mul", "cmp")):
    """Return accuracy on each exam section."""
    rng = np.random.default_rng(1)
    train_pairs, exam_pairs = split_pairs(split_seed)
    res = {}

    def run(qs, answer_fn, key):
        if len(qs) > n_max:
            qs = [qs[i] for i in rng.choice(len(qs), n_max, replace=False)]
        prompts, targets = [], []
        for q in qs:
            p, t = answer_fn(q)
            prompts.append(p)
            targets.append(t)
        max_new = max(len(t) for t in targets) + 4
        outs = []
        for i in range(0, len(prompts), 200):
            outs += complete(model, prompts[i:i + 200], msg, codec, ctx, max_new, dev)
        res[key] = float(np.mean([o == t for o, t in zip(outs, targets)]))

    def add_q(q):
        a, b = q
        full = codec.add(a, b)
        k = len(codec.num(a)) + 1 + len(codec.num(b)) + 1
        return full[:k], full[k:]

    def mul_q(q):
        a, b = q
        full = codec.mul(a, b)
        k = len(codec.num(a)) + 1 + len(codec.num(b)) + 1
        return full[:k], full[k:]

    def cmp_q(q):
        full = codec.cmp(*q)
        return full[:-1], full[-1:]

    if "add_heldout" in sections:
        qs = [(a, b) if rng.random() < .5 else (b, a) for a, b in exam_pairs]
        run(qs, add_q, "add_heldout")
    if "add_seen" in sections:
        qs = [(a, b) if rng.random() < .5 else (b, a) for a, b in train_pairs]
        run(qs, add_q, "add_seen")
    if "mul" in sections and "*" in [codec.vocab[t] for t in set(msg.tolist())]:
        qs = [(a, b) for a in range(MUL_RANGE) for b in range(MUL_RANGE)]
        run(qs, mul_q, "mul")
    if "cmp" in sections and "?" in [codec.vocab[t] for t in set(msg.tolist())]:
        qs = [(int(a), int(b)) for a, b in rng.integers(0, 64, (n_max, 2))]
        run(qs, cmp_q, "cmp")
    return res
