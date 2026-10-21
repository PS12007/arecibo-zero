"""K independent transformer receivers trained in lock-step as one batched
model (stacked weights, batched matmuls).  Each member has its own message,
initialisation seed, data stream and gradient clipping; Adam is elementwise,
so every member follows exactly the update it would follow alone.  This only
removes kernel-launch overhead, which dominates for models this small."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .receivers import device


def _lin(K, fan_in, fan_out, gens):
    bound = 1 / math.sqrt(fan_in)
    W = torch.stack([(torch.rand(fan_in, fan_out, generator=g) * 2 - 1) * bound for g in gens])
    b = torch.stack([(torch.rand(fan_out, generator=g) * 2 - 1) * bound for g in gens])
    return nn.Parameter(W), nn.Parameter(b)


class EnsembleTransformer(nn.Module):
    def __init__(self, K, V, ctx, seeds, d=128, layers=2, heads=4):
        super().__init__()
        gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
        self.K, self.V, self.d, self.h, self.L = K, V, d, heads, layers
        self.tok = nn.Parameter(torch.stack([torch.randn(V, d, generator=g) for g in gens]))
        self.pos = nn.Parameter(torch.stack([torch.randn(ctx, d, generator=g) for g in gens]))
        self.p = nn.ParameterDict()
        for l in range(layers):
            for n in ("ln1", "ln2"):
                self.p[f"{l}_{n}_w"] = nn.Parameter(torch.ones(K, d))
                self.p[f"{l}_{n}_b"] = nn.Parameter(torch.zeros(K, d))
            self.p[f"{l}_qkv_W"], self.p[f"{l}_qkv_b"] = _lin(K, d, 3 * d, gens)
            self.p[f"{l}_proj_W"], self.p[f"{l}_proj_b"] = _lin(K, d, d, gens)
            self.p[f"{l}_fc1_W"], self.p[f"{l}_fc1_b"] = _lin(K, d, 4 * d, gens)
            self.p[f"{l}_fc2_W"], self.p[f"{l}_fc2_b"] = _lin(K, 4 * d, d, gens)
        self.p["lnf_w"] = nn.Parameter(torch.ones(K, d))
        self.p["lnf_b"] = nn.Parameter(torch.zeros(K, d))
        self.p["head_W"], self.p["head_b"] = _lin(K, d, V, gens)

    def _ln(self, x, name, ks):
        x = F.layer_norm(x, (self.d,))
        return x * self.p[name + "_w"][ks][:, None, None] + self.p[name + "_b"][ks][:, None, None]

    def _lin(self, x, name, ks):
        k, B, T, _ = x.shape
        y = torch.bmm(x.reshape(k, B * T, -1), self.p[name + "_W"][ks])
        return y.view(k, B, T, -1) + self.p[name + "_b"][ks][:, None, None]

    def forward(self, x, ks=None, return_h=False):
        """x: (k, B, T) token ids for members ks (default all)."""
        ks = torch.arange(self.K, device=x.device) if ks is None else ks
        k, B, T = x.shape
        h = self.tok[ks[:, None, None], x] + self.pos[ks][:, None, :T]
        dh = self.d // self.h
        for l in range(self.L):
            q, kk, v = self._lin(self._ln(h, f"{l}_ln1", ks), f"{l}_qkv", ks).split(self.d, -1)
            sh = lambda t: t.reshape(k * B, T, self.h, dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(sh(q), sh(kk), sh(v), is_causal=True)
            a = a.transpose(1, 2).reshape(k, B, T, self.d)
            h = h + self._lin(a, f"{l}_proj", ks)
            m = F.gelu(self._lin(self._ln(h, f"{l}_ln2", ks), f"{l}_fc1", ks))
            h = h + self._lin(m, f"{l}_fc2", ks)
        h = self._ln(h, "lnf", ks)
        logits = self._lin(h, "head", ks)
        return (logits, h) if return_h else logits


class Member(nn.Module):
    """Single-receiver view with the (B, T) interface used by the exams."""

    def __init__(self, ens, k):
        super().__init__()
        self.ens, self.k = ens, k

    def forward(self, x, return_h=False):
        ks = torch.tensor([self.k], device=x.device)
        out = self.ens(x[None], ks=ks, return_h=return_h)
        return (out[0][0], out[1][0]) if return_h else out[0]


def train_ensemble(msgs, V, seeds, ctx=128, steps=8000, bs=64, lr=1e-3, wd=0.1, dev=None,
                   d=128, layers=2, log_every=0, amp=False):
    """msgs: list of K token arrays (one message per member)."""
    dev = dev or device()
    torch.backends.cuda.matmul.allow_tf32 = True
    K = len(msgs)
    ens = EnsembleTransformer(K, V, ctx, seeds, d=d, layers=layers).to(dev)
    opt = torch.optim.AdamW(ens.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    datas, ns = [], []
    for m in msgs:
        t = torch.as_tensor(m, device=dev)
        if len(t) < ctx + 2:
            t = t.repeat(math.ceil((ctx + 2) / len(t)))
        datas.append(t)
        ns.append(len(t) - ctx - 1)
    # one flat buffer; per-member offsets
    offs = torch.tensor(np.cumsum([0] + [len(t) for t in datas[:-1]]), device=dev)
    flat = torch.cat(datas)
    nsT = torch.tensor(ns, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(int(seeds[0]) * 7919 + K)
    ar = torch.arange(ctx + 1, device=dev)
    warm = 200
    for step in range(steps):
        lr_t = lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        ix = (torch.rand(K, bs, device=dev, generator=gen) * nsT[:, None]).long() + offs[:, None]
        w = flat[ix[..., None] + ar]                      # (K, bs, ctx+1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            logits = ens(w[..., :-1])
        loss_k = F.cross_entropy(logits.float().reshape(-1, V), w[..., 1:].reshape(-1),
                                 reduction="none").view(K, -1).mean(1)
        opt.zero_grad(set_to_none=True)
        loss_k.sum().backward()
        # per-member gradient clipping at norm 1
        with torch.no_grad():
            sq = sum(p.grad.reshape(K, -1).pow(2).sum(1) for p in ens.parameters())
            scale = (1.0 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in ens.parameters():
                p.grad.mul_(scale.view(K, *[1] * (p.grad.dim() - 1)))
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {loss_k.mean().item():.3f}", flush=True)
    ens.eval()
    return ens, loss_k.detach().cpu().numpy()
