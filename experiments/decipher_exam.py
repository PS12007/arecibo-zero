"""Exam the cached decipherment receivers (does decipherability track understanding?).
The receivers were trained on relabelled messages, so questions are mapped
through the same permutation."""
import json
import sys

import numpy as np
import torch

from common import ROOT, Design, build_message, device
sys.path.insert(0, ROOT)
from arecibo.receivers import ARCHS, exam  # noqa: E402
from arecibo.decipher import relabel  # noqa: E402

MIX = {"add": 0.5, "count": 0.2, "cmp": 0.15, "mul": 0.15}
dev = device()


class Relabelled(torch.nn.Module):
    def __init__(self, m, perm):
        super().__init__()
        self.m = m
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, x):
        return self.m(self.perm[x])[..., self.perm]


path = f"{ROOT}/results/decipher.json"
D = json.load(open(path))
for key, r in D.items():
    d = Design(base=r["base"], lsb_first=bool(r["lsb"]), mix=MIX)
    true_msg, codec, _ = build_message(d, 2**17, seed=r["seed"])
    _, perm = relabel(true_msg, codec.V, seed=77 + r["seed"])
    m = ARCHS["tf"](codec.V, 128)
    m.load_state_dict(torch.load(f"{ROOT}/runs/decipher_{key}.pt", map_location="cpu"))
    m = Relabelled(m, perm).to(dev).eval()
    r["exam"] = exam(m, codec, true_msg, ctx=128, dev=dev)
    print(key, r["exam"], flush=True)
json.dump(D, open(path, "w"), indent=1)
