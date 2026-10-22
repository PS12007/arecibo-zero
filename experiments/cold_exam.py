"""How much does a receiver lean on the message it has just heard?  The main
exam asks every question after about 128 symbols of real message.  Here the
same held-out sums are asked after only k whole statements (k = 0 is a bare
question; "sep" is a bare question after a single end-of-statement mark).  Uses the 12 cached decipherment receivers, on the CPU."""
import json
import sys

import numpy as np
import torch

from common import ROOT, Design, build_message
sys.path.insert(0, ROOT)
from arecibo.receivers import ARCHS, split_pairs  # noqa: E402
from arecibo.decipher import relabel  # noqa: E402

MIX = {"add": 0.5, "count": 0.2, "cmp": 0.15, "mul": 0.15}     # as in decipher.py


class Relabelled(torch.nn.Module):
    """The receiver heard a relabelled message; ask it in its own labels."""
    def __init__(self, m, perm):
        super().__init__()
        self.m = m
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, x):
        return self.m(self.perm[x])[..., self.perm]


KS = [0, "sep", 1, 4, 16]     # "sep": bare question preceded by one ";" only
N_Q = 200
torch.set_num_threads(4)


@torch.no_grad()
def answer(model, seq, end, ctx, max_new):
    x = torch.tensor([seq])
    out = []
    for _ in range(max_new):
        nxt = int(model(x[:, -ctx:])[0, -1].argmax())
        if nxt == end:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return out


def main():
    D = json.load(open(f"{ROOT}/results/decipher.json"))
    res = {}
    for key, r in D.items():
        d = Design(base=r["base"], lsb_first=bool(r["lsb"]), mix=MIX)
        msg, codec, _ = build_message(d, 2**17, seed=r["seed"])
        _, perm = relabel(msg, codec.V, seed=77 + r["seed"])
        m = ARCHS["tf"](codec.V, 128)
        m.load_state_dict(torch.load(f"{ROOT}/runs/decipher_{key}.pt", map_location="cpu"))
        m = Relabelled(m, perm).eval()
        end = codec.o(";")
        bounds = np.flatnonzero(msg == end)
        rng = np.random.default_rng(1)
        _, exam_pairs = split_pairs(0)
        idx = rng.choice(len(exam_pairs), N_Q, replace=False)
        qs = [exam_pairs[i] if rng.random() < .5 else exam_pairs[i][::-1] for i in idx]
        starts = rng.integers(20, len(bounds) - 1, N_Q)
        res[key] = {}
        for k in KS:
            ok = 0
            for (a, b), j in zip(qs, starts):
                if k == "sep":
                    pre = [end]
                else:
                    pre = msg[bounds[j - k] + 1: bounds[j] + 1].tolist() if k else []
                full = codec.add(int(a), int(b))
                n = len(codec.num(int(a))) + 1 + len(codec.num(int(b))) + 1
                prompt, target = full[:n], full[n:]
                out = answer(m, pre + list(prompt), end, 128, len(target) + 3)
                ok += out == list(target)
            res[key][k] = ok / N_Q
        print(key, res[key], flush=True)
    json.dump(res, open(f"{ROOT}/results/cold_exam.json", "w"), indent=1)


if __name__ == "__main__":
    main()
