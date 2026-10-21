"""Export one trained receiver for in-browser inference (docs/receiver.json).

Uses the decipherment receiver for base 10, LSB first, mixed content (sums,
products, comparisons, counting), trained on the relabelled message."""
import base64
import json
import sys

import numpy as np
import torch

from common import ROOT, Design, build_message
sys.path.insert(0, ROOT)
from arecibo.receivers import ARCHS  # noqa: E402
from arecibo.decipher import relabel  # noqa: E402
from arecibo.message import split_pairs  # noqa: E402

MIX = {"add": 0.5, "count": 0.2, "cmp": 0.15, "mul": 0.15}
d = Design(base=10, lsb_first=True, mix=MIX)
true_msg, codec, _ = build_message(d, 2**17, seed=0)
msg, perm = relabel(true_msg, codec.V, seed=77)
inv = np.argsort(perm)
model = ARCHS["tf"](codec.V, 128)
model.load_state_dict(torch.load(f"{ROOT}/runs/decipher_b10-lsb-s0.pt", map_location="cpu"))
model.eval()

sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
weights = {k: {"shape": list(v.shape), "b64": base64.b64encode(v.astype("<f4").tobytes()).decode()}
           for k, v in sd.items()}

# a real stretch of the transmission ending at a statement boundary, in shown ids
end = int(perm[codec.o(";")])
ends = np.flatnonzero(msg == end)
e = int(ends[len(ends) // 2])
context = msg[e - 90:e + 1].tolist()
context = context[context.index(end) + 1:]          # start at a statement start

# reference outputs for checking the JS implementation
probe = torch.tensor([context[-40:]])
with torch.no_grad():
    ref = model(probe)[0, -1].numpy().tolist()

out = {"V": codec.V, "d": 128, "layers": 2, "heads": 4, "ctx": 128,
       "vocab": [codec.vocab[int(inv[i])] for i in range(codec.V)],   # shown id -> true meaning
       "weights": weights, "context": context, "probe": context[-40:], "probe_logits": ref,
       "heldout_add_pairs": [list(p) for p in split_pairs(0)[1]]}
json.dump(out, open(f"{ROOT}/docs/receiver.json", "w"))
print("params", sum(v.size for v in sd.values()), "keys", list(sd)[:6])
