"""Q2: at a fixed bit budget, does spending part of the message on *other*
mathematics (counting, comparisons, products) help or hurt the receiver's
grasp of addition?"""
import os
import sys
from common import Design, run_grid, ROOT

quick = "--quick" in sys.argv
base = int(os.environ.get("BASE", "10"))
lsb = os.environ.get("LSB", "1") == "1"
mixes = {
    "add": {"add": 1.0},
    "add+count": {"add": 0.7, "count": 0.3},
    "add+cmp": {"add": 0.7, "cmp": 0.3},
    "add+mul": {"add": 0.7, "mul": 0.3},
    "all": {"add": 0.4, "count": 0.2, "cmp": 0.2, "mul": 0.2},
}
designs = [Design(base=base, lsb_first=lsb, mix=m, name=f"{k}") for k, m in mixes.items()]
budgets = [2**15] if quick else [2**14, 2**15, 2**16]
seeds = [0] if quick else [0, 1, 2]
jobs = [(d, B, "tf", s, "content") for B in budgets for d in designs for s in seeds]
if not quick:
    jobs += [(d, 2**15, "lstm", 0, "content") for d in designs]
run_grid(f"{ROOT}/results/content{'_quick' if quick else ''}.jsonl", jobs)
