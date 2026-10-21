"""Q1: which way of writing numbers does a prior-free receiver understand
with the fewest transmitted bits?  Addition-only messages."""
import sys
from common import Design, run_grid, ROOT

quick = "--quick" in sys.argv
bases = [1, 2, 3, 4, 6, 10, 16]
budgets = [2**14, 2**16] if quick else [2**13, 2**14, 2**15, 2**16, 2**17]
archs = ["tf"] if quick else ["tf", "lstm"]
seeds = [0] if quick else [0, 1]

designs = []
for b in bases:
    designs.append(Design(base=b))
    if b > 1:
        designs.append(Design(base=b, lsb_first=True))

jobs = [(d, B, "tf", s, "ens") for B in budgets for d in designs for s in seeds]
if not quick:  # LSTMs are slow: two budgets, one seed
    jobs += [(d, B, "lstm", 0) for B in (2**14, 2**16) for d in designs]
run_grid(f"{ROOT}/results/radix{'_quick' if quick else ''}.jsonl", jobs)
