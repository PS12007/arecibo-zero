import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arecibo.message import Design, Codec, build_message, split_pairs  # noqa: E402
from arecibo.channel import BitChannel  # noqa: E402


@pytest.mark.parametrize("base", [1, 2, 3, 10, 16])
@pytest.mark.parametrize("lsb", [False, True])
def test_number_roundtrip(base, lsb):
    c = Codec(Design(base=base, lsb_first=lsb))
    for n in range(0, 300, 7):
        assert c.decode_num(c.num(n)) == n


def test_exam_pairs_never_transmitted():
    _, exam = split_pairs(0)
    exam = set(exam)
    d = Design(base=10, delimiters=True)
    msg, c, _ = build_message(d, 2**15)
    toks = msg.tolist()
    end, plus, eq = c.o(";"), c.o("+"), c.o("=")
    stmt, seen = [], 0
    for t in toks:
        if t == end:
            i, j = stmt.index(plus), stmt.index(eq)
            a, b = c.decode_num(stmt[:i]), c.decode_num(stmt[i + 1:j])
            assert (min(a, b), max(a, b)) not in exam
            assert c.decode_num(stmt[j + 1:]) == a + b
            seen += 1
            stmt = []
        else:
            stmt.append(t)
    assert seen > 100


def test_budget_respected():
    for base in (1, 2, 10):
        msg, c, info = build_message(Design(base=base, mix={"add": 1, "count": 1}), 2**14)
        assert info["bits"] <= 2**14


@pytest.mark.parametrize("rep", [1, 3])
def test_channel_roundtrip(rep):
    c = Codec(Design(base=10))
    ch = BitChannel(c, rep=rep, noise=0.0)
    toks = c.add(12, 34) + [c.o(";")]
    assert ch.decode(ch.encode(toks)) == toks


def test_repetition_corrects_single_flips():
    c = Codec(Design(base=10))
    ch = BitChannel(c, rep=3)
    toks = c.add(5, 7)
    bits = ch.encode(toks)
    bits[::3] ^= 1          # one flip in every triple
    assert ch.decode(bits) == toks


def test_sign_test_exact():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from stats import sign_test
    assert sign_test(5, 10) == 1.0
    assert abs(sign_test(10, 10) - 2 / 2**10) < 1e-12
    assert abs(sign_test(0, 10) - 2 / 2**10) < 1e-12


def test_unary_only_in_lstm_u_search():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import numpy as np
    import evolve
    rng = np.random.default_rng(0)
    plain = {evolve.random_genome(rng)["base"] for _ in range(300)}
    wide = {evolve.random_genome(rng, evolve.BASES_FOR["lstm-u"])["base"] for _ in range(300)}
    assert 1 not in plain and 1 in wide
    g = {"base": 1, "lsb": 0, "w": [1, 0, 0, 0]}
    for _ in range(50):     # mutation never leaves the allowed set
        g = evolve.mutate(g, rng, evolve.BASES_FOR["lstm-u"])
        assert g["base"] in evolve.BASES_FOR["lstm-u"]
