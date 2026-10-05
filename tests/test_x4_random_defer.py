"""X4 random accept / defer referee (scratchpad/e6_dev/x4_random_defer.py; EXTRAS_PROTOCOL.md Amendments 2026-10-05)."""
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
sys.path.insert(0, HERE)

import x4_random_defer as X4  # noqa: E402


def _obs(n):
    return {"t": 0.0, "requests": [{"knob": ("ptx", 0), "prop": 1, "cur": 0, "xapp": "a"}] * n}


def _decide(p, seed, steps=200, per=5):
    a = X4.RandomDefer(p, seed)
    return [d for _ in range(steps) for d in a(_obs(per))["decisions"]], a


def test_arms_and_p():
    assert X4.ARMS == ("X4:rand@0.352", "X4:rand@0.25", "X4:rand@0.5")
    assert abs(X4.P_PMRT - 37890 / 107525) < 5e-4


def test_deterministic_and_rate():
    d1, a = _decide(0.352, 191100)
    d2, _ = _decide(0.352, 191100)
    assert d1 == d2
    assert a.n["req"] == 1000 and a.n["rej"] == d1.count("reject")
    assert abs(a.n["rej"] / 1000 - 0.352) < 0.05
    assert _decide(0.352, 191101)[0] != d1


def test_common_random_numbers_nested():
    lo, _ = _decide(0.25, 191100)
    mid, _ = _decide(0.352, 191100)
    hi, _ = _decide(0.5, 191100)
    for a, b in ((lo, mid), (mid, hi)):
        assert all(y == "reject" for x, y in zip(a, b, strict=True) if x == "reject")


def test_edges():
    assert set(_decide(0.0, 1)[0]) == {"accept"}
    assert set(_decide(1.0, 1)[0]) == {"reject"}
    assert np.random.default_rng([1, X4.TAG]).random() == X4.RandomDefer(0.5, 1).rng.random()


def test_jobs_layout():
    J = X4.jobs()
    assert len(J) == 4 + 160 * 3 and len(set(J)) == len(J)
    assert J[:4] == [(191100, "noarb"), (191100, "never_sleep"), (191101, "noarb"), (191101, "never_sleep")]
    assert {s for s, _ in J} == set(range(191100, 191260))
