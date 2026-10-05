"""X7 subset-equivalence check (scratchpad/e6_dev/x7_subset_check.py; EXTRAS_PROTOCOL.md Amendments 2026-10-05)."""
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
sys.path.insert(0, HERE)

import x7_subset_check as X7  # noqa: E402

REQ = [{"knob": ("prot_min", 0), "prop": 0.3, "cur": 0.2, "xapp": "SliceGuarantee"},
       {"knob": ("ptx", 1), "prop": -3.0, "cur": 0.0, "xapp": "PowerES"},
       {"knob": ("sleep", 5), "prop": 1.0, "cur": 0.0, "xapp": "ES"}]


def _obs(t=200.0):
    return {"t": t, "requests": [dict(r) for r in REQ]}


def _all(dec):
    return lambda obs: {"decisions": [dec] * len(obs["requests"]), "writes": []}


def test_restrict():
    pm = X7.Restrict(_all("reject"), frozenset({"prot_min"}))
    assert pm(_obs())["decisions"] == ["reject", "accept", "accept"] and pm.n == {"overridden": 2, "kept": 1}
    other = X7.Restrict(_all("reject"), frozenset(X7.FAMS) - {"prot_min"})
    assert other(_obs())["decisions"] == ["accept", "reject", "reject"]


def test_shadow_tally_and_no_effect():
    sh = X7.Shadowed(_all("accept"), {"s": _all("reject")})
    assert sh(_obs())["decisions"] == ["accept"] * 3
    sh(_obs(10.0))
    assert sh.agree["s"]["prot_min|+1|post"] == [0, 1, 0, 0]
    assert sh.agree["s"]["ptx|-1|pre"] == [0, 1, 0, 0]


def test_jobs_layout():
    J = X7.jobs()
    assert len(J) == 160 * 13 and len({(s, X7.arm_name(r, k)) for s, k, r in J}) == len(J)
    assert J[0] == (191100, "shadow", None)
