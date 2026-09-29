"""E6-P v2 O_tape falsifier (docs/benchmark/E6P_V2_OTAPE_PROTOCOL.md): guardrail admissibility of the oracle.
Imports the scratchpad driver scratchpad/e6_dev/e6p_v2.py; skipped when the scratchpad (or its v2 state) is absent.
DEV seeds only."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
if not os.path.exists(os.path.join(HERE, "e6p_v2.py")) or not os.path.exists(os.path.join(HERE, "e6p_screen.py")):
    pytest.skip("scratchpad/e6_dev/e6p_v2.py not present", allow_module_level=True)
if not os.path.exists(os.path.join(HERE, "e6p_v2_state_json.py")):
    pytest.skip("e6p_v2 state absent (run e6p_v2.py init-state)", allow_module_level=True)
sys.path.insert(0, HERE)
import e6p_v2 as V  # noqa: E402


def test_guard_bound_rules():
    z = {k: 0.0 for k in V.GUARD_KEYS}
    assert V.guard_admissible(dict(z, rlf=1.0), z)[0]                     # AA 0 -> +1 slack
    assert V.guard_admissible(dict(z, rlf=2.0), z) == (False, ["rlf"])
    aa = dict(z, rlf=10.0, svr=100.0)
    assert V.guard_admissible(dict(aa, rlf=11.0, svr=110.0), aa)[0]      # exactly 1.10 x AA
    assert V.guard_admissible(dict(aa, rlf=11.01), aa) == (False, ["rlf"])
    assert V.guard_admissible(dict(aa, svr=110.5), aa) == (False, ["svr"])


def test_seed_mapping_and_scope():
    assert V.screen_seed(1, 0) == 155010 and V.screen_seed(3, 7) == 155037
    st = V.load_state()
    u1, u2 = V.units_for("1", st), V.units_for("2", st)
    assert {(k[1], k[2]) for u in u2 for k in u} == set(V.CELLS) and len(u2) == 24
    assert {k[2] for u in u1 for k in u} == {1, 3}
    assert all(155000 <= k[3] < 155200 for u in u1 + u2 for k in u)


def test_oracle_rejects_rlf_over_bound():
    res = V.check_guard_rejects()
    assert not res["rlf_bound_2.2"]["ok"] and res["off"]["ok"]


def test_constraint_disabled_reproduces_v1():
    res = V.check_disabled_matches_v1()
    assert res["n_deviations"] >= 1
