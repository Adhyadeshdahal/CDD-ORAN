"""cdd_oran.decision.units_p: unit semantics, slew-rule half, rollback flag, obs-only exposure sets, lock == reject."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from cdd_oran.decision import units_p as UP
from cdd_oran.decision.plans import half_step
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

STATIC = {"cells": 3, "neighbours": [[1], [0], [1]], "is_macro": np.array([True, True, False]),
          "knobs": [("carrier", 0), ("carrier", 1), ("sleep", 2), ("prot_min", 0), ("prot_min", 1), ("ptx", 0)],
          "xapps": {"ES": [], "SliceGuarantee": [], "PowerES": []}}


def req(x, k, cur, prop):
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": 0.0}


def obs(t, reqs, config):
    return {"t": float(t), "requests": reqs, "config": dict(config), "new_reports": [], "static": STATIC,
            "locked": {}, "changes": 0, "churn_cap": None}


class Const:
    def __init__(self, mode):
        self.mode, self.calls = mode, []

    def __call__(self, unit):
        self.calls.append((unit["c"], unit["x"], unit["t0"]))
        return self.mode, 0.5


def cfg_p(pair, seed=5, warm=60.0, scored=90.0, lf=0.79):
    px = {"P1": ("SliceGuarantee",), "P3": ("PowerES", "SliceGuarantee")}[pair]
    return C.E6Config(seed=seed, mix="ES", scenario="base", load_factor=lf, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=warm, scored_s=scored, e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=px))


# ------------------------------------------------------------------------------------------------ unit semantics
def test_unit_opens_at_first_request_and_mode_holds_for_T():
    pol = Const("reject")
    arb = UP.UnitArbiter(pol, T=60.0)
    cfg = {("carrier", 0): 2.0, ("prot_min", 0): 0.0}
    d = arb(obs(100, [req("ES", ("carrier", 0), 2.0, 1.0), req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)], cfg))
    assert d["decisions"] == ["reject", "reject"] and len(pol.calls) == 2      # two units: (0, ES), (0, SG)
    for t in (110, 159):
        assert arb(obs(t, [req("ES", ("carrier", 0), 2.0, 1.0)], cfg))["decisions"] == ["reject"]
    assert len(pol.calls) == 2                                                 # still the same unit
    arb(obs(160, [req("ES", ("carrier", 0), 2.0, 1.0)], cfg))
    assert pol.calls[-1] == (0, "ES", 160.0) and len(pol.calls) == 3           # a new unit after T
    u = arb.units[0]
    assert (u["c"], u["x"], u["t0"], u["mode"], u["p"], u["n_req"]) == (0, "ES", 100.0, "reject", 0.5, 3)
    assert u["x_idx"] == UP.XAPPS_P.index("ES") and u["exp"] == [0, 1]


def test_no_units_during_warmup_everything_accepted():
    pol = Const("reject")
    arb = UP.UnitArbiter(pol, warmup_s=120.0)
    d = arb(obs(119, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0}))
    assert d["decisions"] == ["accept"] and not pol.calls and not arb.units
    assert arb(obs(120, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0}))["decisions"] == ["reject"]


def test_half_uses_the_slew_rule_with_e6p_quanta():
    arb = UP.UnitArbiter(Const("half"))
    cfg = {("ptx", 0): 0.0, ("prot_min", 0): 0.0, ("carrier", 0): 2.0}
    d = arb(obs(10, [req("PowerES", ("ptx", 0), 0.0, -3.0)], cfg))
    assert d["decisions"] == [("modify", -2.0)]                               # 3 quanta -> ceil(1.5) = 2 dB
    d1 = arb(obs(11, [req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)], cfg))["decisions"]
    d2 = arb(obs(16, [req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)], cfg))["decisions"]
    assert sorted([d1[0], d2[0]]) == ["accept", "reject"]                     # single quantum -> alternate


def test_half_step_p_matches_the_screen_and_plans_rules():
    hs1, hs2 = {}, {}
    for k, cur, prop in [(("carrier", 3), 2.0, 1.0), (("sleep", 21), 0.0, 1.0), (("carrier", 4), 1.0, 2.0)]:
        for _ in range(3):
            assert UP.half_step_p(k, cur, prop, hs1) == half_step(k, cur, prop, hs2)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
    if not os.path.exists(os.path.join(here, "e6p_screen.py")):
        pytest.skip("scratchpad/e6_dev/e6p_screen.py not present")
    sys.path.insert(0, here)
    try:
        import e6p_screen as S
    finally:
        sys.path.remove(here)
    a, b = {}, {}
    for k, cur, prop in [(("ptx", 0), 0.0, -3.0), (("ptx", 5), -6.0, -3.0), (("prot_min", 2), 0.1, 0.15),
                         (("prot_min", 3), 0.2, 0.1), (("carrier", 3), 2.0, 1.0), (("sleep", 22), 1.0, 0.0)]:
        for _ in range(3):
            assert UP.half_step_p(k, cur, prop, a) == S.half_step(k, cur, prop, b)


def test_rollback_flag_rolls_back_a_recent_change_once_and_never_our_own():
    arb = UP.UnitArbiter(Const("accept+rb"), T=10.0)
    k = ("prot_min", 0)
    arb(obs(99, [], {k: 0.0}))
    arb(obs(100, [], {k: 0.05}))                          # a change applied at t = 99 (seen at 100)
    d = arb(obs(120, [req("SliceGuarantee", k, 0.05, 0.10)], {k: 0.05}))
    assert d["rollback"] == [k] and d["decisions"] == ["accept"]
    arb(obs(121, [], {k: 0.0}))                           # our rollback took effect at 120
    d = arb(obs(131, [req("SliceGuarantee", k, 0.0, 0.05)], {k: 0.0}))      # new unit, same rb flag
    assert d["rollback"] == []                            # the latest change is our own rollback
    arb2 = UP.UnitArbiter(Const("accept+rb"))
    arb2(obs(1, [], {k: 0.0}))
    arb2(obs(2, [], {k: 0.05}))
    assert arb2(obs(200, [req("SliceGuarantee", k, 0.05, 0.1)], {k: 0.05}))["rollback"] == []   # > 60 s ago
    with pytest.raises(ValueError):
        UP.UnitArbiter(Const("accept+rb"))(obs(5, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0}))


def test_feasible_open_rule_skips_dwell_blocked_requests():
    pol = Const("reject")
    arb = UP.UnitArbiter(pol, open_rule="feasible")
    k = ("carrier", 0)
    arb(obs(49, [], {k: 2.0}))
    arb(obs(50, [], {k: 1.0}))                            # changed at 49: dwell 120 s
    assert arb(obs(100, [req("ES", k, 1.0, 2.0)], {k: 1.0}))["decisions"] == ["accept"] and not pol.calls
    assert arb(obs(169, [req("ES", k, 1.0, 2.0)], {k: 1.0}))["decisions"] == ["reject"] and len(pol.calls) == 1


def test_exposure_set_is_symmetrised_and_contains_every_pico_candidate():
    s = {"cells": 3, "neighbours": [[1], [], [0]]}
    assert UP.exposure_sets(s) == {0: (0, 1, 2), 1: (0, 1), 2: (0, 2)}
    env = E6Env(cfg_p("P3", scored=10.0, warm=0.0), log=False, wg3=True)
    ex = UP.exposure_sets(env.static())
    es = next(x for x in env.xapps if x.name == "ES")
    for pc, m in es.cand.items():
        assert m in ex[pc] and pc in ex[m]
    for c, s_ in ex.items():
        assert c in s_ and all(c in ex[v] for v in s_)


def test_split_mode_and_lock_equivalence_is_declared_for_p1_p3():
    assert UP.split_mode("half+rb") == ("half", True) and UP.split_mode("reject") == ("reject", False)
    with pytest.raises(ValueError):
        UP.split_mode("maybe")
    assert UP.LOCK_EQUIV == {"P1": "reject", "P3": "reject"}


# ------------------------------------------------------------------------------------------------ lock == reject
def _run(cfg, mode_of):
    env = E6Env(cfg, log=False, wg3=True)
    pol = lambda u: (mode_of(u), 1.0)                     # noqa: E731
    arb = UP.UnitArbiter(pol, T=60.0, warmup_s=cfg.warmup_s)
    writers = {}
    while env.sec < env.total_s:
        o = env.step_propose()
        for r in o["requests"]:
            writers.setdefault(r["knob"], set()).add(r["xapp"])
        env.step_apply(arb(o))
    return env, writers


@pytest.mark.parametrize("pair", ["P1", "P3"])
def test_lock_is_plant_identical_to_reject_in_p1_p3(pair):
    cfg = cfg_p(pair)

    def mode(which):                                       # every other unit locked / rejected, the rest accepted
        return lambda u: which if (u["c"] + int(u["t0"])) % 2 == 0 else "accept"

    e_lock, writers = _run(cfg, mode("lock"))
    e_rej, _ = _run(cfg, mode("reject"))
    assert all(len(v) == 1 for v in writers.values())      # one writer per knob (the premise)
    assert e_lock.stats["locks"] > 0 and e_rej.stats["locks"] == 0
    assert e_lock.stats["changes"] > 0
    assert e_lock.plant.sla == e_rej.plant.sla
    assert np.array_equal(e_lock.plant.q, e_rej.plant.q) and e_lock.config() == e_rej.config()
    assert e_lock.stats["changes"] == e_rej.stats["changes"]
