"""E6-P step-2 DEV arms driver (scratchpad/e6_dev/e6p_step2_dev.py): job list, seed guard, obs-only policies and the
summary math on synthetic records. Fast (no episode is simulated). Skipped when the scratchpad driver is absent."""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
if not os.path.exists(os.path.join(HERE, "e6p_step2_dev.py")) or not os.path.exists(os.path.join(HERE, "e6p_screen.py")):
    pytest.skip("scratchpad/e6_dev/e6p_step2_dev.py not present", allow_module_level=True)
sys.path.insert(0, HERE)
import e6p_step2_dev as D  # noqa: E402

N_CELLS = 4                     # cells 0, 1 macros; 2, 3 picos; neighbours 0-1, 0-2, 1-3


# ---------------------------------------------------------------------------------------------- jobs / seeds
def test_jobs_cover_seeds_and_arms():
    J = D.jobs("dev")
    assert len(J) == 40 * len(D.ARMS) == len(set(J))
    assert {s for s, _ in J} == set(range(184200, 184240))
    assert D.ARMS == ("freeze", "sub:ES+PowerES", "noarb", "sub:ES", "sub:PowerES", "sub:SliceGuarantee",
                      "B1", "B2", "M1", "M2")
    assert [a for s, a in J[:len(D.ARMS)]] == list(D.ARMS) and all(s == 184200 for s, _ in J[:len(D.ARMS)])
    with pytest.raises(SystemExit):
        D.jobs("eval")


@pytest.mark.parametrize("seed", [184199, 184240, 155300, 183300, 0, 185000])
def test_check_seed_rejects_outside_block(seed):
    with pytest.raises(AssertionError):
        D.check_seed(seed)


def test_check_seed_accepts_block():
    assert D.check_seed(184200) == 184200 and D.check_seed(184239) == 184239
    with pytest.raises(AssertionError):
        D.run_job(184100, "noarb", 1.0, smoke=True)          # guard fires before any simulation


# ---------------------------------------------------------------------------------------------- policies
class ObsDict(dict):
    """obs that records which top-level keys a policy reads."""
    seen: set = set()

    def __getitem__(self, k):
        ObsDict.seen.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        ObsDict.seen.add(k)
        return super().get(k, default)


def static():
    return {"cells": N_CELLS, "neighbours": [[1, 2], [0, 3], [0], [1]], "is_macro": np.array([1, 1, 0, 0], bool),
            "knobs": [], "xapps": {}}


def fast(util, ll=None):
    return {"gran": "fast", "t0": 0, "t1": 1, "prb_util": np.asarray(util, float),
            "ll_delay_p95": np.asarray(ll if ll is not None else [np.nan] * N_CELLS, float)}


def req(x, knob, cur, prop):
    return {"xapp": x, "knob": knob, "cur": float(cur), "prop": float(prop)}


REQS = [req("PowerES", ("ptx", 0), -3.0, 0.0),        # ptx up, cell 0
        req("PowerES", ("ptx", 1), 0.0, -3.0),        # ptx down, cell 1
        req("ES", ("sleep", 2), 0.0, 1.0),            # pico sleep
        req("ES", ("sleep", 3), 1.0, 0.0),            # pico wake
        req("ES", ("carrier", 0), 2.0, 1.0),          # carrier off
        req("ES", ("carrier", 1), 1.0, 2.0),          # carrier on
        req("SliceGuarantee", ("prot_min", 0), 0.0, 0.1),
        req("PowerES", ("ptx", 1), -3.0, 0.0)]        # ptx up, cell 1


def obs(t, reqs=REQS, reports=()):
    cfg = {("ptx", 0): -3.0, ("ptx", 1): 0.0, ("sleep", 2): 0.0, ("sleep", 3): 1.0, ("carrier", 0): 2.0,
           ("carrier", 1): 1.0, ("prot_min", 0): 0.0}
    return ObsDict(t=float(t), new_reports=list(reports), config=cfg, requests=list(reqs), static=static(),
                   locked={}, changes=0, churn_cap=None)


def test_blanket_gates_follow_F():
    b1, b2 = D.make_arbiter("B1", 120.0), D.make_arbiter("B2", 120.0)
    assert b1(obs(120))["decisions"] == ["accept"] * len(REQS)          # F: gate only for t > warm-up
    d1 = b1(obs(121))["decisions"]
    assert d1 == ["reject", "accept", "accept", "accept", "accept", "accept", "accept", "reject"]
    d2 = b2(obs(121))["decisions"]
    assert d2 == ["reject", "accept", "reject", "accept", "reject", "accept", "accept", "reject"]
    assert b2.n == {"rej_car_off": 1, "rej_ptx_up": 2, "rej_sleep": 1}


def test_sat_gate_follows_G_and_ll_guard():
    m1, m2 = D.make_arbiter("M1", 120.0), D.make_arbiter("M2", 120.0)
    util = [1.0, 0.5, 0.2, 0.2]                                         # cell 0 saturated, cell 1 not
    assert m1(obs(119, reports=[fast(util)]))["decisions"] == ["accept"] * len(REQS)
    d = m1(obs(120, reports=[fast(util)]))["decisions"]                  # G: gate for t >= warm-up
    assert d == ["reject"] + ["accept"] * (len(REQS) - 1)
    # M2: region of cell 0 = {0, 1, 2}; LL delay flat -> keeps rejecting; rising -> releases
    for t in range(100, 120):
        m2(obs(t, reqs=[], reports=[fast(util, [0.01, 0.01, 0.01, 0.5])]))
    assert m2(obs(120, reports=[fast(util, [0.01] * 3 + [0.5])]))["decisions"][0] == "reject"
    for t in range(121, 131):
        m2(obs(t, reqs=[], reports=[fast(util, [0.05, 0.05, 0.05, 0.0])]))
    assert m2.ll_rising(0, 131.0) is True
    d = m2(obs(131, reports=[fast(util, [0.05] * 3 + [0.0])]))["decisions"]
    assert d == ["accept"] * len(REQS) and m2.n["ll_release"] == 1
    # the LL rise must be in the requesting cell's region: cell 3 (not in N(0)) rising alone does not release
    m3 = D.make_arbiter("M2", 120.0)
    for t in range(100, 110):
        m3(obs(t, reqs=[], reports=[fast(util, [0.01, 0.01, 0.01, 0.01])]))
    for t in range(110, 121):
        m3(obs(t, reqs=[], reports=[fast(util, [0.01, 0.01, 0.01, 0.9])]))
    assert m3(obs(121, reports=[fast(util, [0.01] * 3 + [0.9])]))["decisions"][0] == "reject"


def test_policies_read_only_obs():
    ObsDict.seen = set()
    for arm in D.POLICY_ARMS:
        a = D.make_arbiter(arm, 120.0)
        for t in (119, 121, 122):
            out = a(obs(t, reports=[fast([1.0, 1.0, 0.0, 0.0], [0.1] * N_CELLS)]))
            assert len(out["decisions"]) == len(REQS) and out["writes"] == [] and out["rollback"] == []
    assert ObsDict.seen <= {"t", "new_reports", "config", "requests", "static"}
    for obj in (D.BlanketGate, D.SatPtxGate, D.direction):
        src = inspect.getsource(obj)
        assert "plant" not in src and "env." not in src and "sla" not in src


# ---------------------------------------------------------------------------------------------- summary math
def rec(seed, arm, pv, E, viol=100.0, nonp=50.0, ll=10.0, rlf=5.0, pu=3600.0, ue=3600.0):
    return {"kind": "job", "schema": D.SCHEMA, "key": [seed, arm], "seed": seed, "arm": arm, "smoke": False,
            "prot_viol": pv, "prot_ue_s": pu, "energy_j": E, "viol_ue_s": viol, "ue_s": ue,
            "nonprot_embb_viol": nonp, "ll_viol": ll, "rlf": rlf, "cpu_s": 1.0}


def synth(seeds=(184200, 184201, 184202)):
    """Per seed (psvr = pv since prot_ue_s = 3600): AA 200, singles 150/180/190 -> V_ref 150, den 50;
    freeze E 1000, A E 800 (saving 200); B1 V 180, B2 V 160 with E 880 (retention .6), M1 V 170 with ll 12 (1.2 x AA)."""
    out = []
    for s in seeds:
        out += [rec(s, "freeze", 300, 1000), rec(s, "sub:ES+PowerES", 260, 800), rec(s, "noarb", 200, 810),
                rec(s, "sub:ES", 190, 820), rec(s, "sub:PowerES", 180, 900), rec(s, "sub:SliceGuarantee", 150, 1000),
                rec(s, "B1", 180, 800), rec(s, "B2", 160, 880), rec(s, "M1", 170, 790, ll=12.0),
                rec(s, "M2", 175, 812, viol=110.0)]
    return out


def test_summary_math(tmp_path):
    import json
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in synth()) + "\n" + json.dumps(dict(rec(184203, "noarb", 1, 1),
                                                                                   smoke=True)) + "\n")
    rep = D.summary([str(f)], n_boot=200)
    A = rep["arms"]
    assert rep["n_seeds_complete"] == 3 and A["noarb"]["n_seeds"] == 3        # smoke record ignored
    assert A["noarb"]["V_AA"] == pytest.approx(200) and A["noarb"]["V_ref"] == pytest.approx(150)
    assert A["noarb"]["ref_arm"] == "sub:SliceGuarantee" and A["noarb"]["den"] == pytest.approx(50)
    assert A["noarb"]["R"] == pytest.approx(0) and A["sub:SliceGuarantee"]["R"] == pytest.approx(1)
    assert A["B1"]["R"] == pytest.approx(0.4) and A["B2"]["R"] == pytest.approx(0.8)
    assert A["B2"]["dR_vs_base"] == pytest.approx(0.4) and A["B1"]["dR_vs_base"] == pytest.approx(0)
    assert A["B1"]["R_fixed_den"] == pytest.approx(20 / D.DEN_GATE_A)
    assert A["noarb"]["retention"] == pytest.approx(190 / 200) and A["noarb"]["eligible"]
    assert A["B2"]["retention"] == pytest.approx(0.6) and not A["B2"]["matched"] and not A["B2"]["eligible"]
    assert A["M1"]["guard_ratio"]["ll_viol"] == pytest.approx(1.2) and not A["M1"]["guard_ok"]
    assert A["M1"]["retention"] == pytest.approx(1.05) and A["M1"]["matched"] and not A["M1"]["eligible"]
    assert A["M2"]["guard_ratio"]["svr"] == pytest.approx(1.1) and A["M2"]["guard_ok"]     # <= 1.10 inclusive
    assert A["M2"]["retention"] == pytest.approx(0.94) and A["M2"]["eligible"]
    # identical seeds -> degenerate bootstrap: CI collapses on the point estimate
    assert A["B1"]["R_ci90"] == pytest.approx([0.4, 0.4]) and A["B2"]["dR_vs_base_ci90"] == pytest.approx([0.4, 0.4])


def test_summary_bootstrap_paired_and_missing(tmp_path):
    import json
    rs = synth()
    for r in rs:                                 # make seeds differ: scale protected violations of every arm per seed
        r["prot_viol"] *= {184200: 0.8, 184201: 1.0, 184202: 1.3}[r["seed"]]
    rs = [r for r in rs if not (r["seed"] == 184202 and r["arm"] == "M2")]      # one missing job
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    rep = D.summary([str(f)], n_boot=500)
    A = rep["arms"]
    assert rep["missing"] == {"M2": [184202]} and A["M2"]["n_seeds"] == 2 and A["B1"]["n_seeds"] == 3
    # every seed has the same relative structure -> R is scale-free and the paired CI stays tight around it
    lo, hi = A["B2"]["R_ci90"]
    assert lo <= A["B2"]["R"] <= hi and hi - lo < 0.2
    lo, hi = A["B2"]["dR_vs_base_ci90"]
    assert lo <= A["B2"]["dR_vs_base"] <= hi
    assert np.isfinite(A["M2"]["R"])
