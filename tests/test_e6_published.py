"""Behavioural tests for the E6 re-implementations of published conflict-mitigation methods (no golden numbers)."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.published import (
    CMF,
    QACM,
    Djidjev,
    KPIPredictor,
    Pacifista,
    infer_row,
    int_distance,
    one_step_values,
    run_logged,
    transitions,
)

NC = 4
STATIC = {"xapps": {"MRO": [], "TS": [], "ES": [], "SLICE": []}}


def fast(t, delay, util=0.5):
    return {"gran": "fast", "t0": t - 1, "t1": t, "ll_delay_p95": np.full(NC, delay, float),
            "prb_util": np.full(NC, util, float)}


def thp(t, p5):
    return {"gran": "thp", "t0": t - 5, "t1": t, "embb_thp_p5": np.full(NC, p5, float)}


def obs(t, reqs, reports=(), config=None):
    return {"t": float(t), "new_reports": list(reports), "requests": reqs, "static": STATIC,
            "config": config or {("cio", 0, 1): 0.0, ("ll_ratio", 0): 0.1, ("carrier", 0): 2.0}}


def req(x, k, cur, prop, t=1.0):
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": t}


class LinearPredictor:
    """Stub predictor: z-KPI moves linearly with the pending change (per knob type / kpi slope)."""

    def __init__(self, slope):
        self.slope = slope
        self.zs = {"ll_delay": (0.05, 0.05), "embb_thp": (3e6, 1e6), "ho_fail": (0.0, 0.05), "energy": (0, 1)}

    def z(self, name, v):
        mu, sd = self.zs[name]
        return (np.asarray(v, float) - mu) / sd

    def predict_z(self, typ, name, X):
        X = np.asarray(X, float)
        return self.z(name, X[:, 5]) + self.slope.get((typ, name), 0.0) * X[:, 3]


# ------------------------------------------------------------------------------------------ QACM
def test_qacm_rejects_request_that_degrades_a_violating_xapp():
    # SLICE's LL delay is above target on the cells; TS's CIO push is predicted to raise it further
    q = QACM(LinearPredictor({("cio", "ll_delay"): 1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.2), thp(5, 3e6)]))
    assert out["decisions"] == ["reject"] and q.stats["conflicts"] == 1


def test_qacm_accepts_when_no_other_xapp_violates():
    q = QACM(LinearPredictor({("cio", "ll_delay"): 1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.01), thp(5, 3e6)]))
    assert out["decisions"] == ["accept"] and q.stats["conflicts"] == 0


def test_qacm_accepts_request_that_helps_the_violating_xapp():
    q = QACM(LinearPredictor({("cio", "ll_delay"): -1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.2), thp(5, 3e6)]))
    assert out["decisions"] == ["accept"] and q.stats["conflicts"] == 1


def test_qacm_modifies_to_an_interior_compromise():
    # SLICE asks +0.10 LL ratio; TS's p5 throughput is below target and falls with the ratio, LL delay falls with it:
    # the cost-minimising value lies strictly between the current value and the request -> MODIFY
    pred = LinearPredictor({("ll_ratio", "ll_delay"): -10.0, ("ll_ratio", "embb_thp"): -1.0})
    q = QACM(pred)
    out = q(obs(5, [req("SLICE", ("ll_ratio", 0), 0.1, 0.2)], [fast(5, 0.12), thp(5, 1.9e6)]))
    d = out["decisions"][0]
    assert isinstance(d, tuple) and d[0] == "modify" and 0.1 < d[1] < 0.2


def test_qacm_direct_same_knob_conflict_keeps_one_request():
    q = QACM(LinearPredictor({}))
    out = q(obs(5, [req("MRO", ("cio", 0, 1), 0.0, -1.0), req("TS", ("cio", 0, 1), 0.0, 1.0)],
                [fast(5, 0.01), thp(5, 3e6)]))
    assert sorted(out["decisions"]) == ["accept", "reject"]


def test_one_step_values_respect_actuator_step():
    assert one_step_values(("cio", 0, 1), 0.0) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert one_step_values(("ttt", 0), 320.0) == [256.0, 320.0, 480.0]
    assert one_step_values(("carrier", 0), 2.0) == [1.0, 2.0]


# ------------------------------------------------------------------------------------------ CMF priority / SBD
def test_cmf_priority_blocks_lower_priority_within_span_only():
    arb = CMF(order=("SLICE", "MRO", "TS", "ES"))
    k = ("cio", 0, 1)
    assert arb(obs(1, [req("MRO", k, 0.0, 1.0)]))["decisions"] == ["accept"]
    assert arb(obs(5, [req("TS", k, 1.0, 2.0)]))["decisions"] == ["reject"]          # MRO's decision in force
    assert arb(obs(40, [req("TS", k, 1.0, 2.0)]))["decisions"] == ["accept"]         # MRO span (30 s) expired
    assert arb(obs(41, [req("MRO", k, 2.0, 1.0)]))["decisions"] == ["accept"]        # higher priority wins


def test_cmf_sbd_sets_back_to_default():
    arb = CMF(order=("MRO", "TS"), mode="sbd")
    k = ("cio", 0, 1)
    arb(obs(1, [req("MRO", k, 0.0, 1.0)]))
    cfg = {k: 1.0, ("ll_ratio", 0): 0.1, ("carrier", 0): 2.0}
    assert arb(obs(5, [req("TS", k, 1.0, 2.0)], config=cfg))["decisions"] == [("modify", 0.0)]


def test_cmf_manifest_groups_ignore_unrelated_knob_types():
    arb = CMF(order=("SLICE", "TS"))
    arb(obs(1, [req("SLICE", ("ll_ratio", 0), 0.1, 0.15)]))
    assert arb(obs(1.5, [req("TS", ("cio", 0, 1), 0.0, 1.0)]))["decisions"] == ["accept"]
    arb_cell = CMF(order=("SLICE", "TS"), groups="cell")
    arb_cell(obs(1, [req("SLICE", ("ll_ratio", 0), 0.1, 0.15)]))
    assert arb_cell(obs(1.5, [req("TS", ("cio", 0, 1), 0.0, 1.0)]))["decisions"] == ["reject"]


# ------------------------------------------------------------------------------------------ PACIFISTA
def test_int_distance_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    assert int_distance(a, a) == 0.0
    assert 0 < int_distance(a, a + 0.5) < int_distance(a, a + 2.0) <= 1.0


def test_pacifista_greedy_respects_tolerance_and_priority():
    p = Pacifista(delta_tol=0.3, order=("SLICE", "MRO", "TS", "ES"))
    p.sigma = {}
    for (a, b), s in {("SLICE", "MRO"): 0.1, ("SLICE", "TS"): 0.5, ("MRO", "TS"): 0.1, ("SLICE", "ES"): 0.2,
                      ("MRO", "ES"): 0.2, ("TS", "ES"): 0.1}.items():
        p.sigma[(a, b)] = p.sigma[(b, a)] = s
    assert p.deploy_set(("ES", "TS", "MRO", "SLICE")) == ("SLICE", "MRO", "ES")
    arb = p.arbiter(("ES", "TS", "MRO", "SLICE"))
    out = arb(obs(1, [req("TS", ("cio", 0, 1), 0.0, 1.0), req("ES", ("carrier", 0), 2.0, 1.0)]))
    assert out["decisions"] == ["reject", "accept"]


# ------------------------------------------------------------------------------------------ Djidjev & Kaminski
def test_infer_row_recovers_or_parent():
    rng = np.random.default_rng(1)
    BP = rng.uniform(size=(32, 7)) < 0.2
    bK = BP[:, 2] | BP[:, 4]
    row = infer_row(BP, bK)
    assert row[2] and row[4] and row.sum() == 2


def test_djidjev_rejects_only_through_tracked_parent_of_violating_kpi():
    d = Djidjev(kpis=("ll_delay",))
    d.L[("ll_delay", 0)] = np.array([1, 0, 0, 0, 0, 0, 0], bool)       # cio_out of cell 0 is a tracked parent
    d.view.ingest([fast(5, 0.2)])
    o = obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0), req("TS", ("cio", 1, 0), 0.0, 1.0),
                req("SLICE", ("ll_ratio", 0), 0.1, 0.15)])
    o["config"] = {("cio", 0, 1): 0.0, ("cio", 1, 0): 0.0, ("ll_ratio", 0): 0.1}
    assert d(o)["decisions"] == ["reject", "accept", "accept"]


# ------------------------------------------------------------------------------------------ end to end (short)
@pytest.fixture(scope="module")
def dev_log():
    return run_logged(C.E6Config(seed=11, mix="M4", load="high", warmup_s=10, scored_s=50))[1]


def test_transitions_and_predictor_fit_from_logs(dev_log):
    data = transitions([dev_log], ("ll_delay", "embb_thp"))
    assert ("ll_ratio", "ll_delay") in data
    X, y = data[("ll_ratio", "ll_delay")]
    assert X.shape[1] == 7 and len(X) == len(y) > 0 and np.all(np.isfinite(X))
    pred = KPIPredictor(kpis=("ll_delay", "embb_thp")).fit([dev_log])
    assert pred.report[("ll_ratio", "ll_delay")]["model"] in ("ANN", "PR")


def test_published_arbiters_run_closed_loop(dev_log):
    cfg = C.E6Config(seed=12, mix="M4", load="high", warmup_s=10, scored_s=20)
    pred = KPIPredictor(kpis=("ll_delay", "embb_thp", "ho_fail")).fit([dev_log])
    arbs = {"qacm": QACM(pred), "cmf": CMF(), "djidjev": Djidjev().fit([dev_log])}
    for name, arb in arbs.items():
        s = E6Env(cfg, log=False).run(arb)
        assert s["req"] > 0 and s["acc"] + s["mod"] <= s["req"], name
    q = arbs["qacm"].stats
    assert q["accept"] + q["reject"] + q["modify"] > 0
