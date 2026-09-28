"""Behavioural tests for the E6-P screen arms (docs/benchmark/E6P_SCREEN_PROTOCOL.md sec. 5): cell-priority lock,
per-region subset / hindsight static, and the QACM mapping for the E6-P xApps and knob types. DEV seeds only, no golden
numbers."""
from __future__ import annotations

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.baselines import (
    CellPriorityLock,
    all_subsets,
    region_hindsight,
    region_subset,
)
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.published import MANIFEST_P, QACM, QOS_P, kpi_of, one_step_values

NC = 4


def req(x, k, cur, prop, t=1.0):
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": t}


def obs(t, reqs, config, reports=(), xapps=("ES", "SliceGuarantee")):
    return {"t": float(t), "requests": reqs, "config": dict(config), "new_reports": list(reports),
            "static": {"xapps": {x: [] for x in xapps}}}


# ------------------------------------------------------------------------------------------ cell-priority lock
def test_cell_lock_higher_priority_change_blocks_lower_priority_on_that_cell_for_60s():
    arb = CellPriorityLock(("SliceGuarantee", "ES"))
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0, ("carrier", 4): 2.0}
    d = arb(obs(100, [req("ES", ("carrier", 3), 2.0, 1.0), req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)], cfg))
    assert d["decisions"] == ["reject", "accept"]                        # same second: priority order
    cfg[("prot_min", 3)] = 0.05                                          # the change took effect
    d = arb(obs(101, [req("ES", ("carrier", 3), 2.0, 1.0), req("ES", ("carrier", 4), 2.0, 1.0)], cfg))
    assert d["decisions"] == ["reject", "accept"]                        # other cells are free
    cfg[("carrier", 4)] = 1.0
    d = arb(obs(159, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))
    assert d["decisions"] == ["reject"]
    d = arb(obs(160, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))
    assert d["decisions"] == ["accept"]                                  # hold expired after 60 s


def test_cell_lock_never_blocks_the_higher_priority_xapp():
    arb = CellPriorityLock(("SliceGuarantee", "ES"))
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0}
    assert arb(obs(10, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["accept"]
    cfg[("carrier", 3)] = 1.0
    d = arb(obs(11, [req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)], cfg))
    assert d["decisions"] == ["accept"]


def test_cell_lock_ignores_accepted_requests_that_did_not_take_effect():
    arb = CellPriorityLock(("SliceGuarantee", "ES"))
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0}
    arb(obs(10, [req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)], cfg))
    # the actuator NACKed it: config unchanged -> no hold on cell 3
    assert arb(obs(11, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["accept"]


# ------------------------------------------------------------------------------------------ per-region hindsight
def test_region_hindsight_picks_the_best_admissible_subset_per_region():
    xapps = ("A", "B")
    cost = {0: {("A", "B"): 5, ("A",): 3, ("B",): 4, (): 1},             # region 0: () is cheapest but inadmissible
            1: {("A", "B"): 5, ("A",): 6, ("B",): 2, (): 7}}
    calls = []

    def evaluate(kb):
        calls.append(kb)
        return sum(cost[g][tuple(v)] for g, v in kb.items()), kb[0] != ()

    res = region_hindsight(evaluate, [0, 1], xapps)
    assert res["best"] == {0: ("A",), 1: ("B",)} and res["best_ok"]
    assert res["best_val"] == 3 + 2
    assert calls[0] == {0: ("A", "B"), 1: ("A", "B")}                     # starts from accept-all
    assert res["n_eval"] == 1 + 3 + 3                                     # one pass, every other subset per region
    assert len(all_subsets(("A", "B", "C"))) == 8


def test_region_hindsight_keeps_incumbent_without_strict_improvement():
    res = region_hindsight(lambda kb: (1.0, True), [0, 1, 2], ("A", "B"))
    assert all(v == ("A", "B") for v in res["best"].values())


def test_region_subset_only_changes_kept_xapps_knobs_in_kept_regions():
    P = C.E6PConfig(ptx_on=True, prot_on=True, xapps=("SliceGuarantee",))
    cfg = C.E6Config(seed=3, mix="ES", load="high", warmup_s=10, scored_s=80, e6p=P)
    env = E6Env(cfg, log=False, wg3=True)
    site = np.asarray(env.plant.lay.cell_site)
    regions = sorted({int(x) for x in site})
    keep = {g: (("SliceGuarantee",) if g < 4 else ()) for g in regions}
    init = env.config()
    env.run(region_subset(keep, site))
    changed = [k for k, v in env.config().items() if v != init[k]]
    assert changed and all(k[0] == "prot_min" and site[k[1]] < 4 for k in changed)
    assert env.stats["rej"] > 0


# ------------------------------------------------------------------------------------------ QACM E6-P mapping
class LinearPredictor:
    def __init__(self, slope):
        self.slope = slope
        self.zs = {"prot_below": (0.02, 0.02), "edge_sinr": (0.0, 3.0), "energy": (0.0, 1.0)}

    def z(self, name, v):
        mu, sd = self.zs[name]
        return (np.asarray(v, float) - mu) / sd

    def predict_z(self, typ, name, X):
        X = np.asarray(X, float)
        return self.z(name, X[:, 5]) + self.slope.get((typ, name), 0.0) * X[:, 3]


def fast(t, below, edge=0.0):
    return {"gran": "fast", "t0": t - 1, "t1": t, "prb_util": np.full(NC, 0.3), "prot_below_frac": np.full(NC, below),
            "edge_sinr_p": np.full(NC, edge)}


def test_one_step_values_cover_e6p_knob_types():
    assert one_step_values(("ptx", 0), 0.0) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert one_step_values(("ptx", 0), -8.0) == [-9.0, -8.0, -7.0, -6.0, -5.0]
    assert one_step_values(("prot_min", 0), 0.0) == [0.0, 0.05, 0.1]
    assert one_step_values(("sleep", 5), 0.0) == [0.0, 1.0]
    rep = fast(5, 0.1, -7.0)
    assert np.all(kpi_of(rep, "prot_below") == 0.1) and np.all(kpi_of(rep, "edge_sinr") == -7.0)


def test_qacm_e6p_rejects_carrier_off_when_protected_floors_fail_and_it_hurts_them():
    q = QACM(LinearPredictor({("carrier", "prot_below"): -2.0}), manifest=MANIFEST_P, qos=QOS_P)
    cfg = {("carrier", 1): 2.0}
    out = q(obs(5, [req("ES", ("carrier", 1), 2.0, 1.0)], cfg, [fast(5, 0.2)]))       # fewer carriers -> worse
    assert out["decisions"] == ["reject"] and q.stats["conflicts"] == 1
    q2 = QACM(LinearPredictor({("carrier", "prot_below"): -2.0}), manifest=MANIFEST_P, qos=QOS_P)
    out = q2(obs(5, [req("ES", ("carrier", 1), 2.0, 1.0)], cfg, [fast(5, 0.0)]))        # floors met: no conflict
    assert out["decisions"] == ["accept"] and q2.stats["conflicts"] == 0


def test_qacm_e6p_direct_ptx_conflict_between_power_xapps():
    q = QACM(LinearPredictor({("ptx", "edge_sinr"): 1.0}), manifest=MANIFEST_P, qos=QOS_P)
    out = q(obs(5, [req("PowerES", ("ptx", 1), 0.0, -3.0), req("Coverage", ("ptx", 1), 0.0, 3.0)],
                {("ptx", 1): 0.0}, [fast(5, np.nan, -8.0)], xapps=("PowerES", "Coverage")))
    assert out["decisions"] == ["reject", "accept"]                       # edge below Q_in: raising power wins
