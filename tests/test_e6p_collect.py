"""cdd_oran.decision.collect_p: pi0 tables and draws, all-accept pi0 == accept-all (bit-identical), the per-cell KPI
tap, per-unit logs, reproduction inside env.copy."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import units_p as UP
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env


def cfg_p3(seed=6, warm=60.0, scored=90.0, lf=1.0):
    return C.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=lf, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=warm, scored_s=scored,
                      e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


def test_pi0_tables_match_the_design_and_the_propensity_floor():
    assert CP.PI0_HIGH["ES"] == {"accept": 0.5, "half": 0.2, "reject": 0.3}
    assert CP.PI0_HIGH["SliceGuarantee"] == {"accept": 0.5, "half": 0.15, "reject": 0.25, "accept+rb": 0.10}
    lo = CP.PI0_LOW
    assert lo["ES"] == pytest.approx({"accept": 0.78, "half": 0.10, "reject": 0.12})
    assert lo["PowerES"] == pytest.approx(lo["ES"])
    assert lo["SliceGuarantee"] == pytest.approx({"accept": 0.70, "half": 0.10, "reject": 0.10, "accept+rb": 0.10})
    for reg in ("high", "low"):
        for t in CP.PI0[reg].values():
            assert sum(t.values()) == pytest.approx(1.0) and min(t.values()) >= CP.P_MIN - 1e-12
    assert CP.low_table(CP.PI0_HIGH["ES"], p_min=None) == pytest.approx({"accept": 0.8, "half": 0.08, "reject": 0.12})
    assert [CP.regime_for(j) for j in range(4)] == ["high", "low", "high", "low"]


def test_draws_are_keyed_by_seed_cell_xapp_t0_and_follow_the_table():
    pol = CP.RandomizedUnitPolicy(123, "high")
    u = {"c": 4, "x": "SliceGuarantee", "x_idx": 2, "t0": 300.0}
    first = pol(dict(u))
    assert all(pol(dict(u)) == first for _ in range(5))                  # state-free: same key -> same draw
    assert CP.RandomizedUnitPolicy(123, "high").draw_u(u) == \
        float(np.random.default_rng([123, 6612, 4, 2, 300]).random())
    n, cnt = 0, {}
    for c in range(24):
        for t0 in range(120, 720):
            m, p = pol({"c": c, "x": "SliceGuarantee", "x_idx": 2, "t0": float(t0)})
            cnt[m] = cnt.get(m, 0) + 1
            n += 1
            assert p == CP.PI0_HIGH["SliceGuarantee"][m]
    for m, p in CP.PI0_HIGH["SliceGuarantee"].items():
        assert cnt[m] / n == pytest.approx(p, abs=0.01)
    with pytest.raises(ValueError):
        CP.RandomizedUnitPolicy(1, "mid")


def test_all_accept_pi0_is_bit_identical_to_accept_all_and_the_tap_matches_sla():
    cfg = cfg_p3()
    ref = E6Env(cfg, log=False, wg3=True)
    while ref.sec < ref.total_s:
        ref.step(None)
    res = CP.run_collection(cfg, CP.RandomizedUnitPolicy(cfg.seed, tables={x: {"accept": 1.0} for x in CP.PI0_HIGH}))
    env = res["env"]
    assert env.plant.sla == ref.plant.sla and env.stats == ref.stats
    assert np.array_equal(env.plant.q, ref.plant.q) and env.config() == ref.config()
    assert res["units"] and all(u["mode"] == "accept" and u["p"] == 1.0 for u in res["units"])
    tap, S = res["tap"], env.plant.sla
    assert tap.pv.sum() == S["prot_viol"] and tap.v.sum() == S["viol_ue_s"]
    assert tap.e.sum() == pytest.approx(S["energy_j"], rel=1e-12)
    assert S["prot_viol"] > 0 and S["energy_j"] > 0


def test_unit_log_context_propensity_and_window_kpis():
    cfg = cfg_p3(seed=7)
    res = CP.run_collection(cfg, CP.RandomizedUnitPolicy(cfg.seed, "high"))
    units = res["units"]
    assert {u["x"] for u in units} >= {"ES", "SliceGuarantee"}
    assert {u["mode"] for u in units} >= {"accept", "half", "reject"}
    for u in units:
        assert u["t0"] >= cfg.warmup_s
        assert u["p"] == CP.PI0_HIGH[u["x"]][u["mode"]]
        assert u["c"] in u["exp"]
        k = u["kpi"]
        assert k["trunc"] or k["secs"] == UP.T_UNIT
        assert k["e"] >= k["e_own"] > 0 and k["pv"] >= k["pv_own"] >= 0 and k["v"] >= k["v_own"] >= 0
        assert "own_prot_below_frac" in u["ctx"] and "nbr_max_prb_util" in u["ctx"]
        assert set(u["ctx"]) == set(units[0]["ctx"])
    ok = [u for u in units if not u["kpi"]["trunc"]]
    assert ok and any(u["kpi"]["pv"] > 0 for u in units)
    # independent replay: per-second cumulative tap arrays -> window sums over N(c), seconds t0+1 .. t0+T
    env = E6Env(cfg, log=False, wg3=True)
    tap = CP.install_tap(env)
    arb = UP.UnitArbiter(CP.RandomizedUnitPolicy(cfg.seed, "high"), warmup_s=cfg.warmup_s)
    cum = {}
    while env.sec < env.total_s:
        o = env.step_propose()
        cum[int(o["t"])] = tap.arrays()
        env.step_apply(arb(o))
    for u in ok:
        t0, t1, ex = int(u["t0"]), int(u["t0"] + UP.T_UNIT), u["exp"]
        want = [float(cum[t1][i][ex].sum() - cum[t0][i][ex].sum()) for i in range(3)]
        assert [u["kpi"]["pv"], u["kpi"]["e"], u["kpi"]["v"]] == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_collection_reproduces_inside_env_copy():
    cfg = cfg_p3(seed=8, scored=60.0)
    pol = CP.RandomizedUnitPolicy(cfg.seed, "high")
    env = E6Env(cfg, log=False, wg3=True)
    CP.install_tap(env)
    arb = UP.UnitArbiter(pol, warmup_s=cfg.warmup_s)
    while env.sec < 90:
        env.step_apply(arb(env.step_propose()))
    env2, arb2 = env.copy(), copy.deepcopy(arb)
    assert CP.get_tap(env2) is not CP.get_tap(env) and CP.get_tap(env2).plant is env2.plant
    n0 = len(arb.units)
    for e, a in ((env, arb), (env2, arb2)):
        while e.sec < e.total_s:
            e.step_apply(a(e.step_propose()))
    key = [(u["c"], u["x"], u["t0"], u["mode"], u["p"]) for u in arb.units[n0:]]
    assert key and key == [(u["c"], u["x"], u["t0"], u["mode"], u["p"]) for u in arb2.units[n0:]]
    assert env.plant.sla == env2.plant.sla
    assert np.array_equal(CP.get_tap(env).pv, CP.get_tap(env2).pv)
