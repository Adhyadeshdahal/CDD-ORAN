"""Behavioural mechanism tests for the E6-P extension (docs/benchmark/E6P_SPEC.md; ``config.E6PConfig``): Tx-power
knob, protected slice with a work-conserving min PRB share, graceful pico sleep, E6-P KPM fields, E6-P xApps and the
re-seeded lookahead copy. DEV seeds 0-30 only (the protocol's own M1-M6 seeds are not touched here). The tests check
directions and invariants of the mechanics, not calibrated magnitudes.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6 import sim
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import feasible, knob_get, knob_set
from cdd_oran.envs.e6.xapps import ES
from cdd_oran.envs.e6.xapps_p import ESPico

PTX = C.E6PConfig(ptx_on=True)
PROT = C.E6PConfig(prot_on=True)
BOTH = C.E6PConfig(ptx_on=True, prot_on=True)


def _cfg(seed=5, **kw):
    base = dict(seed=seed, load="medium", mobility="mixed", mix="none", warmup_s=10, scored_s=110)
    base.update(kw)
    return C.E6Config(**base)


def _freeze(obs):
    return {"decisions": ["reject"] * len(obs["requests"]), "writes": []}


def _run(cfg, arbiter, secs):
    env = E6Env(cfg, log=False)
    for _ in range(secs):
        env.step(arbiter)
    return env


def _fast(env, t_from):
    return [r for r in env.kpm.delivered if r["gran"] == "fast" and r["t1"] > t_from]


def _sinr(plant, g, occ):
    pw = 10 ** (g / 10) / 1000.0
    load = pw * occ[None, :]
    idx = np.arange(plant.n)
    s = pw[idx, plant.serv]
    return 10 * np.log10(s / (load.sum(1) - load[idx, plant.serv] + sim.NOISE_W))


def _ticks(p, k):
    for _ in range(k):
        p.tick()


# ------------------------------------------------------------------------------------------------ (a) Tx power
def test_ptx_step_follows_gain_math():
    a = sim.Plant(_cfg(e6p=PTX))
    _ticks(a, 50)
    c = int(np.argmax(np.bincount(a.serv, minlength=a.nc) * a.lay.is_macro))          # a populated macro
    ga = a._gains()
    awake = ga[:, c] > -299
    occ = np.full(a.nc, 0.5)
    sa = _sinr(a, ga, occ)
    own = a.serv == c
    near = ~own & (ga[:, c] >= ga[np.arange(a.n), a.serv] - 6.0)                          # neighbour-edge UEs of c
    assert near.any()
    for off in (-3.0, 3.0, -6.0):
        b = copy.deepcopy(a)
        b.ptx_off[c] = off
        gb = b._gains()
        np.testing.assert_allclose(gb[awake, c], ga[awake, c] + off)                     # own column (L3 input) shifts
        np.testing.assert_array_equal(np.delete(gb, c, 1), np.delete(ga, c, 1))          # other columns untouched
        sb = _sinr(b, gb, occ)
        d = sb - sa
        assert np.all(np.sign(d[~own]) * -np.sign(off) >= -1e-12)                        # interference moves the other way
        assert np.sign(d[near].mean()) == -np.sign(off)
        assert np.all(np.sign(d[own]) == np.sign(off))                                   # own served signal follows


def test_ptx_down_raises_neighbour_edge_and_lowers_own_power_dynamically():
    a = sim.Plant(_cfg(e6p=PTX))
    _ticks(a, 50)
    c = int(np.argmax(np.bincount(a.serv, minlength=a.nc) * a.lay.is_macro))
    b = copy.deepcopy(a)
    b.ptx_off[c] = -6.0
    en, edge = {}, {}
    for nm, p in (("a", a), ("b", b)):
        en[nm], edge[nm] = 0.0, []
        for _ in range(40):
            _ticks(p, C.TICKS_PER_CONTROL)
            ctr = p.take_counters()
            srv, val = np.concatenate(ctr["sinr_serv"]), np.concatenate(ctr["sinr_val"])
            edge[nm].append([np.percentile(val[srv == n], 5) if np.any(srv == n) else np.nan for n in a.lay.neighbours[c]])
            en[nm] += ctr["energy_j"][c]
    assert np.nanmean(edge["b"]) > np.nanmean(edge["a"])
    assert en["b"] < en["a"]


def test_ptx_energy_is_earth_with_scaled_pmax():
    p = sim.Plant(_cfg(e6p=PTX))
    c = int(np.nonzero(p.lay.is_macro)[0][0])
    p.ptx_off[c] = -3.0
    p.tick()
    rho, car = p.rho[c], p.n_car[c]
    want = car * (C.MACRO_P0_W + C.MACRO_DP * rho * C.MACRO_PMAX_W * 10 ** (-0.3)) + \
        (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX
    assert p.ctr["energy_j"][c] == pytest.approx(want * C.TICK_S, abs=1e-9)


def test_ptx_knob_registry_limits_and_quantisation():
    env = E6Env(_cfg(e6p=PTX), log=False)
    ks = [k for k in env.knobs if k[0] == "ptx"]
    assert [k[1] for k in ks] == list(np.nonzero(env.plant.lay.is_macro)[0])          # macro sectors only
    k = ks[0]
    assert feasible(env.plant, k, -10.3, 100.0, {}) == (-3.0, True)                   # max step 3 dB, 1 dB grid
    knob_set(env.plant, k, -8.0, 100.0)
    assert feasible(env.plant, k, -12.0, 200.0, {})[0] == -9.0                         # floor -9 dB
    assert not feasible(env.plant, k, -1.0, 100.0, {k: 95.0})[1]                       # 10 s dwell


# ------------------------------------------------------------------------------------------------ (b) min share
def _quota_plant(prot_min, prot_q, other_q, ll_ratio=0.0):
    cfg = _cfg(seed=7, frac_ll=0.0, embb_files_per_s=0.0, be_files_per_s=0.0,
               e6p=dataclasses.replace(PROT, prot_frac=0.5))
    p = sim.Plant(cfg)
    _ticks(p, 5)
    in_ok = p.int_until <= p.t * C.TICK_S
    cnt = np.bincount(p.serv[p.prot & in_ok], minlength=p.nc)
    oth = np.bincount(p.serv[~p.prot & in_ok], minlength=p.nc)
    c = int(np.argmax(np.minimum(cnt, oth) * p.lay.is_macro))
    assert cnt[c] >= 2 and oth[c] >= 2
    p.prot_min[:] = prot_min
    p.ll_ratio[:] = ll_ratio
    p.q[:] = np.where(p.prot, prot_q, other_q)
    p.ctr = p._new_counters()
    p.tick()
    return p, c


def test_idle_min_share_is_bit_identical_to_no_min_share():
    p0, _ = _quota_plant(prot_min=0.0, prot_q=0.0, other_q=1e9)
    p1, c = _quota_plant(prot_min=0.5, prot_q=0.0, other_q=1e9)
    np.testing.assert_array_equal(p1.q, p0.q)
    for key in ("prb_used", "prb_rsv", "prb_cap", "bits", "energy_j"):                  # util / rsv unchanged
        np.testing.assert_array_equal(p1.ctr[key], p0.ctr[key])
    assert p1.ctr["prb_used"][c].sum() == pytest.approx(p1.ctr["prb_cap"][c])          # borrowed by others


def test_min_share_coexists_with_non_work_conserving_ll_pool():
    p, c = _quota_plant(prot_min=0.5, prot_q=0.0, other_q=1e9, ll_ratio=0.3)           # no LL UEs at all
    cap = p.ctr["prb_cap"][c]
    assert p.ctr["prb_used"][c].sum() == pytest.approx(0.7 * cap)                      # dedicated LL PRBs stay idle
    assert p.ctr["prb_rsv"][c] == pytest.approx(0.3 * cap)
    p2, c2 = _quota_plant(prot_min=0.5, prot_q=1e9, other_q=1e9, ll_ratio=0.5)
    assert p2.ctr["prot_min_prb"][c2] <= p2.ctr["prb_cap"][c2] * 0.5 + 1e-9             # min share <= cap - dedicated


def test_small_protected_demand_is_served_and_rest_is_shared():
    p, c = _quota_plant(prot_min=0.5, prot_q=2e3, other_q=1e9)
    cap = p.ctr["prb_cap"][c]
    dem = p.ctr["prot_dem"][c] * cap
    assert 0 < dem < 0.5 * cap
    assert p.ctr["prot_used"][c] == pytest.approx(dem, rel=1e-9)
    assert p.ctr["prb_used"][c].sum() == pytest.approx(cap)


def test_min_share_guarantees_protected_when_demand_appears():
    p0, c = _quota_plant(prot_min=0.0, prot_q=1e9, other_q=1e9)
    p1, c1 = _quota_plant(prot_min=0.5, prot_q=1e9, other_q=1e9)
    assert c == c1
    cap = p1.ctr["prb_cap"][c]
    assert p1.ctr["prot_used"][c] >= 0.5 * cap - 1e-9
    assert p1.ctr["prot_used"][c] > p0.ctr["prot_used"][c] + 1.0
    assert p1.ctr["prb_used"][c].sum() == pytest.approx(cap)


def test_protected_floor_accounting_only_in_scored_seconds():
    p = sim.Plant(_cfg(seed=3, load="high", warmup_s=3, scored_s=5, e6p=PROT))
    _ticks(p, 8 * C.TICKS_PER_CONTROL)
    assert p.sla["prot_ue_s"] == 5 * p.prot.sum()
    assert 0 <= p.sla["prot_viol"] <= p.sla["prot_ue_s"]
    assert np.all(p.sec_viol_prot <= p.prot)
    assert p.prot.any() and np.all(p.sl[p.prot] == sim.EMBB)


# ------------------------------------------------------------------------------------------------ (c) carrier / sleep
def test_carrier_off_halves_macro_capacity_and_reactivation_waits():
    p = sim.Plant(_cfg(seed=2))
    c = int(np.nonzero(p.lay.is_macro)[0][0])
    p.tick()
    assert p.ctr["prb_cap"][c] == pytest.approx(C.N_PRB)
    knob_set(p, ("carrier", c), 1, p.t * C.TICK_S)
    p.ctr = p._new_counters()
    p.tick()
    assert p.ctr["prb_cap"][c] == pytest.approx(C.N_PRB / 2)
    knob_set(p, ("carrier", c), 2, p.t * C.TICK_S)
    p.ctr = p._new_counters()
    _ticks(p, int(C.CARRIER_ON_S / C.TICK_S) - 1)
    assert p.ctr["prb_cap"][c] == pytest.approx(C.N_PRB / 2 * p.ctr["ticks"])          # still reactivating
    _ticks(p, 3)
    p.ctr = p._new_counters()
    p.tick()
    assert p.ctr["prb_cap"][c] == pytest.approx(C.N_PRB)


def _sleep_plant(ho, seed=3):
    p = sim.Plant(_cfg(seed=seed, e6p=dataclasses.replace(BOTH, pico_sleep_ho=ho)))
    _ticks(p, 100)
    picos = np.nonzero(~p.lay.is_macro)[0]
    c = int(picos[np.argmax(np.bincount(p.serv, minlength=p.nc)[picos])])
    ues = np.nonzero((p.serv == c) & (p.int_until <= p.t * C.TICK_S))[0]
    assert len(ues) >= 2
    rlf = []
    orig = p._rlf
    p._rlf = lambda u, now, ho_fail=False: (rlf.append(int(u)), orig(u, now, ho_fail))
    return p, c, ues, rlf


def test_graceful_pico_sleep_hands_ues_over_without_rlf():
    p, c, ues, rlf = _sleep_plant(ho=True)
    now = p.t * C.TICK_S
    knob_set(p, ("sleep", c), 1.0, now)
    assert not np.any(p.serv[ues] == c)
    g = p._gains()
    ok = [u for u in ues if u not in rlf]                                  # target SINR above HO failure at sleep
    assert ok and np.allclose(p.int_until[ok], now + C.HO_EXEC_S)
    _ticks(p, int(2.0 / C.TICK_S))
    assert not set(ok) & set(rlf)
    p.ctr = p._new_counters()
    p.tick()
    assert p.ctr["prb_cap"][c] == 0.0 and p.ctr["energy_j"][c] == pytest.approx(C.PICO_SLEEP_W * C.TICK_S)
    assert g[:, c].max() <= -299
    # without graceful sleep (E6-scn-v1 behaviour) the same UEs fall through T310 into RLF
    q, c2, ues2, rlf2 = _sleep_plant(ho=False)
    knob_set(q, ("sleep", c2), 1.0, q.t * C.TICK_S)
    _ticks(q, int(2.0 / C.TICK_S))
    assert set(ues2) & set(rlf2)


def test_pico_wake_restores_capacity_after_wake_delay():
    p, c, _, _ = _sleep_plant(ho=True)
    knob_set(p, ("sleep", c), 1.0, p.t * C.TICK_S)
    _ticks(p, 5)
    knob_set(p, ("sleep", c), 0.0, p.t * C.TICK_S)
    p.ctr = p._new_counters()
    _ticks(p, int(C.PICO_WAKE_S / C.TICK_S) - 1)
    assert p.ctr["prb_cap"][c] == 0.0
    _ticks(p, 3)
    p.ctr = p._new_counters()
    p.tick()
    assert p.ctr["prb_cap"][c] == pytest.approx(C.N_PRB)


# ------------------------------------------------------------------------------------------------ (d) xApps
def test_es_pico_keeps_v1_macro_behaviour_and_sleeps_light_picos():
    cfg = _cfg(seed=3, mix="ES", load_factor=0.5, warmup_s=10, scored_s=90, e6p=BOTH)
    env = E6Env(cfg, log=False)
    x = env.xapps[0]
    assert type(x) is ESPico and x.name == "ES"
    v1 = ES(env, 0)
    assert (v1.scale, v1.behaviour, v1.next_t) == (x.scale, x.behaviour, x.next_t)     # same variant draw
    orig, n_cmp = x.propose, []

    def spy(now):                     # v1 ES on the identical state must propose the identical macro requests
        v1.u = None if x.u is None else x.u.copy()
        v1.low_since, v1.reports = dict(x.low_since), dict(x.reports)
        v1.pending_target, v1.hold_until = dict(x.pending_target), x.hold_until
        ref = v1.propose(now)
        out = orig(now)
        assert [(r["knob"], r["prop"]) for r in out if r["knob"][0] == "carrier"] ==             [(r["knob"], r["prop"]) for r in ref]
        n_cmp.append(len(ref))
        return out

    x.propose = spy
    for _ in range(90):
        env.step(None)
    assert len(n_cmp) >= 8 and sum(n_cmp) > 0                                          # macro part exercised
    assert env.plant.asleep.any()                                                      # light picos went to sleep


def test_power_es_alone_saves_energy_vs_freeze():
    cfg = _cfg(seed=3, load_factor=0.5, e6p=dataclasses.replace(PTX, xapps=("PowerES",)))
    fr, on = _run(cfg, _freeze, 120), _run(cfg, None, 120)
    assert on.stats["acc"] > 0 and np.all(on.plant.ptx_off <= 0) and on.plant.ptx_off.min() <= -3
    assert on.score()["energy_kwh"] < fr.score()["energy_kwh"]


def test_coverage_rules_on_synthetic_reports():
    env = E6Env(_cfg(seed=4, e6p=dataclasses.replace(BOTH, xapps=("Coverage",))), log=False)
    x = env.xapps[0]
    nc = env.plant.nc
    c = int(np.nonzero(env.plant.lay.is_macro)[0][0])
    rep = {"gran": "fast", "edge_sinr_p": np.full(nc, 0.0), "prot_below_frac": np.full(nc, np.nan)}
    x.hist = [rep]
    assert x.propose(100.0) == []                                                      # healthy network: nothing
    rep["edge_sinr_p"] = np.full(nc, -8.0)
    assert x.propose(110.0) == []                                                      # uniformly weak: no raise
    rep["edge_sinr_p"] = np.full(nc, 0.0)
    rep["edge_sinr_p"][c] = -8.0
    r = x.propose(120.0)
    assert [(q["knob"], q["prop"]) for q in r] == [(("ptx", c), 3.0)]                  # relatively weak: +3 dB
    rep["edge_sinr_p"][c] = 0.0
    rep["prot_below_frac"][c] = 0.5 / x.scale                                          # floors failing: raise too
    assert [q["knob"] for q in x.propose(130.0)] == [("ptx", c)]
    rep["prot_below_frac"][c] = np.nan
    env.plant.ptx_off[c] = 3.0
    assert [(q["knob"], q["prop"]) for q in x.propose(140.0)] == [(("ptx", c), 0.0)]   # comfortable: release
    env.plant.ptx_off[c] = -6.0
    assert x.propose(150.0) == []                                                      # never releases a negative offset


def test_coverage_weak_edge_rule_raises_edge_sinr_of_raised_cells_vs_freeze():
    """Mechanism of the weak-edge rule only: absolute threshold lifted to 0 dB and margin 2 dB so the rule fires at this
    load, protected-floor trigger off (ptx only). Not a claim about the frozen E6P_SPEC values."""
    P = dataclasses.replace(PTX, xapps=("Coverage",), cov_sinr_low_db=0.0, cov_sinr_ok_db=4.0, cov_margin_db=2.0)
    cfg = _cfg(seed=5, e6p=P)
    fr, on = _run(cfg, _freeze, 120), _run(cfg, None, 120)
    up = on.plant.ptx_off > 0
    assert up.any() and not up.all()
    e_fr = np.nanmean([r["edge_sinr_p"] for r in _fast(fr, 60)], 0)[up]
    e_on = np.nanmean([r["edge_sinr_p"] for r in _fast(on, 60)], 0)[up]
    assert e_on.mean() > e_fr.mean()


def test_slice_guarantee_alone_cuts_protected_violations_vs_freeze():
    cfg = _cfg(seed=3, load="high", e6p=dataclasses.replace(PROT, xapps=("SliceGuarantee",)))
    fr, on = _run(cfg, _freeze, 120), _run(cfg, None, 120)
    assert on.plant.prot_min.max() > 0
    assert on.score()["prot_viol"] < fr.score()["prot_viol"]


# ------------------------------------------------------------------------------------------------ (e) copy / reseed
def _state(env):
    p = env.plant
    return (dict(p.sla), p.q.copy(), p.pos.copy(), p.serv.copy(), p.ptx_off.copy(), p.prot_min.copy())


def _same(a, b):
    return a[0] == b[0] and all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:], strict=True))


def test_copy_preserves_e6p_state_for_lookahead():
    P = C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "Coverage", "SliceGuarantee"))
    env = E6Env(_cfg(seed=4, load="high", mix="ES", warmup_s=5, scored_s=40, e6p=P), log=False)
    for _ in range(12):
        env.step(None)
    env.plant.ptx_off[0], env.plant.prot_min[1] = -2.0, 0.2
    obs = env.step_propose()
    c = env.copy()
    assert _same(_state(c), _state(env)) and np.array_equal(c.plant.prot, env.plant.prot)
    assert c.plant.ptx_off is not env.plant.ptx_off
    dec = {"decisions": ["accept"] * len(obs["requests"]), "writes": []}
    env.step_apply(dec)
    c.step_apply(dec)
    for _ in range(15):
        env.step(None)
        c.step(None)
    assert _same(_state(c), _state(env)) and env.knobs == c.knobs


def test_copy_reseed_redraws_future_but_keeps_state():
    env = E6Env(_cfg(seed=6, mix="M2", warmup_s=5, scored_s=40, e6p=BOTH), log=False)
    for _ in range(10):
        env.step(None)
    obs = env.step_propose()
    s0 = _state(env)
    dec = {"decisions": ["accept"] * len(obs["requests"]), "writes": []}
    same, r1a, r1b, r2 = env.copy(), env.copy(reseed=1), env.copy(reseed=1), env.copy(reseed=2)
    for e in (same, r1a, r1b, r2):
        assert _same(_state(e), s0)                                                    # identical at the copy point
    for e in (env, same, r1a, r1b, r2):
        e.step_apply(dec)
        for _ in range(8):
            e.step(None)
    assert _same(_state(same), _state(env))                                            # default copy: same tape
    assert _same(_state(r1a), _state(r1b))                                             # same salt: same re-draw
    assert not np.array_equal(r1a.plant.q, env.plant.q) and not np.array_equal(r2.plant.q, r1a.plant.q)
    assert env.plant.rng_salt is None and r2.plant.rng_salt == 2


# ------------------------------------------------------------------------------------------------ (f) disabled / M5
def test_disabled_adds_no_knobs_sla_keys_or_kpm_fields():
    cfg = _cfg(seed=6, mix="M4", warmup_s=2, scored_s=6)
    off = E6Env(cfg, log=False)
    on = E6Env(dataclasses.replace(cfg, e6p=BOTH), log=False)
    assert not any(k[0] in ("ptx", "prot_min") for k in off.knobs)
    assert on.knobs[:len(off.knobs)] == off.knobs                                      # E6 prefix/order unchanged
    assert {k[0] for k in on.knobs[len(off.knobs):]} == {"ptx", "prot_min"}
    assert type(off.xapps[2]) is ES and type(on.xapps[2]) is ESPico
    off.run()
    assert not {"prot_viol", "prot_ue_s", "lowsinr_ue_s"} & set(off.plant.sla) and "prot_viol" not in off.score()
    fast = [r for r in off.kpm.delivered if r["gran"] == "fast"]
    assert fast and not any(k in fast[0] for k in ("ptx_db", "edge_sinr_p", "prot_below_frac"))
    assert not off.plant.prot.any() and np.all(off.plant.ptx_off == 0)


def test_enabled_at_default_knobs_matches_e6_allocation_and_energy():
    cfg = _cfg(seed=8, mix="MRO", warmup_s=5, scored_s=25)
    off, on = _run(cfg, None, 30), _run(dataclasses.replace(cfg, e6p=BOTH), None, 30)
    assert {k: v for k, v in on.plant.sla.items() if k in off.plant.sla} == off.plant.sla
    np.testing.assert_array_equal(on.plant.q, off.plant.q)
    np.testing.assert_array_equal(on.plant.rho, off.plant.rho)


def test_p_xapps_require_their_knobs():
    with pytest.raises(ValueError):
        E6Env(_cfg(e6p=C.E6PConfig(xapps=("PowerES",))), log=False)
    with pytest.raises(ValueError):
        E6Env(_cfg(e6p=C.E6PConfig(ptx_on=True, xapps=("SliceGuarantee",))), log=False)
    env = E6Env(_cfg(mix="M2", e6p=C.E6PConfig(ptx_on=True, xapps=("Coverage",))), log=False)
    assert [x.name for x in env.xapps] == ["MRO", "TS", "ES", "Coverage"]
    assert knob_get(env.plant, ("ptx", 0)) == 0.0
