"""Behavioural mechanics tests for the E6 stress scenarios (S1 surge, S2 mistune). No golden numbers, no SLA outcomes."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6 import sim
from cdd_oran.envs.e6.baselines import SMORestore, freeze
from cdd_oran.envs.e6.env import E6Env


def cfg(**kw):
    return C.E6Config(**{**dict(seed=5, load="medium", mobility="mixed", mix="none", warmup_s=0.0, scored_s=100.0), **kw})


def ticks(p, n):
    for _ in range(n):
        p.tick()


def seconds(s):
    return int(round(s / C.TICK_S))


# ------------------------------------------------------------------------------------------------------------ base
def test_base_ignores_scenario_parameters():
    """scenario='base' builds no scenario and the plant evolves identically whatever the scenario knobs say."""
    a = sim.Plant(cfg())
    b = sim.Plant(cfg(scenario="base", surge_mult=9.0, corr_frac_ue=0.5, mis_hys_db=5.0, mis_ttt_ms=640))
    assert a.scn is None and b.scn is None
    ticks(a, 100)
    ticks(b, 100)
    assert a.sla == b.sla
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.serv, b.serv)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        sim.Plant(cfg(scenario="outage"))


def test_holdout_variants_are_single_factor_config_overrides():
    fields = {f.name for f in dataclasses.fields(C.E6Config)}
    dev = C.E6Config()
    for scn, variants in C.SCENARIO_HOLDOUT.items():
        assert scn in sim.SCENARIOS
        for name, over in variants.items():
            assert set(over) <= fields, name
            assert any(getattr(dev, k) != v for k, v in over.items()), name     # differs from the DEV default
            dataclasses.replace(dev, scenario=scn, **over).validate_scenario()     # and is valid


# ------------------------------------------------------------------------------------------------------------ S1
def test_surge_multiplier_inside_region_during_window_only():
    c = cfg(scenario="surge", warmup_s=50.0, scored_s=100.0)
    p = sim.Plant(c)
    s = p.scn
    t0, ramp, t_end = s.t0, s.ramp, s.t_end
    assert t0 == pytest.approx(50.0 + c.surge_onset_frac * 100.0) and t_end < 150.0
    pos = np.vstack([s.centre(t0), s.centre(t0) + [c.surge_radius_m + 60.0, 0.0]])
    pos = p.lay.wrap(pos)
    hold = t0 + ramp + 1.0
    np.testing.assert_allclose(s.traffic_mult(hold, pos), [c.surge_mult, 1.0])
    np.testing.assert_allclose(s.traffic_mult(t0 + ramp / 2, pos), [1 + (c.surge_mult - 1) / 2, 1.0])
    for t in (0.0, t0 - 1.0, t_end + 1.0):
        assert np.all(s.traffic_mult(t, pos) == 1.0)


def test_surge_realised_offered_load_in_region(monkeypatch):
    """Realised eMBB/BE file arrivals of in-region UEs are ~surge_mult x the base arm on the same seed; out-of-region
    arrivals are unchanged in expectation."""
    orig = sim._rng
    rec = {}

    def rng(seed, stream, t=0):
        g = orig(seed, stream, t)
        if stream != "traffic":
            return g

        class Rec:
            calls = 0

            def poisson(self, lam):
                out = g.poisson(lam)
                Rec.calls += 1
                if Rec.calls == 2:                  # 1st draw = LL packets, 2nd = eMBB/BE files
                    rec["files"] = out
                return out
        return Rec()

    monkeypatch.setattr(sim, "_rng", rng)
    kw = dict(surge_onset_frac=0.0, surge_ramp_frac=0.0, surge_hold_frac=1.0, load="high")
    tot = {}
    for arm in ("base", "surge"):
        p = sim.Plant(cfg(scenario=arm, **kw))
        geo = sim.Plant(cfg(scenario="surge", **kw)).scn
        acc = np.zeros(2)
        for _ in range(seconds(60)):
            now = p.t * C.TICK_S
            p.tick()
            files = np.where(p.sl != sim.LL, rec["files"], 0)
            ins = geo.inside(now, p.pos)
            acc += [files[ins].sum(), files[~ins].sum()]
        tot[arm] = acc
    r_in = tot["surge"][0] / tot["base"][0]
    r_out = tot["surge"][1] / tot["base"][1]
    assert tot["base"][0] >= 50                                  # enough events for the ratio to mean something
    assert 0.8 * C.E6Config().surge_mult < r_in < 1.2 * C.E6Config().surge_mult
    assert 0.85 < r_out < 1.15


def test_moving_surge_drifts_at_declared_speed():
    v = C.SCENARIO_HOLDOUT["surge"]["H-move"]["surge_speed_mps"]
    p = sim.Plant(cfg(scenario="surge", surge_speed_mps=v))
    s = p.scn
    d = s._dist(s.centre(s.t0 + 20.0)[None], s.centre(s.t0))[0]
    assert s.t_end > s.t0 + 20.0
    assert d == pytest.approx(20.0 * v, rel=1e-6)
    assert np.allclose(s.centre(s.t_end + 50.0), s.centre(s.t_end))  # stops when the event ends
    assert np.allclose(s.centre(0.0), s.centre(s.t0))           # still before onset


def test_surge_location_modes():
    for seed in range(4):
        band = sim.Plant(cfg(seed=seed, scenario="surge")).scn.c0
        vert = sim.Plant(cfg(seed=seed, scenario="surge", surge_loc="vertex")).scn.c0
        lay = sim.Plant(cfg(seed=seed)).lay
        db = sim.wrap_dist(lay, lay.sites, band).min()
        dv = np.sort(sim.wrap_dist(lay, lay.sites, vert))
        assert 150.0 - 1e-6 <= db <= 250.0 + 1e-6
        np.testing.assert_allclose(dv[:3], C.ISD_M / np.sqrt(3.0), rtol=1e-6)   # equidistant from three sites


# ------------------------------------------------------------------------------------------------------------ S2
def _on_segment(p):
    s = p.scn
    rel = p.pos[s.idx][:, None, :] - (s.centre + p.lay._cands())[None]          # all images of the road centre
    along = rel @ s.u
    perp = np.abs(rel @ np.array([-s.u[1], s.u[0]]))
    ok = (perp < 1e-6) & (np.abs(along) <= s.half + 1e-6)
    return ok.any(1)


def test_mistune_corridor_vehicles_from_t0():
    c = cfg(scenario="mistune")
    p = sim.Plant(c)
    s = p.scn
    assert len(s.idx) == round(c.corr_frac_ue * p.n)
    np.testing.assert_allclose(p.speed[s.idx], c.corr_speed_kmh / 3.6)
    assert not p.indoor[s.idx].any()
    assert _on_segment(p).all()
    assert not np.isin(s.idx, p.mobile_idx).any()                  # driven by the road, not the random walk
    s0 = s.s.copy()
    ticks(p, seconds(5))
    assert _on_segment(p).all()
    moved = np.abs(s.s - s0)
    far = np.abs(s0) < s.half - 5 * s.v                           # no U-turn possible within 5 s
    np.testing.assert_allclose(moved[far], 5 * s.v, rtol=1e-6)


def test_mistune_rollout_sets_cluster_parameters_at_onset_only():
    c = cfg(scenario="mistune", warmup_s=2.0, scored_s=20.0, mis_onset_frac=0.2)   # onset at t = 6 s
    p = sim.Plant(c)
    s = p.scn
    hys0, ttt0 = p.hys.copy(), p.ttt.copy()
    assert len(s.cluster) >= 2
    ticks(p, seconds(5.9))
    np.testing.assert_array_equal(p.hys, hys0)
    np.testing.assert_array_equal(p.ttt, ttt0)
    ticks(p, seconds(0.2))
    out = np.setdiff1d(np.arange(p.nc), s.cluster)
    assert np.all(p.hys[s.cluster] == c.mis_hys_db) and np.all(p.ttt[s.cluster] == c.mis_ttt_ms)
    np.testing.assert_array_equal(p.hys[out], hys0[out])
    np.testing.assert_array_equal(p.ttt[out], ttt0[out])
    p.hys[s.cluster] = 2.0                                          # a later RIC write is not overwritten again
    ticks(p, 5)
    assert np.all(p.hys[s.cluster] == 2.0)


def test_mistune_cluster_covers_the_corridor():
    p = sim.Plant(cfg(scenario="mistune"))
    s = p.scn
    pts = p.lay.wrap(s.centre + np.linspace(-s.half, s.half, 41)[:, None] * s.u)
    best = np.argmax(p.gm.lookup(pts), 1)
    assert np.isin(best, s.cluster).all()


# ------------------------------------------------------------------------------------------------------------ tape
@pytest.mark.parametrize("scn", ["surge", "mistune"])
def test_scenario_tape_is_action_independent(scn):
    """Two arms that configure the network differently see the same scenario tape (positions, surge multiplier)."""
    c = cfg(scenario=scn, surge_onset_frac=0.0, surge_ramp_frac=0.0)
    a, b = sim.Plant(c), sim.Plant(c)
    b.cio[:, :] = 3.0
    b.ll_ratio[:] = 0.3
    for _ in range(seconds(4)):
        a.tick()
        b.tick()
    np.testing.assert_array_equal(a.pos, b.pos)
    now = a.t * C.TICK_S
    np.testing.assert_array_equal(a.scn.traffic_mult(now, a.pos), b.scn.traffic_mult(now, b.pos))


# ------------------------------------------------------------------------------------------------------------ validity
@pytest.mark.parametrize("scn,bad", [
    ("surge", dict(surge_radius_m=-1.0)), ("surge", dict(surge_radius_m=0.0)), ("surge", dict(surge_speed_mps=-0.1)),
    ("surge", dict(surge_mult=0.5)), ("surge", dict(surge_onset_frac=1.2)), ("surge", dict(surge_ramp_frac=-0.1)),
    ("surge", dict(surge_onset_frac=0.5, surge_ramp_frac=0.2, surge_hold_frac=0.3)),     # ends after scoring
    ("surge", dict(surge_loc="nowhere")), ("surge", dict(scored_s=0.0)),
    ("mistune", dict(corr_frac_ue=0.0)), ("mistune", dict(corr_frac_ue=1.5)), ("mistune", dict(corr_frac_ue=0.001)),
    ("mistune", dict(corr_len_m=0.0)), ("mistune", dict(corr_len_m=1e5)), ("mistune", dict(corr_speed_kmh=0.0)),
    ("mistune", dict(corr_loc="nowhere")), ("mistune", dict(mis_onset_frac=1.0)),
    ("mistune", dict(mis_onset_frac=-0.1)), ("mistune", dict(mis_hys_db=5.5)), ("mistune", dict(mis_hys_db=2.25)),
    ("mistune", dict(mis_hys_db=-0.5)), ("mistune", dict(mis_ttt_ms=500)), ("mistune", dict(mis_ttt_ms=1280)),
])
def test_invalid_scenario_parameters_rejected(scn, bad):
    c = cfg(scenario=scn, **bad)
    with pytest.raises(ValueError):
        c.validate_scenario()
    with pytest.raises(ValueError):
        sim.make_scenario(c, None)                      # the builder validates before touching the plant


def test_invalid_parameters_rejected_by_plant_constructor():
    with pytest.raises(ValueError):
        sim.Plant(cfg(scenario="surge", surge_radius_m=-5.0))


def test_default_length_event_timing():
    d = C.E6Config()                                    # default episode: warm-up 300 s, scored 1800 s
    end = d.warmup_s + d.scored_s
    s = sim.Plant(dataclasses.replace(d, scenario="surge")).scn
    assert s.t0 == pytest.approx(d.warmup_s + d.surge_onset_frac * d.scored_s)
    assert s.t_end == pytest.approx(s.t0 + (2 * d.surge_ramp_frac + d.surge_hold_frac) * d.scored_s)
    assert d.warmup_s < s.t0 < s.t_end <= end
    m = sim.Plant(dataclasses.replace(d, scenario="mistune")).scn
    assert m.t_mis == pytest.approx(d.warmup_s + d.mis_onset_frac * d.scored_s)
    assert d.warmup_s < m.t_mis < end


# ------------------------------------------------------------------------------------------ S2 OAM push vs RIC history
def run_to(env, t, arbiter=None):
    while env.sec < t:
        env.step(arbiter)


def rollback(env, knobs):
    obs = env.step_propose()
    env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": list(knobs)})


def test_oam_push_once_persists_and_is_not_rollbackable():
    """Default scored length (1800 s) with a short warm-up so the rollout (t = 10 + 180 s) is crossed quickly."""
    c = cfg(scenario="mistune", warmup_s=10.0, scored_s=1800.0)
    env = E6Env(c, log=False)
    scn, p = env.plant.scn, env.plant
    c0 = int(scn.cluster[0])
    run_to(env, 50)
    assert env._apply(("hys", c0), 2.5, float(env.sec))[0] == "ok"      # a pre-rollout RIC change on the cluster
    assert env.prev_val[("hys", c0)] == 2.0
    run_to(env, int(np.ceil(scn.t_mis)) + 1)
    assert scn.push_t == pytest.approx(scn.t_mis, abs=C.TICK_S + 1e-9)
    assert [tag for _, tag in scn.oam_log] == ["push"]
    assert np.all(p.hys[scn.cluster] == c.mis_hys_db) and np.all(p.ttt[scn.cluster] == c.mis_ttt_ms)
    assert scn.pre_hys[0] == 2.5                                          # the SMO knows what it overwrote
    for x in scn.cluster:
        assert env.prev_val[("hys", int(x))] == c.mis_hys_db
        assert env.prev_val[("ttt", int(x))] == c.mis_ttt_ms
    # a RIC rollback cannot revert the OAM push (last-known-good = the pushed value)
    rollback(env, [("hys", c0)])
    assert p.hys[c0] == c.mis_hys_db and env.stats["rollbacks"] == 0
    # the first post-push RIC write works (the OAM push sets no dwell) and records the OAM value as its prior
    now = float(env.sec)
    assert env._apply(("ttt", c0), 320, now)[0] == "ok" and p.ttt[c0] == 320
    assert env.prev_val[("ttt", c0)] == c.mis_ttt_ms
    # a RIC action on another knob leaves the push in place; the push is never re-applied
    nb = p.lay.neighbours[c0][0]
    assert env._apply(("cio", c0, nb), 1.0, now)[0] == "ok"
    run_to(env, env.sec + 11)
    assert np.all(p.hys[scn.cluster] == c.mis_hys_db)
    assert [tag for _, tag in scn.oam_log] == ["push"]
    rollback(env, [("ttt", c0)])                                          # rollback of the RIC change -> OAM value
    assert p.ttt[c0] == c.mis_ttt_ms and env.stats["rollbacks"] == 1


def test_tape_is_arbiter_independent_through_the_rollout():
    c = cfg(scenario="mistune", mix="M4", warmup_s=10.0, scored_s=1800.0)
    a, b = E6Env(c, log=False), E6Env(c, log=False)
    t = int(np.ceil(a.plant.scn.t_mis)) + 5
    run_to(a, t)                  # no arbitration: the xApps change the network
    run_to(b, t, freeze)          # freeze: nothing changes except the OAM push
    assert a.stats["changes"] > 0 and b.stats["changes"] == 0
    assert a.plant.scn.push_t is not None and a.plant.scn.push_t == b.plant.scn.push_t
    np.testing.assert_array_equal(a.plant.pos, b.plant.pos)
    np.testing.assert_array_equal(a.plant.scn.s, b.plant.scn.s)
    np.testing.assert_array_equal(a.plant.scn.cluster, b.plant.scn.cluster)
    assert np.all(b.plant.hys[b.plant.scn.cluster] == c.mis_hys_db)
    assert np.all(b.plant.ttt[b.plant.scn.cluster] == c.mis_ttt_ms)


# ------------------------------------------------------------------------------------------------------ SMO restore
def mob_report(p, t0, tl, att):
    nc = p.nc
    rep = {"gran": "mob", "t0": t0, "t1": t0 + 30}
    for k in ("ho_att", "ho_succ", "too_late", "too_early", "wrong_cell", "pingpong"):
        rep[k] = np.zeros((nc, nc))
    c0 = int(p.scn.cluster[0])
    c1 = int(p.lay.neighbours[c0][0])
    rep["too_late"][c0, c1], rep["ho_att"][c0, c1] = tl, att
    rep["rlf"] = np.zeros(nc)
    return rep


def test_smo_alarm_rule():
    env = E6Env(cfg(scenario="mistune", warmup_s=2.0, scored_s=100.0), log=False)
    p = env.plant
    bad, good = (5, 100), (1, 100)                        # 5/105 = 4.8 % > 2 %; 1 too-late is below min evidence
    smo = SMORestore(env)
    assert not smo.alarm(mob_report(p, 0, *bad), p.scn)
    assert smo.alarm(mob_report(p, 30, *bad), p.scn)      # two consecutive -> fire
    smo = SMORestore(env)
    fired = [smo.alarm(mob_report(p, 0, *r), p.scn) for r in (bad, good, bad)]
    assert fired == [False, False, False]                  # streak broken
    smo = SMORestore(env)
    fired = [smo.alarm(mob_report(p, 0, 2, 200), p.scn) for _ in range(3)]
    assert fired == [False, False, False]                  # 2/202 < 2 %


def test_smo_restore_fires_restores_and_is_not_rollbackable():
    c = cfg(scenario="mistune", warmup_s=10.0, scored_s=600.0)        # rollout at 10 + 60 s
    env = E6Env(c, log=False)
    scn, p = env.plant.scn, env.plant
    smo = SMORestore(env)
    run_to(env, int(np.ceil(scn.t_mis)) + 1, smo)
    assert smo.fired_t is None and np.all(p.hys[scn.cluster] == c.mis_hys_db)
    pre_h, pre_t = scn.pre_hys.copy(), scn.pre_ttt.copy()
    # two consecutive post-push alarm reports reach the SMO (report windows start after the push)
    obs = env.step_propose()
    t0 = float(np.ceil(scn.push_t))
    obs["new_reports"] = [mob_report(p, t0, 5, 100), mob_report(p, t0 + 30, 5, 100)]
    env.step_apply(smo(obs))
    assert smo.fired_t == float(env.sec)
    assert [tag for _, tag in scn.oam_log] == ["push", "restore"]
    np.testing.assert_array_equal(p.hys[scn.cluster], pre_h)
    np.testing.assert_array_equal(p.ttt[scn.cluster], pre_t)
    c0 = int(scn.cluster[0])
    rollback(env, [("hys", c0), ("ttt", c0)])                          # the RIC cannot undo the SMO restore either
    assert p.hys[c0] == pre_h[0] and p.ttt[c0] == pre_t[0]
    run_to(env, env.sec + 5, smo)                                       # fires once only
    assert [tag for _, tag in scn.oam_log] == ["push", "restore"]


def test_env_copy_rebinds_the_oam_hook():
    env = E6Env(cfg(scenario="mistune", warmup_s=2.0, scored_s=100.0), log=False)
    c = env.copy()
    assert c.plant.oam_hook.__self__ is c and env.plant.oam_hook.__self__ is env
