"""Mechanical tests for the E6 v2 xApp stack (``cdd_oran.envs.e6.xapps_v2``).

The xApps are driven with SYNTHETIC KPM reports on a small fake plant whose static ``Layout.neighbours`` list is EMPTY
(the v2 stack must work from its run-time NRT alone): each test checks that a mechanism reacts as specified
(docs/benchmark/E6_V2_XAPPS_SPEC.md), respects its bounds, and proposes values the RIC can apply. The env tests check
integration mechanics only (bit-identity of v1 under KPMV2, V2 mixes build and run, NRT coverage of HO / RLF pairs).
No SLA / SVR / own-KPI outcome is computed or compared anywhere in this file.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6 import xapps as X1
from cdd_oran.envs.e6 import xapps_v2 as X2
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import KPM, feasible, knob_get, knob_set

TGT = C.EMBB_THP_TARGET_BPS
LOW, HIGH = 0.5 * TGT, 3.0 * TGT


# ------------------------------------------------------------------------------------------------ fakes / helpers
def fake_env(nc=3, seed=0):
    lay = SimpleNamespace(n_cells=nc, neighbours=[[] for _ in range(nc)], is_macro=np.ones(nc, bool))
    p = SimpleNamespace(nc=nc, lay=lay, cio=np.zeros((nc, nc)), hys=np.full(nc, 2.0), ttt=np.full(nc, 320),
                        ll_ratio=np.zeros(nc), asleep=np.zeros(nc, bool), n_car=np.full(nc, 2), n_trx=np.full(nc, 2))
    return SimpleNamespace(cfg=SimpleNamespace(seed=seed, xapp_threshold_scale=1.0), plant=p)


def mlb(env, hold=1):
    x = X2.MLB(env, 1)
    x.scale, x.behaviour = 1.0, "retry"
    x.gamma, x.ho_guard = X2.MLB.GAMMA, X2.MLB.HO_GUARD
    x.HOLD_N = hold
    return x


def mro(env):
    x = X2.MROv2(env, 0)
    x.scale, x.behaviour = 1.0, "retry"
    x.tl_thr, x.te_thr = X2.MROv2.TL_THR, X2.MROv2.TE_THR
    return x


def apply(x, reqs, now, last_change=None):
    """RIC stand-in: accept everything through the hard actuator limits; ACK the xApp. Returns applied values."""
    lc = {} if last_change is None else last_change
    out = {}
    for r in reqs:
        v, ok = feasible(x.p, r["knob"], r["prop"], now, lc)
        if ok and abs(v - knob_get(x.p, r["knob"])) > 1e-9:
            knob_set(x.p, r["knob"], v, now)
            lc[r["knob"]] = now
        x.result(r, ok, v, now)
        out[r["knob"]] = v
    return out


def feed_mlb(x, t, act, p5, border_act, mob=None, carriers=None, se_src=1.0, se_tgt=1.0):
    """10 fast reports [t-10, t), two thp reports [t-10, t), one mr report [t-5, t), optional mob report."""
    nc = x.p.nc
    a = np.zeros((nc, 3))
    a[:, 1] = act                                       # eMBB active UEs
    car = x.p.n_car if carriers is None else np.asarray(carriers)
    reps = [{"gran": "fast", "t0": t - 10 + i, "t1": t - 9 + i, "act_ue": a, "carriers": car} for i in range(10)]
    reps += [{"gran": "thp", "t0": t - 10 + 5 * i, "t1": t - 5 + 5 * i, "embb_thp_p5": np.asarray(p5, float)}
             for i in range(2)]
    b = np.zeros((nc, nc, X2.MR_BINS_DB))
    for (s, n), v in border_act.items():
        b[s, n] = np.cumsum(list(v) + [0] * (X2.MR_BINS_DB - len(v)))
    reps.append({"gran": "mr", "t0": t - 5, "t1": t, "conn": np.full(nc, 10.0), "border_all": b, "border_act": b,
                 "se_src": np.where(b > 0, se_src, np.nan), "se_tgt": np.where(b > 0, se_tgt, np.nan)})
    if mob is not None:
        reps.append(mob)
    x.observe(reps)


def mob_rep(nc, t0, **entries):
    rep = {k: np.zeros((nc, nc)) for k in ("ho_att", "ho_succ", "too_late", "too_early", "wrong_cell", "pingpong")}
    rep.update(gran="mob", t0=t0, t1=t0 + 30, rlf=np.zeros(nc))
    for k, pairs in entries.items():
        for (s, n), v in pairs.items():
            rep[k][s, n] = v
    return rep


def mobq_rep(nc, t0, dwell):
    d = np.zeros((nc, nc))
    for (s, n), v in dwell.items():
        d[s, n] = v
    return {"gran": "mobq", "t0": t0, "t1": t0 + 30, "lowq_dwell": d}


def knobs(reqs):
    return {r["knob"]: r["prop"] for r in reqs}


BUSY = dict(act=[10, 1, 10], p5=[LOW, HIGH, LOW])


# ================================================================================================= NRT
def test_nrt_builds_symmetric_relations_from_ho_rlf_and_measurements_only():
    nrt = X2.NRT(4)
    assert nrt.neighbours(0) == []
    nrt.update([mob_rep(4, 0, ho_att={(0, 1): 1}, too_late={(2, 3): 1})])
    b = np.zeros((4, 4, X2.MR_BINS_DB))
    b[0, 3, -1] = 1
    nrt.update([{"gran": "mr", "border_all": b}, {"gran": "fast"}])
    assert nrt.neighbours(0) == [1, 3] and nrt.neighbours(1) == [0] and nrt.neighbours(2) == [3]
    assert nrt.neighbours(3) == [0, 2]
    assert not nrt.rel.diagonal().any()


# ================================================================================================= MLB
def test_mlb_steers_toward_spare_nrt_neighbour_with_one_antisymmetric_pair_decision():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    k = knobs(x.propose(100.0))
    assert k == {("cio", 0, 1): 1.0, ("cio", 1, 0): -1.0}     # cell 2 has no spare neighbour left (1 is busy)


def test_mlb_push_needs_hold_time():
    x = mlb(fake_env(), hold=X2.MLB.HOLD_N)
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    assert x.propose(100.0) == []
    feed_mlb(x, 110, border_act={(0, 1): [2]}, act=[10, 1, 10], p5=[HIGH, HIGH, LOW])   # condition breaks
    assert x.propose(110.0) == []
    feed_mlb(x, 120, border_act={(0, 1): [2]}, **BUSY)
    assert x.propose(120.0) == []
    feed_mlb(x, 130, border_act={(0, 1): [2]}, **BUSY)
    assert ("cio", 0, 1) in knobs(x.propose(130.0))


def test_mlb_no_step_without_border_ues():
    x = mlb(fake_env())
    x.nrt.rel[:] = ~np.eye(3, dtype=bool)
    feed_mlb(x, 100, border_act={}, **BUSY)
    assert x.propose(100.0) == []


def test_mlb_requires_moved_ue_rate_gain():
    x = mlb(fake_env())                                   # target SINR penalty kills the gain: 0.3 x SE at target
    feed_mlb(x, 100, border_act={(0, 1): [2]}, se_src=2.0, se_tgt=0.6, **BUSY)
    assert x.propose(100.0) == []
    y = mlb(fake_env())
    feed_mlb(y, 100, border_act={(0, 1): [2]}, se_src=2.0, se_tgt=1.5, **BUSY)
    assert ("cio", 0, 1) in knobs(y.propose(100.0))


def test_mlb_uses_minimal_step_that_moves_an_active_ue_and_caps_at_2db():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [0, 1]}, **BUSY)
    assert knobs(x.propose(100.0))[("cio", 0, 1)] == 2.0
    y = mlb(fake_env())
    feed_mlb(y, 100, border_act={(0, 1): [0, 0, 3]}, **BUSY)     # needs 3 dB
    assert y.propose(100.0) == []


def test_mlb_requires_genuinely_spare_neighbour():
    x = mlb(fake_env())                                   # neighbour suffering -> never a target
    feed_mlb(x, 100, act=[10, 1, 10], p5=[LOW, LOW, LOW], border_act={(0, 1): [2]})
    assert x.propose(100.0) == []
    y = mlb(fake_env())                                   # neighbour not less loaded by the gain margin
    feed_mlb(y, 100, act=[10, 9, 10], p5=[LOW, HIGH, LOW], border_act={(0, 1): [2]})
    assert y.propose(100.0) == []
    e = fake_env()
    e.plant.asleep[1] = True                              # no capacity
    z = mlb(e)
    feed_mlb(z, 100, act=[10, 0, 10], p5=[LOW, np.nan, LOW], border_act={(0, 1): [2]})
    assert z.propose(100.0) == []


def test_mlb_capacity_share_counts_carriers_and_ll_reservation():
    e = fake_env()
    e.plant.ll_ratio[:] = [0.0, 0.5, 0.0]
    kappa = X2._cap_share(e.plant, [2, 1, 2])
    np.testing.assert_allclose(kappa, [1.0, 0.25, 1.0])


def test_mlb_window_mean_rate_not_last_sample():
    x = mlb(fake_env())
    x.observe([{"gran": "thp", "t0": 85, "t1": 90, "embb_thp_p5": np.array([0.2 * TGT, HIGH, HIGH])}])
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    x.thp_hist = [x.thp_hist[-1], {"t0": 95, "embb_thp_p5": np.array([1.9 * TGT, HIGH, HIGH])}]
    _, _, suffer = x._state()
    assert not suffer[0]                                  # mean of 0.5 T and 1.9 T = 1.2 T >= T
    x.thp_hist[0] = {"t0": 90, "embb_thp_p5": np.array([np.nan, np.nan, HIGH])}
    _, _, suffer = x._state()
    assert not suffer[0] and not suffer[1]                # one NaN ignored; all-NaN = not suffering


def test_mlb_no_overshoot():
    x = mlb(fake_env())                                   # moving 3 UEs from 4 to 1 would invert the imbalance
    feed_mlb(x, 100, act=[4, 1, 0], p5=[LOW, HIGH, HIGH], border_act={(0, 1): [3]})
    assert ("cio", 0, 1) not in knobs(x.propose(100.0))


def test_mlb_respects_offset_band_and_never_pushes_against_own_reverse_push():
    e = fake_env()
    e.plant.cio[0, 1] = X2.MLB.CIO_MAX
    x = mlb(e)
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    assert ("cio", 0, 1) not in knobs(x.propose(100.0))
    y = mlb(fake_env())
    y.own[(1, 0)] = [{"k": 1.0, "t": 0.0, "n_ok": False}]   # n_ok False: no degradation rollback here
    feed_mlb(y, 100, border_act={(0, 1): [2]}, **BUSY)
    assert ("cio", 0, 1) not in knobs(y.propose(100.0))


def test_mlb_ho_guard_blocks_step():
    x = mlb(fake_env())
    mob = mob_rep(3, 60, ho_att={(0, 1): 10}, pingpong={(0, 1): 5})
    feed_mlb(x, 100, border_act={(0, 1): [2]}, mob=mob, **BUSY)
    assert ("cio", 0, 1) not in knobs(x.propose(100.0))


def test_mlb_waits_for_post_change_evidence():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    apply(x, x.propose(100.0), 100.0)
    assert x.p.cio[0, 1] == 1.0 and x.p.cio[1, 0] == -1.0
    assert all(r["knob"][1] != 0 for r in x.propose(110.0))       # same (stale) evidence: no new step from cell 0
    feed_mlb(x, 111, border_act={(0, 1): [2]}, **BUSY)
    assert knobs(x.propose(111.0)).get(("cio", 0, 1)) == 2.0


def test_mlb_rolls_back_own_step_when_neighbour_degrades_and_blocks_pair():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    apply(x, x.propose(100.0), 100.0)
    feed_mlb(x, 111, act=[10, 3, 10], p5=[LOW, LOW, LOW], border_act={(0, 1): [2]})
    k = knobs(x.propose(111.0))
    assert k == {("cio", 0, 1): 0.0, ("cio", 1, 0): 0.0}
    apply(x, x.propose(111.0), 111.0, last_change={})
    assert x.own[(0, 1)] == []
    feed_mlb(x, 130, border_act={(0, 1): [2]}, **BUSY)
    assert ("cio", 0, 1) not in knobs(x.propose(130.0))            # blocked for BLOCK_S
    t = 111 + X2.MLB.BLOCK_S + 5
    feed_mlb(x, t, border_act={(0, 1): [2]}, **BUSY)
    assert ("cio", 0, 1) in knobs(x.propose(float(t)))


def test_mlb_verifies_moved_ues_and_uses_post_change_ho_evidence_only():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    apply(x, x.propose(100.0), 100.0)
    stale = mob_rep(3, 60, ho_att={(0, 1): 10}, too_early={(0, 1): 5})
    feed_mlb(x, 112, border_act={}, mob=stale, **BUSY)
    assert knobs(x.propose(112.0)).get(("cio", 0, 1)) != 0.0       # pre-change HO evidence does not roll back
    moved = mob_rep(3, 120, ho_att={(0, 1): 4})
    feed_mlb(x, 151, border_act={}, mob=moved, **BUSY)
    assert knobs(x.propose(151.0)).get(("cio", 0, 1)) != 0.0       # UEs moved, guard fine: keep the push
    bad = mob_rep(3, 120, ho_att={(0, 1): 10}, too_early={(0, 1): 5})
    feed_mlb(x, 152, border_act={}, mob=bad, **BUSY)
    assert knobs(x.propose(152.0)).get(("cio", 0, 1)) == 0.0       # HO guard trips after the push
    y = mlb(fake_env())
    feed_mlb(y, 100, border_act={(0, 1): [2]}, **BUSY)
    apply(y, y.propose(100.0), 100.0)
    feed_mlb(y, 151, border_act={}, mob=mob_rep(3, 120, ho_att={(1, 2): 3}), **BUSY)
    assert knobs(y.propose(151.0)).get(("cio", 0, 1)) == 0.0       # nobody moved 0 -> 1: release the useless push


def test_mlb_releases_own_offset_after_recovery():
    x = mlb(fake_env())
    feed_mlb(x, 100, border_act={(0, 1): [2]}, **BUSY)
    apply(x, x.propose(100.0), 100.0)
    feed_mlb(x, 120, act=[3, 3, 3], p5=[HIGH, HIGH, HIGH], border_act={})
    assert x.propose(120.0) == []
    feed_mlb(x, 185, act=[3, 3, 3], p5=[HIGH, HIGH, HIGH], border_act={})
    k = knobs(x.propose(185.0))
    assert k == {("cio", 0, 1): 0.0, ("cio", 1, 0): 0.0}
    apply(x, x.propose(185.0), 185.0, last_change={})
    assert x.own[(0, 1)] == [] and x.p.cio[0, 1] == 0.0


def test_mlb_proposals_are_bounded_and_feasible_on_random_inputs():
    rng = np.random.default_rng(0)
    n_req = 0
    for trial in range(200):
        e = fake_env(nc=4, seed=trial)
        e.plant.cio[:] = rng.integers(-6, 7, (4, 4))
        e.plant.n_car[:] = rng.integers(1, 3, 4)
        e.plant.ll_ratio[:] = rng.choice([0.0, 0.1, 0.3], 4)
        x = mlb(e)
        border = {(s, n): list(rng.integers(0, 3, 3)) for s in range(4) for n in range(4) if s != n}
        feed_mlb(x, 100, act=rng.uniform(0, 12, 4), p5=rng.choice([LOW, HIGH, np.nan], 4), border_act=border,
                 se_tgt=float(rng.uniform(0.5, 2.0)))
        before = e.plant.cio.copy()
        reqs = x.propose(100.0)
        n_req += len(reqs)
        for r in reqs:
            _, s, n = r["knob"]
            assert C.CIO_RANGE[0] <= r["prop"] <= C.CIO_RANGE[1]
            assert abs(r["prop"] - before[s, n]) <= X2.MLB.MAX_STEP + 1e-9
            v, ok = feasible(e.plant, r["knob"], r["prop"], 100.0, {})
            assert ok and v == pytest.approx(r["prop"])
        fw = [r for r in reqs if r["prop"] > before[r["knob"][1], r["knob"][2]]]
        for r in fw:                                                   # an outward push stays inside its band ...
            assert r["prop"] <= X2.MLB.CIO_MAX + 1e-9
            _, s, n = r["knob"]                                        # ... and is paired with the reverse write
            assert any(q["knob"] == ("cio", n, s) for q in reqs) or before[n, s] <= C.CIO_RANGE[0]
    assert n_req > 0


# ================================================================================================= MRO v2
def two_windows(x, t0, dwell=None, **entries):
    nc = x.p.nc
    reps = [mob_rep(nc, t0, **entries), mob_rep(nc, t0 + 30, **entries)]
    if dwell is not None:
        reps += [mobq_rep(nc, t0, dwell), mobq_rep(nc, t0 + 30, dwell)]
    x.observe(reps)


LATE2 = dict(ho_att={(0, 1): 20, (0, 2): 20}, too_late={(0, 1): 2, (0, 2): 2})
EARLY2 = dict(ho_att={(0, 1): 20, (0, 2): 20}, pingpong={(0, 1): 3, (0, 2): 3})
QUIET2 = dict(ho_att={(0, 1): 20, (0, 2): 20})


def test_mro_late_cell_wide_lowers_ttt_then_hys_after_a_full_fresh_window():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 3.0, 480
    x = mro(e)
    two_windows(x, 0, **LATE2)
    assert knobs(x.propose(60.0)) == {("ttt", 0): 320.0}          # tie in the envelope -> TTT first
    apply(x, x.propose(60.0), 60.0)
    assert x.propose(90.0) == []                                  # old windows only: wait
    x.observe([mob_rep(3, 60, **LATE2)])
    assert x.propose(90.0) == []                                  # half a fresh window: still wait
    x.observe([mob_rep(3, 90, **LATE2)])
    assert knobs(x.propose(120.0)) == {("hys", 0): 2.5}           # now Hys sits higher in the envelope


def test_mro_repairs_a_late_misset_toward_the_pre_push_values_and_stops():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 3.0, 480                      # S2-like rollout state; calm default 2 / 320
    x = mro(e)
    t = 0
    for _ in range(2):
        two_windows(x, t, **LATE2)
        t += 60
        apply(x, x.propose(float(t)), float(t), last_change={})
    assert (e.plant.hys[0], e.plant.ttt[0]) == (2.5, 320)
    two_windows(x, t, **QUIET2)
    assert x.propose(float(t + 60)) == []                         # no issue left -> no action


def test_mro_low_sinr_dwell_counts_as_too_late_evidence():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 3.0, 480
    x = mro(e)
    two_windows(x, 0, dwell={(0, 1): 3.0, (0, 2): 3.0}, **QUIET2)  # no RLF at all, only pre-RLF dwell
    assert knobs(x.propose(60.0)) == {("ttt", 0): 320.0}
    y = mro(fake_env())
    two_windows(y, 0, **QUIET2)
    assert y.propose(60.0) == []


def test_mro_never_raises_hys_on_a_too_late_cell():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 0.0, 320                     # early says "raise Hys first" (lower position)
    x = mro(e)
    two_windows(x, 0, ho_att={(0, 1): 10, (0, 2): 10}, too_late={(0, 1): 0.5},   # tl = 1 (< MIN_TL), 1/41 > thr
                pingpong={(0, 1): 3, (0, 2): 3})
    assert knobs(x.propose(60.0)) == {("ttt", 0): 480.0}
    y = mro(fake_env())
    y.p.hys[0], y.p.ttt[0] = 0.0, 320
    two_windows(y, 0, **EARLY2)
    assert knobs(y.propose(60.0)) == {("hys", 0): 0.5}            # same without too-late evidence: Hys+


def test_mro_early_cell_wide_raises_ttt_from_the_floor():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 0.0, 40
    x = mro(e)
    two_windows(x, 0, **EARLY2)
    assert knobs(x.propose(60.0)) == {("ttt", 0): 80.0}


def test_mro_overshoot_reverses_last_step_and_locks_direction():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 3.0, 480
    x = mro(e)
    two_windows(x, 0, **LATE2)
    apply(x, x.propose(60.0), 60.0)                               # TTT 480 -> 320
    two_windows(x, 60, **EARLY2)
    assert knobs(x.propose(120.0)) == {("ttt", 0): 480.0}         # reverse the same knob
    apply(x, x.propose(120.0), 120.0, last_change={})
    two_windows(x, 120, **LATE2)
    assert knobs(x.propose(180.0)) == {("hys", 0): 2.5}           # TTT-down locked; Hys instead


def test_mro_single_pair_issue_moves_that_pair_cio_within_band():
    x = mro(fake_env())
    two_windows(x, 0, ho_att={(0, 1): 20}, too_late={(0, 1): 3})
    assert knobs(x.propose(60.0)) == {("cio", 0, 1): 1.0}
    e2 = fake_env()
    e2.plant.cio[0, 1] = X2.MROv2.CIO_ENV
    y = mro(e2)
    two_windows(y, 0, ho_att={(0, 1): 20}, too_late={(0, 1): 3})
    assert knobs(y.propose(60.0)) == {("ttt", 0): 256.0}         # band exhausted -> cell-level fallback
    z = mro(fake_env())
    two_windows(z, 0, ho_att={(0, 1): 20}, pingpong={(0, 1): 4})
    assert knobs(z.propose(60.0)) == {("cio", 0, 1): -1.0}


def test_mro_needs_minimum_evidence_and_two_windows():
    x = mro(fake_env())
    two_windows(x, 0, ho_att={(0, 1): 20, (0, 2): 20}, too_late={(0, 1): 0.5, (0, 2): 0.0})   # tl = 1 < 2
    assert x.propose(60.0) == []
    y = mro(fake_env())
    y.observe([mob_rep(3, 0, **LATE2)])
    assert y.propose(30.0) == []


def test_mro_conflicting_late_and_early_leaves_trigger_alone():
    x = mro(fake_env())
    two_windows(x, 0, ho_att={(0, 1): 20, (0, 2): 20}, too_late={(0, 1): 2, (0, 2): 2},
                pingpong={(0, 1): 3, (0, 2): 3})
    assert not any(r["knob"][0] in ("hys", "ttt") for r in x.propose(60.0))


def test_mro_envelope_caps_increases_but_allows_decreases_from_outside():
    e = fake_env()
    e.plant.hys[0], e.plant.ttt[0] = 3.0, 480
    x = mro(e)
    two_windows(x, 0, **EARLY2)
    assert x.propose(60.0) == []                                  # already at the envelope top
    e2 = fake_env()
    e2.plant.hys[0], e2.plant.ttt[0] = 5.0, 640
    y = mro(e2)
    two_windows(y, 0, **LATE2)
    assert knobs(y.propose(60.0)) == {("hys", 0): 4.5}            # Hys furthest above the envelope


def test_mro_wrong_cell_lowers_pair_cio():
    x = mro(fake_env())
    two_windows(x, 0, ho_att={(0, 1): 20}, wrong_cell={(0, 1): 3})
    assert knobs(x.propose(60.0)) == {("cio", 0, 1): -1.0}


def test_mro_proposals_bounded_and_feasible_on_random_inputs():
    rng = np.random.default_rng(1)
    n_req = 0
    for trial in range(200):
        e = fake_env(nc=4, seed=trial)
        e.plant.hys[:] = rng.integers(0, 11, 4) / 2
        e.plant.ttt[:] = rng.choice(C.TTT_SET_MS, 4)
        e.plant.cio[:] = rng.integers(-6, 7, (4, 4))
        x = mro(e)
        ent = {k: {(s, n): float(rng.poisson(lam)) for s in range(4) for n in range(4) if s != n}
               for k, lam in (("ho_att", 15), ("too_late", 0.5), ("too_early", 0.3), ("pingpong", 0.8),
                              ("wrong_cell", 0.3))}
        dw = {(s, n): float(rng.exponential(0.3)) for s in range(4) for n in range(4) if s != n}
        two_windows(x, 0, dwell=dw, **ent)
        hys0, ttt0, cio0 = e.plant.hys.copy(), e.plant.ttt.copy(), e.plant.cio.copy()
        reqs = x.propose(60.0)
        n_req += len(reqs)
        for r in reqs:
            k, v = r["knob"], r["prop"]
            if k[0] == "hys":
                assert abs(v - hys0[k[1]]) <= X2.MROv2.HYS_STEP + 1e-9
                assert v <= max(X2.MROv2.HYS_ENV[1], hys0[k[1]]) and v >= X2.MROv2.HYS_ENV[0]
            elif k[0] == "ttt":
                i0, i1 = C.TTT_SET_MS.index(int(ttt0[k[1]])), C.TTT_SET_MS.index(int(v))
                assert abs(i1 - i0) == 1 and v <= max(X2.MROv2.TTT_ENV[1], ttt0[k[1]])
            else:
                assert abs(v - cio0[k[1], k[2]]) == 1.0
                assert abs(v) <= X2.MROv2.CIO_ENV or abs(v) < abs(cio0[k[1], k[2]])
            ok = feasible(e.plant, k, v, 60.0, {})
            assert ok[1] and ok[0] == pytest.approx(v)
    assert n_req > 0


# ================================================================================================= KPM extension
def _fake_meas_plant():
    nc = 3
    return SimpleNamespace(nc=nc, n=4, t=0, serv=np.array([0, 0, 0, 1]), int_until=np.array([0.0, 0.0, 5.0, 0.0]),
                           q=np.array([0.0, 1e3, 1e3, 0.0]), cio=np.zeros((nc, nc)), hys=np.full(nc, 2.0),
                           occ=np.full(nc, 0.5), sinr_ewma=np.array([-9.0, 5.0, -9.0, -9.0]),
                           l3=np.array([[-80.0, -80.5, -95.0],       # margin to 1 = -2.5 dB -> needs +3 dB
                                        [-80.0, -78.0, -95.0],       # margin to 1 = 0 dB (not entering) -> +1 dB
                                        [-80.0, -78.5, -95.0],       # in outage: excluded
                                        [-77.5, -80.0, -95.0]]))     # serving 1, margin to 0 = +0.5: entering


def test_border_snapshot_bins_margins_cumulatively_with_rate_estimates():
    rep = X2.border_snapshot(_fake_meas_plant())
    np.testing.assert_array_equal(rep["conn"], [2, 1, 0])
    np.testing.assert_array_equal(rep["border_all"][0, 1], [1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(rep["border_act"][0, 1], [1, 1, 1, 1, 1, 1])
    assert rep["border_all"][0, 2].sum() == 0 and rep["border_all"][1].sum() == 0
    assert np.isfinite(rep["se_src"][0, 1]).all() and np.isnan(rep["se_src"][0, 2]).all()
    assert rep["se_tgt"][0, 1, 0] > rep["se_src"][0, 1, 0]           # UE 1 measures cell 1 2 dB stronger


def test_lowq_snapshot_attributes_low_sinr_dwell_to_strongest_other_cell():
    d = X2.lowq_snapshot(_fake_meas_plant())
    # UE0: SINR < Qout but serving is the strongest -> no; UE1: SINR fine; UE2: outage; UE3: serving 1, cell 0 stronger
    expect = np.zeros((3, 3))
    expect[1, 0] = 1.0
    np.testing.assert_array_equal(d, expect)


def _run(env, secs):
    for _ in range(secs):
        env.step(None)


def test_kpmv2_leaves_v1_stack_bit_identical():
    cfg = C.E6Config(seed=7, mix="M4", mobility="mixed", warmup_s=0.0, scored_s=40.0)
    a, b = E6Env(cfg), E6Env(cfg)
    b.kpm = X2.KPMV2(cfg, b.plant)
    _run(a, 40)
    _run(b, 40)
    assert a.stats == b.stats
    assert [la["config"] for la in a.log] == [lb["config"] for lb in b.log]
    ra = [(r["gran"], r["t1"], r["arrived"]) for r in a.kpm.delivered]
    rb = [(r["gran"], r["t1"], r["arrived"]) for r in b.kpm.delivered if r["gran"] not in ("mr", "mobq")]
    assert ra == rb
    assert {"mr", "mobq"} <= {r["gran"] for r in b.kpm.delivered}


def test_make_env_v2_builds_runs_and_keeps_v1_registry():
    before = dict(X1.MIXES)
    cfg = C.E6Config(seed=7, mix="V2_M4", mobility="mixed", warmup_s=0.0, scored_s=60.0, scenario="mistune")
    env = X2.make_env_v2(cfg, trace=True)
    assert all(X1.MIXES[k] == v for k, v in before.items())
    assert isinstance(env.kpm, X2.KPMV2) and not isinstance(E6Env(C.E6Config(seed=7, mix="M4")).kpm, X2.KPMV2)
    assert [x.name for x in env.xapps] == ["MRO", "TS", "ES", "SLICE"]
    v1 = E6Env(C.E6Config(seed=7, mix="M4", mobility="mixed", warmup_s=0.0, scored_s=60.0, scenario="mistune"))
    for i in (2, 3):                                              # reused ES / SLICE: same variant draws as v1
        assert (env.xapps[i].scale, env.xapps[i].behaviour) == (v1.xapps[i].scale, v1.xapps[i].behaviour)
    nc = env.plant.nc
    assert all(("ll_ratio", c) in env.knobs for c in range(nc))
    assert sum(k[0] == "cio" for k in env.knobs) == nc * (nc - 1) and len(set(env.knobs)) == len(env.knobs)
    _run(env, 60)
    reqs = [r for row in env.log for r in row["requests"]]
    assert all(r["knob"] in env.knobs for r in reqs)
    for r in reqs:
        if r["knob"][0] == "cio":
            assert C.CIO_RANGE[0] <= r["prop"] <= C.CIO_RANGE[1]
    env.get_trace()
    with pytest.raises(ValueError):
        X2.make_env_v2(C.E6Config(seed=7, mix="M4"))


def test_nrt_covers_observed_ho_and_rlf_pairs_in_a_running_env():
    """Mechanical acceptance criterion from E6_V1_DIAGNOSTICS.md (defect 1): >= 95 % of HO attempts and too-late RLF
    pairs in each mob report fall on a relation already in the run-time NRT (built from the reports delivered before
    it). Mechanism only: no SLA field is read."""
    cfg = C.E6Config(seed=7, mix="V2_M4", mobility="mixed", warmup_s=0.0, scored_s=240.0, scenario="mistune")
    env = X2.make_env_v2(cfg, log=False)
    _run(env, 240)
    nrt = X2.NRT(env.plant.nc)
    lay = np.zeros((env.plant.nc,) * 2, bool)
    for s, ns in enumerate(env.plant.lay.neighbours):
        lay[s, ns] = True
    cov, cov_lay = [], []
    for rep in sorted(env.kpm.delivered, key=lambda r: r["arrived"]):
        if rep["gran"] == "mob" and rep["t0"] >= 60:
            ev = rep["ho_att"] + rep["too_late"]
            cov.append((ev[nrt.rel].sum(), ev.sum()))
            cov_lay.append((ev[lay].sum(), ev.sum()))
        nrt.update([rep])
    frac = sum(a for a, _ in cov) / max(sum(b for _, b in cov), 1)
    frac_lay = sum(a for a, _ in cov_lay) / max(sum(b for _, b in cov_lay), 1)
    assert frac >= 0.95 and frac > frac_lay


def test_v2_mix_positions_match_v1_m4():
    assert [c.name for c in X2.MIXES_V2["V2_M4"]] == [c.name for c in X1.MIXES["M4"]]
    assert X2.MIXES_V2["V2_M4"][2] is X1.ES and X2.MIXES_V2["V2_M4"][3] is X1.SliceSLA
    assert KPM is not X2.KPMV2 and issubclass(X2.KPMV2, KPM)
