"""DEV probe campaign (decision/probe.py) + design-based CRT (decision/crt.py)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from cdd_oran.decision import crt
from cdd_oran.decision import probe as P
from cdd_oran.discovery import mscr
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import LIMITS

SEEDS = range(110000, 110040)


def _cfg(seed, **kw):
    kw = {"load": "medium", "mobility": "mixed", "mix": "M4", "warmup_s": 60, "scored_s": 300, **kw}
    return C.E6Config(seed=seed, **kw)


@pytest.fixture(scope="module")
def layout():
    env = E6Env(_cfg(110000, scored_s=1800), log=False)
    return env.knobs, np.asarray(env.plant.lay.cell_site)


def test_schedule_caps_washout_isolation(layout):
    knobs, site = layout
    pc = P.ProbeConfig(max_treated=3)
    regions, nbhd, _ = P.region_units(knobs, site)
    unit = {g: {g} | nbhd[g] for g in regions}
    macro_regions = {int(site[k[1]]) for k in knobs if k[0] == "carrier"}
    n_blocks = 0
    for s in SEEDS:
        blocks = P.make_schedule(s, knobs, site, 300.0, 2100.0, pc)
        n_blocks += len(blocks)
        worst = 0.0
        for b in blocks:
            assert 300.0 <= b["t0"] < b["obs_end"] <= b["slot_end"] <= 2100.0
            assert b["obs_end"] - b["t0"] == (pc.obs_b_s if b["type"] else pc.obs_a_s)
            if b["type"] == 1:                                  # carrier slots: observe/wash out past the dwell
                assert b["obs_end"] - b["t0"] > LIMITS["carrier"][3] and b["region"] in macro_regions
                assert b["arm"] in pc.arms_b
            else:
                assert b["arm"] in pc.arms_a
            worst += 2 * max(len(P.family_knobs(knobs, site, b["region"], f)) for f in P.FAMILIES
                             if b["elig"][P.ARMS.index(f)])
            same = [c for c in blocks if c["slot"] == b["slot"] and c is not b]
            assert len(same) + 1 <= pc.max_treated
            for c in same:                                      # neighbourhoods untreated, units disjoint
                assert not unit[b["region"]] & unit[c["region"]]
                assert c["t0"] == b["t0"] and c["slot_end"] == b["slot_end"]
            for c in blocks:                                    # time-disjoint blocks across slots
                if c["slot"] != b["slot"]:
                    assert c["t0"] >= b["slot_end"] or c["slot_end"] <= b["t0"]
        assert worst <= pc.cap_changes_per_hour * 1800 / 3600 + 1e-9
    assert n_blocks > 100
    tight = P.ProbeConfig(cap_changes_per_hour=120.0)            # 60 changes over 30 min
    for s in list(SEEDS)[:10]:
        bl = P.make_schedule(s, knobs, site, 300.0, 2100.0, tight)
        assert sum(2 * max(len(P.family_knobs(knobs, site, b["region"], f)) for f in P.FAMILIES
                           if b["elig"][P.ARMS.index(f)]) for b in bl) <= 60
    a, b = (P.make_schedule(7, knobs, site, 300.0, 2100.0, pc) for _ in range(2))
    assert [(x["t0"], x["region"], x["arm"], x["level"]) for x in a] == \
        [(x["t0"], x["region"], x["arm"], x["level"]) for x in b]


def test_mechanism_resampling_reproduces_logged_propensities(layout):
    knobs, site = layout
    pc = P.ProbeConfig()
    mech = P.AssignmentMechanism(pc)
    blocks = [b for s in SEEDS for b in P.make_schedule(s, knobs, site, 300.0, 2100.0, pc)]
    for b in blocks:
        assert b["prop"] == pytest.approx(mech.prob(b["type"], b["elig"], b["arm"], b["level"]))
    rng = np.random.default_rng(0)
    for typ in (0, 1):
        b = next(x for x in blocks if x["type"] == typ)
        n = 20000
        draws = [mech.sample(rng, typ, b["elig"]) for _ in range(n)]
        for arm in P.ARMS:
            for lv in mech.levels(arm):
                freq = sum(1 for d in draws if d == (arm, lv)) / n
                p = mech.prob(typ, b["elig"], arm, lv)
                assert abs(freq - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-12
    # conditional re-draws (arm in {f, sham}): P(f) and levels as the mechanism says
    st = np.array([b["type"] for b in blocks])
    el = np.array([b["elig"] for b in blocks])
    sel = st == 0
    X = mech.conditional_draws(rng, st[sel], el[sel], "hys", 400)
    pf = mech.arm_probs(0, el[sel][0])[P.ARMS.index("hys")]
    target = pf / (pf + mech.arm_probs(0, el[sel][0])[P.ARMS.index("sham")])
    assert abs((X != 0).mean() - target) < 0.01
    assert set(np.unique(X)) <= {-1.0, 0.0, 1.0} and abs((X == 1).mean() - (X == -1).mean()) < 0.01


def test_mscr_statistic_is_reused_exactly():
    rng = np.random.default_rng(3)
    n = 120
    Z = rng.normal(size=(n, 3))
    x = rng.choice([-1.0, 0.0, 1.0], size=n)
    y = x * (Z[:, 0] > 0) + rng.normal(size=n)
    cfg = crt.CRTConfig()
    st = crt.MSCRStat(Z, y, cfg)
    frozen = mscr.discover_mscr(np.column_stack([x, Z]), y, 1, seed=1,
                                config=crt.CRTConfig(B=9).mscr_config(), n_jobs=1, device="cpu")
    assert st.observed(x) == pytest.approx(frozen.s_star[0, 0], abs=0, rel=1e-12)
    X = np.stack([rng.permutation(x) for _ in range(5)])
    assert crt.check_draws(st, X, k=5) < 1e-12


def test_crt_valid_on_null_and_powered_on_effect():
    cal = crt.calibrate(n_rep=150, effects=(0.0, 1.0), seed=1)
    null, eff = cal[0.0], cal[1.0]
    assert null["n"] == 150 and eff["n"] == 150
    # binomial(150, .05): 99.9 % band ~ [0, 0.12]
    assert null["rej_mscr"] <= 0.12 and null["rej_signed"] <= 0.12
    assert eff["rej_mscr"] >= 0.6 and eff["rej_signed"] >= 0.5


@pytest.fixture(scope="module")
def forced_breach_trace():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return P.run_probe_episode(_cfg(110007), P.ProbeConfig(abort_ll_abs_s=-1.0, abort_ll_rel=0.0))


def test_abort_restore_fires_on_forced_breach(forced_breach_trace):
    a = forced_breach_trace.arrays
    assert len(a["blk_arm"]) >= 1 and a["blk_abort"].all()
    assert (a["blk_abort_reason"] == P.ABORT_REASONS.index("ll")).all()
    assert (a["blk_abort_t"] < a["blk_obs_end"]).all()
    treated = a["blk_arm"] != P.ARMS.index("sham")
    assert treated.any() and (a["blk_n_applied"][treated] > 0).all()
    # every applied probe knob is restored (or was superseded by an xApp) before its slot ends
    assert (a["blk_n_restored"] + a["blk_n_superseded"] == a["blk_n_applied"]).all()
    rest = (a["pw_kind"] == 1) & (a["pw_out"] == P.PW_OUT.index("applied"))
    for i in np.nonzero(treated)[0]:
        t_r = a["pw_t"][rest & (a["pw_blk"] == i)]
        assert len(t_r) and t_r.min() >= a["blk_abort_t"][i] and t_r.max() <= a["blk_slot_end"][i]
    assert forced_breach_trace.meta["probe"]["not_a_wg3_policy"] and not forced_breach_trace.meta["wg3"]


def test_end_to_end_tiny_real_probe_episode():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cfg = _cfg(110009)
        tr = P.run_probe_episode(cfg)
        ref = P.run_reference(cfg)
    a = tr.arrays
    assert len(a["blk_t0"]) >= 2 and len(a["wr_t"]) == int(tr.meta["probe"]["probe_changes"])
    cost = P.collection_cost(tr, ref)
    assert cost["blocks"] == len(a["blk_t0"]) and cost["probe_changes"] > 0
    assert np.isfinite(cost["excess_svr"]) and cost["changed_knob_s"] > 0
    small = crt.CRTConfig(B=49, min_f=1, min_sham=0, min_episodes=1)
    data = crt.build_block_data([tr, tr])                    # duplicated episode: plumbing only, not inference
    out = crt.run_crt(data, cfg=small)
    assert out["version"] == crt.CRT_VERSION and "NO MSCR FDR claim" in out["selection"]
    assert {r["status"] for r in out["results"]} <= {"declared", "not_detected", "undetermined"}
    assert any(r["reason"] for r in out["results"] if r["status"] == "undetermined")
    placebo = crt.build_block_data([ref], schedule_from=[tr])
    assert placebo.n == len(a["blk_t0"]) and np.array_equal(placebo.arm, a["blk_arm"])
    default = crt.run_crt(crt.build_block_data([tr]))       # one episode: nothing is supported
    assert all(r["status"] == "undetermined" for r in default["results"])
