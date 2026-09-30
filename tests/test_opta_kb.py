"""Option (a) K-B: the confounded incumbent logging policy (collect_p.IncumbentPolicy / PlaceboIncumbent), per-unit
propensity rows in crt_units.build_unit_data, MSCR-CRT v2 validity under context-dependent logging, the tap's
count_all flag and arbiter-from-t=0 collection, the K-B driver's seed guard, and the analysis pieces (placebo tau,
maps, offline MapGateV2 replay)."""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import crt_units_v2 as V2
from cdd_oran.decision import mapgate as MG
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
E6DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")
if E6DEV not in sys.path:
    sys.path.insert(0, E6DEV)


def ctx(knob, step, own=np.nan, nbr=np.nan, below=np.nan):
    return {"knob": knob, "step": float(step), "own_prb_util": own, "nbr_max_prb_util": nbr,
            "own_prot_below_frac": below}


# ---------------------------------------------------------------------------------------------- incumbent policy
@pytest.mark.parametrize("c, cls, pa", [
    (ctx("sleep", 1, .3, .5), "saving", .85), (ctx("sleep", 1, .3, .7), "saving", .15),
    (ctx("carrier", -1, .61, .1), "saving", .15), (ctx("ptx", -1, .59, np.nan), "saving", .85),
    (ctx("sleep", -1, .2, .2), "restore", .35), (ctx("carrier", 1, .9, .2), "restore", .85),
    (ctx("ptx", 1, np.nan, .6), "restore", .85), (ctx("ptx", 1), "restore", .35),
    (ctx("prot_min", .05, below=.06), "sg_raise", .85), (ctx("prot_min", .05, below=.05), "sg_raise", .30),
    (ctx("prot_min", -.05, below=0.0), "sg_lower", .85), (ctx("prot_min", -.05, below=.01), "sg_lower", .20),
    (ctx("prot_min", -.05), "sg_lower", .85), (ctx("prot_min", 0.0, below=.2), "sg_raise", .85),
])
def test_incumbent_table(c, cls, pa):
    pol = CP.IncumbentPolicy(1)
    p, info = pol.accept_prob(c)
    assert info["cls"] == cls and p == pytest.approx(pa)
    row = pol.row({"ctx": c})
    assert set(row) == {"accept", "reject"} and sum(row.values()) == pytest.approx(1.0)
    assert min(row.values()) >= CP.INC_FLOOR - 1e-12


def test_incumbent_floor_clips_and_unknown_knob_raises():
    pol = CP.IncumbentPolicy(1, table={**CP.INCUMBENT_TABLE, "saving": {"lo": 1.0, "hi": 0.0}})
    assert pol.accept_prob(ctx("sleep", 1, .1))[0] == pytest.approx(.85)
    assert pol.accept_prob(ctx("sleep", 1, .9))[0] == pytest.approx(.15)
    with pytest.raises(ValueError):
        CP.incumbent_class(ctx("cio", 1))


class StrictUnit(dict):
    """A unit that only exposes the obs-only fields a logging policy may read (the RNG key + ctx)."""
    ALLOWED = {"c", "x_idx", "t0", "ctx"}

    def __getitem__(self, k):
        if k not in self.ALLOWED:
            raise KeyError(f"policy read {k!r}")
        return super().__getitem__(k)

    def get(self, k, d=None):
        if k not in self.ALLOWED:
            raise KeyError(f"policy read {k!r}")
        return super().get(k, d)


def test_incumbent_draws_keyed_logged_and_obs_only():
    pol = CP.IncumbentPolicy(77)
    u = StrictUnit(c=5, x_idx=0, t0=61.0, ctx=ctx("sleep", 1, .7, .2), mode="accept", x="ES")
    m, p = pol(u)
    uu = float(np.random.default_rng([77, 6622, 5, 0, 61]).random())
    assert m == ("accept" if uu < .15 else "reject")
    assert p == pytest.approx(dict.__getitem__(u, "probs")[m])
    assert dict.__getitem__(u, "inc")["cls"] == "saving"
    assert all(pol(StrictUnit(c=5, x_idx=0, t0=61.0, ctx=ctx("sleep", 1, .7, .2)))[0] == m for _ in range(3))
    cnt = {"accept": 0, "reject": 0}                       # accept frequency follows the row
    for t0 in range(4000):
        cnt[pol({"c": 3, "x_idx": 1, "t0": float(t0), "ctx": ctx("ptx", -1, .2, .1)})[0]] += 1
    assert cnt["accept"] / 4000 == pytest.approx(.85, abs=.02)


def test_placebo_incumbent_same_draw_applies_accept_and_relabels():
    inc, plc = CP.IncumbentPolicy(9), CP.PlaceboIncumbent(9)
    units = []
    for t0 in range(0, 200, 7):
        c = ctx("carrier", -1 if t0 % 2 else 1, (t0 % 10) / 10, .3)
        a = {"c": 2, "x_idx": 0, "t0": float(t0), "ctx": c}
        b = dict(a)
        m, p = inc(a)
        mode, pp = plc(b)
        assert (mode, pp) == ("accept", 1.0) and (b["pi0_mode"], b["pi0_p"]) == (m, p) and b["probs"] == a["probs"]
        b.update(mode=mode, p=pp)
        units.append((b, m, p))
    CP.finalize_units([b for b, _, _ in units])
    for b, m, p in units:
        assert (b["mode"], b["p"], b["applied_mode"]) == (m, p, "accept") and "pi0_mode" not in b
    assert {m for _, m, _ in units} == {"accept", "reject"}


# ---------------------------------------------------------------------------------------------- build_unit_data
def _with_rows(recs, rng, conf=0.0):
    """Give every unit the IncumbentPolicy row of a random pressure s (own_prb_util) and its knob / step, re-draw its
    mode from that row and (conf > 0) add conf * [s >= .6] to every exposure cell's load and pv over the unit's post
    window: a confounder of the assignment and the outcome, NO treatment effect (the outcome ignores the mode)."""
    from cdd_oran.decision.collect_p import dec, enc
    inc = CP.IncumbentPolicy(0)
    out = []
    for r in recs:
        r = dict(r, pi0_table=None)
        ls = dict(r["lab_series"])
        D = dec(ls["data"]).astype(np.float64)
        f = {k: i for i, k in enumerate(ls["fields"])}
        units = []
        for u in r["units"]:
            u = dict(u)
            s = float(rng.uniform())
            pa = inc.accept_prob({"knob": u["knob"], "step": u["step"], "own_prb_util": s,
                                  "nbr_max_prb_util": np.nan, "own_prot_below_frac": 0.0})[0]
            m = "accept" if rng.uniform() < pa else "reject"
            u.update(mode=m, applied_mode=m, p=pa if m == "accept" else 1 - pa,
                     probs={"accept": pa, "reject": 1 - pa}, ctx=dict(u["ctx"], own_prb_util=s))
            if conf and s >= .6:
                t0 = int(u["t0"])
                for q in u["exp"]:
                    D[t0:t0 + CU.H_UNIT, f["ue"], q] += conf / CU.H_UNIT
                    D[t0:t0 + CU.H_UNIT, f["pv"], q] += conf / CU.H_UNIT
            units.append(u)
        ls["data"] = enc(D)
        out.append(dict(r, units=units, lab_series=ls))
    return out


def test_build_unit_data_prefers_the_per_unit_row():
    recs = _with_rows(CU.synthetic_records(np.random.default_rng(3), n_ep=3, units_per_ep=40), np.random.default_rng(4))
    d = CU.build_unit_data(recs)
    assert d.meta["n_probs_rows"] == d.n > 0 and d.meta["p_mismatch"] == 0
    rows = np.array([[u["probs"].get(m, 0.0) for m in CU.MODES] for r in recs for u in r["units"]
                     if 90 <= u["t0"] <= 720 - 90])
    assert np.allclose(np.sort(d.probs[:, 0]), np.sort(rows[:, 0]))
    assert set(np.round(d.probs[:, 0], 6)) <= {.15, .3, .35, .85} and np.allclose(d.probs.sum(1), 1.0)
    assert len(set(np.round(d.probs[:, 0], 6))) >= 3
    assert np.allclose(d.p, d.probs[np.arange(d.n), d.mode])
    recs[0]["units"][0]["p"] = 0.5                                   # p inconsistent with the row -> counted
    assert CU.build_unit_data(recs).meta["p_mismatch"] == 1
    recs[0]["units"][0]["probs"] = {"accept": .7, "reject": .7}      # not a distribution -> error
    with pytest.raises(ValueError):
        CU.build_unit_data(recs)
    plain = CU.synthetic_records(np.random.default_rng(3), n_ep=2, units_per_ep=20)   # no rows: table path unchanged
    dp = CU.build_unit_data(plain)
    assert dp.meta["n_probs_rows"] == 0 and dp.meta["p_mismatch"] == 0
    assert np.allclose(dp.probs[:, :3], [[.5, .2, .3]] * dp.n)


def test_h_pre_60_keeps_units_opening_at_60_to_89():
    recs = CU.synthetic_records(np.random.default_rng(5), n_ep=1, units_per_ep=10)
    recs[0]["units"][0]["t0"] = 65.0
    assert CU.build_unit_data(recs).meta["dropped"]["window"] >= 1
    d60 = CU.build_unit_data(recs, H_pre=60)
    assert 65 in set(d60.t0.tolist()) and d60.H_pre == 60


def test_crt_v2_valid_under_context_dependent_logging_and_invalid_with_a_pooled_table():
    """Confounder s drives both P(accept) and the outcome; no effect. Per-unit rows: sleep -> nbr load p-values are
    ~uniform; replacing the rows by one pooled (context-free) row makes the same statistic reject."""
    cfg = V2.UnitCRTConfigV2(B=199)
    p_ok, p_bad = [], []
    for rep in range(24):
        recs = _with_rows(CU.synthetic_records(np.random.default_rng(100 + rep), n_ep=4, units_per_ep=60),
                          np.random.default_rng(200 + rep), conf=300.0)
        d = CU.build_unit_data(recs)
        p_ok.append(V2.crt_unit_test_v2(d, "sleep", "nbr", "load", cfg)["p"])
        rows = d.rows_of("sleep")
        pooled = d.probs.copy()
        pooled[rows] = d.probs[rows].mean(0)
        bad = dataclasses.replace(d, probs=pooled)
        p_bad.append(V2.crt_unit_test_v2(bad, "sleep", "nbr", "load", cfg)["p"])
    p_ok, p_bad = np.array(p_ok), np.array(p_bad)
    assert (p_ok <= .05).mean() <= .2 and np.median(p_ok) > .2
    assert (p_bad <= .05).mean() >= .6


# ---------------------------------------------------------------------------------------------- collection (env)
def cfg_p3(seed=6, warm=60.0, scored=40.0):
    return C.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=1.0, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=warm, scored_s=scored,
                      e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


def test_incumbent_collection_from_t0_count_all_tap_and_placebo_is_accept_all():
    cfg = cfg_p3()
    res = CP.run_collection(cfg, CP.IncumbentPolicy(cfg.seed), open_rule="feasible", arb_warmup_s=0.0,
                            count_all=True)
    units, tap = res["units"], res["tap"]
    assert res["arb"].warmup_s == 0.0 and any(u["t0"] < cfg.warmup_s for u in units)
    for u in units:
        assert u["p"] == pytest.approx(u["probs"][u["mode"]]) and u["mode"] in CP.INC_ORDER
        assert u["probs"]["accept"] == pytest.approx(CP.IncumbentPolicy(1).accept_prob(u["ctx"])[0])
    s = tap.series()
    D = s["data"].astype(float)
    f = {k: i for i, k in enumerate(CP.SERIES_FIELDS)}
    assert np.array_equal(D[:, f["pv"]].sum(0), tap.pv) and np.array_equal(D[:, f["ue"]].sum(0), tap.load)
    assert not s["scored"].all()                                   # the warm-up is still unscored by the plant
    ref = E6Env(cfg, log=False, wg3=True)
    while ref.sec < ref.total_s:
        ref.step(None)
    pl = CP.run_collection(cfg, CP.PlaceboIncumbent(cfg.seed), open_rule="feasible", arb_warmup_s=0.0)
    assert pl["env"].plant.sla == ref.plant.sla
    assert all(u["applied_mode"] == "accept" and u["p"] == pytest.approx(u["probs"][u["mode"]]) for u in pl["units"])
    assert np.array_equal(pl["tap"].pv, CP.get_tap(pl["env"]).pv)
    sc = pl["tap"].series()["scored"]
    assert pl["tap"].pv.sum() == pl["tap"].series()["data"][sc, 0].sum()   # default tap: scored seconds only


# ---------------------------------------------------------------------------------------------- driver / analysis
def test_driver_seed_guard_jobs_and_registry():
    import e6p_opta_kb as KB
    J = KB.jobs("all")
    assert len(J) == 80 and {s for s, _, _ in J} == {"placebo", "applied"}
    assert sorted(sd for _, sd, _ in J) == list(range(186000, 186080))
    assert [x[0] for x in J[:4]] == ["placebo", "applied", "placebo", "applied"]
    for bad in (185999, 186080, 183400, 184200):
        with pytest.raises(AssertionError):
            KB.check_seed(bad)
    reg = KB.registry_check()
    assert reg is not None and all(reg.values())


def test_placebo_tau_and_maps():
    import e6p_opta_kb_analyze as AN
    sc = {"sleep|nbr|pv": {"score": 3.0, "sign": 1}, "sleep|own|pv": {"score": 2.0, "sign": -1},
          "ptx|far|e": {"score": 1.0, "sign": 1}, "ptx|own|e": {"score": float("nan"), "sign": 0}}
    assert AN.placebo_tau(sc, ("own", "nbr", "far"))["tau"] == 2.0          # <= 1 placebo declaration
    assert AN.placebo_tau(sc, ("own",))["tau"] == float("-inf")
    d = CU.build_unit_data(CU.synthetic_records(np.random.default_rng(11), n_ep=4, units_per_ep=60, effect=200.0))
    decl = {("sleep", "nbr", "load"): {"declared": True, "sign": 1}, ("sleep", "own", "load"): {"declared": True,
                                                                                               "sign": 0},
            ("ptx", "own", "e"): {"declared": False, "sign": 1}}
    M, drop = AN.map_from_declared(decl, d)
    assert list(M) == [("sleep", "nbr", "load")] and drop == 1
    b = M[("sleep", "nbr", "load")]
    assert b > 0 and b == pytest.approx(abs(AN.naive_slope(d, "sleep", "nbr", "load")))
    q = AN.edge_quality({("sleep", "nbr", "pv"): 2.0, ("sleep", "own", "pv"): 1.0},
                        [{"family": "sleep", "relation": "nbr", "kpi": "pv", "status": "TRUE", "mean": 9.0},
                         {"family": "sleep", "relation": "own", "kpi": "pv", "status": "TRUE", "mean": -3.0}])
    assert q["n_true_right"] == 1 and q["n_true_wrong"] == 1


def test_offline_replay_counts_flips_against_the_gt_map():
    import e6p_opta_kb_analyze as AN
    M = {("sleep", "nbr", "pv"): 5.0, ("sleep", "own", "e"): -100.0}   # sleep hurts nbr pv, saves energy
    flip = MG.sign_flip(M, ("sleep",))
    units = []
    for i, (step, nbr_below) in enumerate([(1, .5), (1, .0), (-1, .5), (1, .9)]):
        units.append({"c": i, "knob": "sleep", "t0": 100.0 + i,
                      "ctx": {"knob": "sleep", "step": float(step), "own_prot_below_frac": 0.0,
                              "nbr_max_prot_below_frac": nbr_below}})
    units.append({"c": 9, "knob": "cio", "t0": 1.0, "ctx": {"knob": "cio", "step": 1.0}})   # not a MapGate family
    eps = [{"units": units}, {"units": units[:1]}]
    rp = AN.replay(eps, {"GT": M, "same": dict(M), "flip": flip, "empty": {}}, "GT")
    assert rp["n_units"] == [4, 1] and rp["decisions"]["GT"] == [2, 1]      # sleep under nbr pressure deferred
    assert rp["flips"]["same"]["per_episode"] == [0, 0]
    assert rp["flips"]["empty"]["per_episode"] == [2, 1]
    assert rp["flips"]["empty"]["by"] == {"sleep+:refdefer->accept": [2, 1]}
    assert rp["flips"]["flip"]["per_episode"][0] >= 2
    fs = AN.flip_summary(rp, 200)
    assert fs["empty"]["mean_per_episode"] == 1.5 and fs["same"]["mean_per_episode"] == 0.0
    v = AN.verdict(True, fs, ["empty"], [])
    assert v["label"] == "CONTINUE" and v["best_dev_tau_vs_gt"]["arm"] == "empty"
    assert v["label_attributable"] == "KILL"                  # an empty map has no wrong edges: missing != confounded
    assert AN.verdict(False, fs, ["empty"], [])["label"] == "KILL"
    assert AN.verdict(True, fs, ["same"], [])["label"] == "KILL"
    # pairs: wrong-edge flips (map vs its clean part) and placebo-edge injection into the GT map
    cells = [{"family": "sleep", "relation": "nbr", "kpi": "pv", "status": "TRUE", "mean": 5.0},
             {"family": "sleep", "relation": "own", "kpi": "e", "status": "TRUE", "mean": -100.0}]
    bad = {**M, ("sleep", "nbr", "pv"): -5.0}                        # wrong sign on the harm edge
    assert AN.clean_map(bad, cells) == {("sleep", "own", "e"): -100.0}
    maps = {"GT": M, "bad": bad, "bad|clean": AN.clean_map(bad, cells), "GT+plc:bad": AN.inject(M, {k: v for k, v in
                                                                                                  bad.items() if k[1] == "nbr"})}
    rp2 = AN.replay(eps, maps, pairs={"bad~clean": ("bad", "bad|clean"), "GT+plc:bad": ("GT+plc:bad", "GT")})
    fs2 = AN.flip_summary(rp2, 200)
    assert fs2["bad~clean"]["mean_per_episode"] > 0 and fs2["GT+plc:bad"]["mean_per_episode"] >= 1
    v2 = AN.verdict(True, fs2, ["bad"], [])
    assert v2["label_attributable"] == "CONTINUE" and v2["best_dev_tau_attributable"]["arm"] == "bad"
