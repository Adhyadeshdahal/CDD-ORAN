"""mscr-eproc-units-v1 (cdd_oran.decision.eprocess_units): the design-centred e-process is a test martingale under the
sharp null EVEN WHEN the unit skeleton depends on the family's own past modes (a reject re-opens the request 60 s
later, an accept starts a dwell), its scores are predictable (no dependence on later units), every factor is
nonnegative, e-BH is correct, a planted effect is detected, and -- the gap it closes -- the within-(episode x sgn)
demeaned CRT statistic is biased on such a skeleton while the predictable residual is not."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import eprocess_units as EP
from cdd_oran.decision.crt_units_v2 import residualise

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

P = np.array([0.5, 0.2, 0.3])          # accept, half, reject (collect_p.PI0_HIGH_NO_RB)
LV = np.array([1.0, 1.0, 0.0])         # LEVEL_V2
M = float(P @ LV)
VMAX = max(M, 1 - M)


def _episode(rng, T=720, H=90, thr=0.8, dwell=150, delta=0.0, rho=0.995):
    """One episode: a persistent latent demand D drives the requests (sgn = sign D while |D| > thr); a unit's mode is
    drawn from pi0 at opening; an applied mode (accept / half) satisfies the request and blocks new units for
    ``dwell`` s, a reject leaves it pending -> the next unit at t0 + 60 (mode-dependent skeleton). The outcome series
    Y is fixed before any mode (sharp null) unless ``delta`` plants an additive effect on the unit window."""
    D = np.zeros(T)
    e = rng.normal(0, np.sqrt(1 - rho ** 2), T)
    for t in range(1, T):
        D[t] = rho * D[t - 1] + e[t]
    Y = 3.0 * D + rng.normal(0, 1, T)
    units, t, blocked = [], 120, -1
    while t < T - H:
        if t >= blocked and abs(D[t]) > thr:
            m = int(rng.choice(3, p=P))
            units.append((t, 1.0 if D[t] > 0 else -1.0, m))
            if LV[m] > 0:
                blocked, t = t + dwell, t + 1
            else:
                t += 60
            continue
        t += 1
    for t0, g, m in units:
        Y[t0:t0 + H] += delta * g * LV[m] / H
    return [(t0, g, m, Y[t0:t0 + H].sum() - Y[t0 - H:t0].sum(), Y[t0 - H:t0].sum()) for t0, g, m in units]


def _dataset(rng, n_ep, delta=0.0):
    rows = [(e,) + u for e in range(n_ep) for u in _episode(rng, delta=delta)]
    a = np.array(rows, float).reshape(-1, 6)
    return {"episode": a[:, 0].astype(int), "sgn": a[:, 2], "mode": a[:, 3].astype(int), "y": a[:, 4], "pre": a[:, 5]}


def _v(d):
    return d["sgn"] * (LV[d["mode"]] - M)


def _log_e(d, cfg=None, lam="mix"):
    cfg = cfg or EP.EProcConfig(lam=lam)
    s = EP.predictable_scores(d["y"], d["pre"], d["episode"], d["sgn"], cfg)
    return EP.eprocess_path(_v(d), s, np.full(len(s), VMAX), cfg)


def _crt_z(d, r):
    return float(_v(d) @ r / np.sqrt((P @ LV ** 2 - M ** 2) * (r @ r)))


# ------------------------------------------------------------------------------------------------ mechanics
def test_design_regressor_is_mean_zero_under_the_logged_table():
    d = CU.build_unit_data(CU.synthetic_records(np.random.default_rng(3), n_ep=3, units_per_ep=40))
    rows = d.rows_of("prot_min")
    v, vmax = EP.design_v(d, rows)
    from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR
    m = d.probs[rows] @ LEVEL_V2_ARR
    ev = d.sgn[rows] * sum(d.probs[rows][:, j] * (LEVEL_V2_ARR[j] - m) for j in range(len(LEVEL_V2_ARR)))
    assert np.allclose(ev, 0.0)
    assert np.all(np.abs(v) <= vmax + 1e-12)


def test_scores_are_predictable_and_ignore_the_modes():
    d = _dataset(np.random.default_rng(5), 6)
    cfg = EP.EProcConfig(burn=3)
    s = EP.predictable_scores(d["y"], d["pre"], d["episode"], d["sgn"], cfg)
    n = len(s)
    for k in (n // 4, n // 2, n - 2):                           # truncating the future never changes the past
        sk = EP.predictable_scores(d["y"][:k], d["pre"][:k], d["episode"][:k], d["sgn"][:k], cfg)
        assert np.array_equal(sk, s[:k])
    y2 = d["y"].copy()
    y2[n // 2:] += 1e6                                          # later outcomes do not move earlier scores
    s2 = EP.predictable_scores(y2, d["pre"], d["episode"], d["sgn"], cfg)
    assert np.array_equal(s2[: n // 2], s[: n // 2])
    assert np.all(np.abs(s) <= 1.0) and np.all(s[: cfg.burn] == 0.0)


def test_every_factor_is_nonnegative_at_the_extremes():
    cfg = EP.EProcConfig()
    v = np.array([VMAX, -VMAX, 1 - VMAX, -(1 - VMAX)] * 5)
    for sgn in (0, 1, -1):
        for s in (np.ones(len(v)), -np.ones(len(v))):
            for lam in EP.LAMS:
                r = EP.eprocess_path(v, s, np.full(len(v), VMAX), dataclasses.replace(cfg, lam=lam), sgn)
                assert np.all(np.isfinite(r["log_path"]))
                assert r["log_e_max"] >= r["log_e"] - 1e-12 or r["log_e"] < 0


def test_e_bh():
    e = np.array([500.0, 90.0, 3.0, 0.0, 40.0])
    # m = 5, q = .1: thresholds 50, 25, 16.7, 12.5, 10 for k = 1..5; sorted e = 500, 90, 40, 3, 0 -> k* = 3
    assert EP.e_bh(e, 0.1).tolist() == [True, True, False, False, True]
    assert not EP.e_bh(np.array([10.0, 10.0]), 0.05).any()
    assert EP.e_bh(np.array([]), 0.05).size == 0


# ------------------------------------------------------------------------------------------------ validity
def test_null_rate_with_a_mode_dependent_skeleton():
    """The skeleton depends on the modes (checked), the outcome series is fixed: P(e >= 20) and P(sup e >= 20) stay
    <= .05 (Ville), for the mixture and the plug-in lambda."""
    rng = np.random.default_rng(2026)
    n_sim, succ_rej, succ_acc = 250, [], []
    fin = {lam: [] for lam in EP.LAMS}
    sup = {lam: [] for lam in EP.LAMS}
    for _ in range(n_sim):
        d = _dataset(rng, 30)
        same_ep = np.r_[d["episode"][1:] == d["episode"][:-1], False]
        rej = LV[d["mode"]] == 0
        succ_rej.append(same_ep[rej].mean() if rej.any() else np.nan)
        succ_acc.append(same_ep[~rej].mean())
        for lam in EP.LAMS:
            r = _log_e(d, lam=lam)
            fin[lam].append(r["log_e"])
            sup[lam].append(r["log_e_max"])
    assert np.nanmean(succ_rej) > np.nanmean(succ_acc) + 0.1          # the skeleton really depends on the modes
    se = np.sqrt(0.05 * 0.95 / n_sim)
    for lam in EP.LAMS:
        assert np.mean(np.array(fin[lam]) >= np.log(20)) <= 0.05 + 2 * se
        assert np.mean(np.array(sup[lam]) >= np.log(20)) <= 0.05 + 2 * se
        assert np.mean(np.exp(fin[lam])) < 1.5                         # E[M_N] = 1 (heavy tail: loose bound)


def test_fe_demeaned_statistic_is_biased_but_the_predictable_residual_is_not():
    """The gap: v2's within-(episode x sgn) demeaning over the REALISED (mode-dependent) unit set shifts the CRT
    statistic under the sharp null; the running past-only residual does not (martingale: E z = 0)."""
    rng = np.random.default_rng(7)
    z_fe, z_pred = [], []
    cfg = EP.EProcConfig(burn=0)
    for _ in range(120):
        d = _dataset(rng, 80)
        strata = d["episode"] * 2 + (d["sgn"] > 0)
        r_fe, _ = residualise(d["y"], None, strata)
        r_pr, _ = EP.predictable_residuals(d["y"], d["pre"], d["episode"], d["sgn"], cfg)
        z_fe.append(_crt_z(d, r_fe))
        z_pred.append(_crt_z(d, r_pr))
    se = 1.0 / np.sqrt(len(z_fe))
    assert np.mean(z_fe) < -4 * se                     # biased (about -0.7 z at 80 episodes)
    assert abs(np.mean(z_pred)) < 3 * se


# ------------------------------------------------------------------------------------------------ power
def test_planted_effect_is_detected():
    rng = np.random.default_rng(99)
    hits = [(_log_e(_dataset(rng, 40, delta=5000.0))["log_e"] >= np.log(1000)) for _ in range(12)]
    assert np.mean(hits) >= 0.75


def test_family_eprocess_on_unit_data():
    planted = CU.build_unit_data(CU.synthetic_records(np.random.default_rng(11), n_ep=8, units_per_ep=60,
                                                      effect=400.0))
    run = EP.run_eprocess_units(planted)
    dec = {(r["family"], r["relation"], r["kpi"]) for r in run["results"] if r["status"] == "declared"}
    assert ("sleep", "nbr", "load") in dec
    got = next(r for r in run["results"] if (r["family"], r["relation"], r["kpi"]) == ("sleep", "nbr", "load"))
    assert got["sign"] == 1 and got["e"] >= run["ebh_rank1_threshold"] / max(1, run["n_declared"])
    null = CU.build_unit_data(CU.synthetic_records(np.random.default_rng(31), n_ep=8, units_per_ep=60))
    rn = EP.run_eprocess_units(null)
    assert rn["n_declared"] == 0
    assert all(r["status"] in ("not_detected", "undetermined") for r in rn["results"])


# ------------------------------------------------------------------------------------------------ fixed-slot units
def test_slot_units_are_keyed_by_the_slot_not_by_the_request():
    from cdd_oran.decision import collect_p as CP
    from cdd_oran.decision import slots_p as SP
    static = {"cells": 3, "neighbours": [[1], [0], [1]], "is_macro": np.array([True, True, False]),
              "knobs": [("carrier", 0), ("carrier", 1), ("sleep", 2), ("prot_min", 0), ("ptx", 0)],
              "xapps": {"ES": [], "SliceGuarantee": [], "PowerES": []}}
    cfg = {("carrier", 0): 2.0, ("carrier", 1): 2.0, ("sleep", 2): 0.0, ("prot_min", 0): 0.0, ("ptx", 0): 0.0}

    def ob(t, reqs):
        return {"t": float(t), "requests": reqs, "config": dict(cfg), "new_reports": [], "static": static,
                "locked": {}, "changes": 0, "churn_cap": None}

    def rq(x, k, cur, prop):
        return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": 0.0}

    pol = CP.RandomizedUnitPolicy(7, tables=CP.PI0_HIGH_NO_RB)
    arb = SP.SlotArbiter(pol, T=60.0, warmup_s=100.0)
    arb(ob(99, [rq("ES", ("sleep", 2), 0.0, 1.0)]))                            # warm-up: no unit
    arb(ob(117, [rq("ES", ("sleep", 2), 0.0, 1.0)]))
    arb(ob(150, [rq("ES", ("sleep", 2), 0.0, 1.0)]))
    arb(ob(163, [rq("ES", ("sleep", 2), 0.0, 1.0)]))
    assert [(u["t0"], u["slot"], u["t_req"], u["n_req"]) for u in arb.units] == [(100.0, 0, 117.0, 2),
                                                                                 (160.0, 1, 163.0, 1)]
    ref = {k: pol({"c": 2, "x": "ES", "x_idx": 0, "t0": 100.0 + 60 * k})[0] for k in range(3)}
    assert [u["mode"] for u in arb.units] == [ref[0], ref[1]]                  # the draw ignores the request time
    tab = arb.slot_table(pol, 3)
    pairs = SP.owner_pairs(static)
    assert pairs == [(0, "ES"), (0, "PowerES"), (0, "SliceGuarantee"), (1, "ES"), (2, "ES")]
    assert len(tab) == 3 * len(pairs)
    s2 = [r for r in tab if (r["c"], r["x"]) == (2, "ES")]
    assert [r["request"] for r in s2] == [True, True, False] and [r["mode"] for r in s2] == [ref[0], ref[1], ref[2]]
