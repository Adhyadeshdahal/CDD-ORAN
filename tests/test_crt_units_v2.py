"""mscr-crt-units-v2 (cdd_oran.decision.crt_units_v2) on synthetic e6p-disc-rec/1 records: p-values are valid under
a sharp null (the observed modes re-drawn from pi0 with the outcomes fixed), a planted additive effect is detected
with its sign, the design-centred / sgn-stratified statistic fixes the v1 slope's sign error when sgn is confounded
with the outcome trend, p is chunk-invariant, the placebo K0 rule holds, and no privileged record field is read."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import kstest

from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import crt_units_v2 as V2

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def planted():
    return CU.build_unit_data(CU.synthetic_records(np.random.default_rng(11), n_ep=8, units_per_ep=60, effect=200.0))


@pytest.fixture(scope="module")
def null_data():
    return CU.build_unit_data(CU.synthetic_records(np.random.default_rng(31), n_ep=8, units_per_ep=60))


def _redraw_modes(d: CU.UnitData, rng) -> CU.UnitData:
    """A fresh pi0 assignment of every unit with the outcomes held fixed (= the sharp null's randomization)."""
    M = CU.PiAssignment(d).mode_draws(rng, np.arange(d.n), 1)[0]
    return dataclasses.replace(d, mode=M)


def test_regressor_is_design_centred_and_uses_the_v2_dose(null_data):
    d = null_data
    rows = d.rows_of("sleep")
    v = V2.design_regressor(d, rows)
    mu = d.probs[rows] @ V2.LEVEL_V2_ARR
    assert np.allclose(v, d.sgn[rows] * (V2.LEVEL_V2_ARR[d.mode[rows]] - mu))
    half = d.mode[rows] == CU.MODES.index("half")
    assert half.any() and np.allclose(v[half], (d.sgn[rows] * (1.0 - mu))[half])    # half = full dose
    M = CU.PiAssignment(d).mode_draws(np.random.default_rng(0), rows, 20000)
    Vd = d.sgn[rows][None, :] * (V2.LEVEL_V2_ARR[M] - mu[None, :])
    assert np.abs(Vd.mean(0)).max() < 0.03                      # E_pi0[v] = 0 per unit


def test_residualise_fe_plus_pre_and_degenerate_pre_guard():
    rng = np.random.default_rng(0)
    st = np.repeat(np.arange(6), 20)
    pre = rng.normal(size=120)
    y = 3.0 * pre + st * 10.0 + rng.normal(size=120)
    r, info = V2.residualise(y, pre, st)
    X = np.column_stack([(st[:, None] == np.arange(6)[None, :]).astype(float), pre])
    want = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert info["resid"] == "fe+pre" and np.allclose(r, want)
    r2, info2 = V2.residualise(y, np.ones(120), st)            # constant pre -> FE only
    assert info2["resid"] == "fe"
    assert np.allclose(r2, y - np.bincount(st, y)[st] / 20.0)
    r3, info3 = V2.residualise(y, st * 1.0, st)                # pre constant within strata -> FE only
    assert info3["resid"] == "fe" and np.allclose(r3, r2)


def test_p_values_are_valid_under_a_sharp_null(null_data):
    cfg = V2.UnitCRTConfigV2(B=199)
    ps = []
    for i in range(200):
        d = _redraw_modes(null_data, np.random.default_rng([77, i]))
        t = V2.crt_unit_test_v2(d, "sleep", "nbr", "load", cfg, split=i)
        if t["status"] == "tested":
            ps.append(t["p"])
    ps = np.array(ps)
    assert len(ps) >= 190
    assert 0.42 < ps.mean() < 0.58
    assert (ps <= 0.05).mean() <= 0.09 and (ps <= 0.10).mean() <= 0.16
    assert kstest(ps, "uniform").pvalue > 0.01


def test_planted_additive_effect_is_detected_with_its_sign(planted):
    run = V2.run_crt_units_v2(planted, V2.UnitCRTConfigV2(B=1999), families=("sleep",))
    res = {(r["relation"], r["kpi"]): r for r in run["results"]}
    nl = res[("nbr", "load")]
    assert nl["status"] == "declared" and nl["sign"] == 1 and nl["z_approx"] > 3
    assert res[("nbr", "pv")]["status"] == "declared" and res[("nbr", "pv")]["sign"] == 1
    assert res[("own", "load")]["sign"] == -1 and res[("own", "load")]["p"] <= 0.01
    for k in CU.KPIS:
        assert res[("far", k)]["status"] != "declared"
    e = V2.edges_from_crt_v2(run)
    assert e[("sleep", "nbr", "load")]["declared"] and e[("sleep", "nbr", "load")]["sign"] == 1
    assert not e[("sleep", "far", "load")]["declared"]
    assert run["power_floor"]["rank1_ok"]


def test_sign_fix_when_sgn_is_confounded_with_the_outcome_trend(null_data):
    """y = K sgn - e * LEVEL_V2 * sgn: a knob-value increase LOWERS y (dir effect -e), but the request direction sgn
    (not randomized) carries a large trend K sgn. v1's within-episode slope of y on x = level * sgn picks up K
    (x and sgn correlate since E[level] > 0) and gets the WRONG sign; v2 (design-centred, episode x sgn FE) is right."""
    d = null_data
    rng = np.random.default_rng(5)
    rows = d.rows_of("sleep")
    key = ("nbr", "load")
    y = d.y[key].copy()
    lv2 = V2.LEVEL_V2_ARR[d.mode]
    y = 50.0 * d.sgn - 10.0 * lv2 * d.sgn + rng.normal(0, 2.0, d.n)
    dd = dataclasses.replace(d, y={**d.y, key: y})
    beta_v1 = CU._beta(dd.x[rows], y[rows], dd.episode[rows])
    assert beta_v1 > 0                                           # v1: wrong sign
    r = V2.crt_unit_test_v2(dd, "sleep", *key, V2.UnitCRTConfigV2(B=999))
    assert r["sign"] == -1 and r["p"] <= 0.01 and r["beta"] == pytest.approx(-10.0, rel=0.25)


def test_chunk_invariance(planted):
    a = V2.UnitCRTConfigV2(B=1500, chunk=1500)
    b = dataclasses.replace(a, chunk=400)
    c = dataclasses.replace(a, max_chunk_bytes=8 * 150 * 37)   # forces a tiny chunk (37 draws)
    for h in (("nbr", "load"), ("far", "pv"), ("own", "e")):
        ra, rb, rc = (V2.crt_unit_test_v2(planted, "sleep", *h, x) for x in (a, b, c))
        assert rc["chunk"] < rb["chunk"] < ra["chunk"]
        assert ra["p"] == rb["p"] == rc["p"]
        assert ra["z_approx"] == pytest.approx(rc["z_approx"], rel=1e-9)
    run = V2.run_crt_units_v2(planted, dataclasses.replace(a, B=199), families=("ptx",))
    single = V2.crt_unit_test_v2(planted, "ptx", "own", "e", dataclasses.replace(a, B=199))
    batched = next(r for r in run["results"] if (r["relation"], r["kpi"]) == ("own", "e"))
    assert batched["p"] == single["p"] and batched["s_obs"] == pytest.approx(single["s_obs"], rel=1e-12)


def test_streams_differ_from_v1(planted):
    """The v2 RNG key carries a version component: its draws are not v1's."""
    fi = CU.FAMILIES.index("sleep")
    u1 = np.random.default_rng([0, CU.CRT_TAG, fi, 0, 0]).random(5)
    u2 = V2._stream(V2.UnitCRTConfigV2(), fi, 0, 0).random(5)
    assert not np.allclose(u1, u2)


def test_placebo_k0_rule():
    recs = CU.synthetic_records(np.random.default_rng(21), n_ep=8, units_per_ep=60, effect=60.0, placebo=True)
    d = CU.build_unit_data(recs)
    assert np.all(d.applied == CU.MODES.index("accept")) and np.any(d.mode != d.applied)
    k0 = V2.placebo_rejection_units_v2(d, V2.UnitCRTConfigV2(B=199))
    assert k0["n_tested"] > 0 and k0["n_by_declared"] <= 1 and k0["pass"]
    assert k0["k_max"] == 7 if k0["n_tested"] == 60 else k0["k_max"] >= 1
    sl = [h for h in k0["per_hypothesis"] if (h["family"], h["relation"], h["kpi"]) == ("sleep", "nbr", "load")][0]
    assert sl["p"] > 0.01
    with pytest.raises(ValueError):
        V2.placebo_rejection_units_v2(CU.build_unit_data(CU.synthetic_records(np.random.default_rng(1), n_ep=2)))


def test_support_rule_as_v1(planted):
    one = planted.subset(planted.episode < 2)
    r = V2.crt_unit_test_v2(one, "sleep", "nbr", "load", V2.UnitCRTConfigV2(B=99))
    assert r["status"] == "undetermined" and r["reason"] == "support"


class _Poison(dict):
    def __getitem__(self, k):
        raise AssertionError("privileged field read")

    def get(self, *a, **k):
        raise AssertionError("privileged field read")

    def __iter__(self):
        raise AssertionError("privileged field read")

    def items(self):
        raise AssertionError("privileged field read")


def test_privileged_fields_are_never_read():
    recs = CU.synthetic_records(np.random.default_rng(8), n_ep=4, units_per_ep=40)
    cfg = V2.UnitCRTConfigV2(B=99, min_episodes=1)
    clean = V2.run_crt_units_v2(CU.build_unit_data(recs), cfg, families=("sleep",))
    for r in recs:
        for k in CU.PRIVILEGED_KEYS:
            r[k] = _Poison(cand={"1": 2})
    pois = V2.run_crt_units_v2(CU.build_unit_data(recs), cfg, families=("sleep",))
    assert [r["p"] for r in pois["results"]] == [r["p"] for r in clean["results"]]
    src = open(V2.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[2]                               # code after the module docstring
    for k in CU.PRIVILEGED_KEYS:
        assert k not in body


def test_carryover_audit_is_optional_and_descriptive(planted):
    run = V2.run_crt_units_v2(planted, V2.UnitCRTConfigV2(B=99), families=("sleep",))
    assert "audit" not in run
    a = V2.carryover_audit_v2(planted, V2.UnitCRTConfigV2(B=99, audit_B=99), families=("sleep", "ptx"))
    assert a["B"] == 99 and all(r["lag"] == 1 for r in a["results"])
