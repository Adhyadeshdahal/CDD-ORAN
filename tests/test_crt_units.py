"""Unit-level CRT (cdd_oran.decision.crt_units) on synthetic e6p-disc-rec/1 records: a planted unit effect is
detected with the right sign, a null gives ~uniform p, the conditional draws follow the logged pi0 tables, the far
relation stays null under a planted nbr effect, the batched / chunked null equals the single-hypothesis one, and the
privileged record fields are never read."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import kstest

from cdd_oran.decision import crt_units as CU

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def planted():
    return CU.build_unit_data(CU.synthetic_records(np.random.default_rng(11), n_ep=8, units_per_ep=60, effect=200.0))


def test_unit_table_targets_are_window_differences_over_the_relation():
    recs = CU.synthetic_records(np.random.default_rng(2), n_ep=2, units_per_ep=10)
    d = CU.build_unit_data(recs)
    from cdd_oran.decision.collect_p import dec
    D = dec(recs[0]["lab_series"]["data"]).astype(float)
    u = recs[0]["units"][0]
    i = 0                                               # first row = first unit of episode 0 (all windows fit)
    t0 = int(u["t0"])
    ue = recs[0]["lab_series"]["fields"].index("ue")
    nbr = [q for q in u["exp"] if q != u["c"]]
    want = D[t0:t0 + 90, ue][:, nbr].sum() - D[t0 - 90:t0, ue][:, nbr].sum()
    assert d.y[("nbr", "load")][i] == pytest.approx(want)
    assert d.pre[("own", "load")][i] == pytest.approx(D[t0 - 90:t0, ue, u["c"]].sum())
    assert d.level[i] == CU.LEVEL[u["mode"]] and d.sgn[i] == np.sign(u["step"])
    assert d.x[i] == CU.LEVEL[u["mode"]] * np.sign(u["step"])
    assert set(d.z) >= {"ctx_cur", "ctx_step", "pre_far_pv", "pre_nbr_load"}
    for k in d.kpis:                                    # own + nbr + far partition the cells
        tot = sum(d.y[(r, k)] for r in ("own", "nbr", "far"))
        assert np.allclose(tot, d.ycell[k].sum(1))


def test_planted_unit_effect_is_detected_with_its_sign(planted):
    cfg = CU.UnitCRTConfig(B=999)
    r = CU.crt_unit_test(planted, "sleep", "nbr", "load", cfg)
    assert r["status"] == "tested" and r["p_mscr"] <= 0.01 and r["p_signed"] <= 0.01 and r["sign"] == 1
    assert r["replica_err"] < 1e-9 and r["joint_err"] == 0.0
    own = CU.crt_unit_test(planted, "sleep", "own", "load", cfg)
    assert own["p_mscr"] <= 0.05 and own["sign"] == -1


def test_far_planted_null_stays_null_and_the_edge_is_declared(planted):
    run = CU.run_crt_units(planted, CU.UnitCRTConfig(B=1999), audit=False, families=("sleep",))
    assert run["power_floor"]["rank1_ok"]
    res = {(r["relation"], r["kpi"]): r for r in run["results"]}
    assert res[("nbr", "load")]["status"] == "declared" and res[("nbr", "load")]["sign"] == 1
    assert res[("nbr", "pv")]["status"] == "declared"
    for k in CU.KPIS:
        assert res[("far", k)]["status"] != "declared"
    e = CU.edges_from_crt(run)
    assert e[("sleep", "nbr", "load")]["declared"] and not e[("sleep", "far", "load")]["declared"]


def test_null_gives_approximately_uniform_p():
    cfg = CU.UnitCRTConfig(B=199)
    ps = []
    for r in range(60):
        d = CU.build_unit_data(CU.synthetic_records(np.random.default_rng([5, r]), n_ep=6, units_per_ep=40))
        t = CU.crt_unit_test(d, "sleep", "nbr", "load", cfg)
        if t["status"] == "tested":
            ps.append(t["p_mscr"])
    ps = np.array(ps)
    assert len(ps) >= 50
    assert 0.35 < ps.mean() < 0.65
    assert (ps <= 0.1).mean() < 0.25
    assert kstest(ps, "uniform").pvalue > 0.01


def test_placebo_is_a_sharp_null_even_with_a_planted_effect():
    recs = CU.synthetic_records(np.random.default_rng(21), n_ep=8, units_per_ep=60, effect=60.0, placebo=True)
    d = CU.build_unit_data(recs)
    assert np.all(d.applied == CU.MODES.index("accept")) and np.any(d.mode != d.applied)
    k0 = CU.placebo_rejection_units(d, CU.UnitCRTConfig(B=199))
    assert k0["n_tested"] > 0 and k0["n_by_declared"] <= 1
    sl = [h for h in k0["per_hypothesis"] if (h["family"], h["relation"], h["kpi"]) == ("sleep", "nbr", "load")][0]
    assert sl["p_mscr"] > 0.01
    with pytest.raises(ValueError):
        CU.placebo_rejection_units(CU.build_unit_data(CU.synthetic_records(np.random.default_rng(1), n_ep=2)))


def test_draws_match_the_logged_pi0_propensities():
    recs = CU.synthetic_records(np.random.default_rng(4), n_ep=4, units_per_ep=50)
    recs[0] = dict(recs[0], pi0_table={**recs[0]["pi0_table"],
                                       "SliceGuarantee": {"accept": 0.7, "half": 0.1, "reject": 0.2}})
    for u in recs[0]["units"]:
        if u["x"] == "SliceGuarantee":
            u["p"] = recs[0]["pi0_table"]["SliceGuarantee"][u["mode"]]
    d = CU.build_unit_data(recs)
    assert d.meta["p_mismatch"] == 0
    pa = CU.PiAssignment(d)
    rows = d.rows_of("prot_min")
    M = pa.conditional_draws(np.random.default_rng([0, CU.CRT_TAG]), "prot_min", 20000, what="mode")
    for sel, tab in ((d.episode[rows] == 0, (0.7, 0.1, 0.2)), (d.episode[rows] > 0, (0.5, 0.2, 0.3))):
        freq = [(M[:, sel] == CU.MODES.index(m)).mean() for m in ("accept", "half", "reject")]
        assert np.allclose(freq, tab, atol=0.01)
    X = pa.conditional_draws(np.random.default_rng([0, CU.CRT_TAG]), "prot_min", 50)
    L = pa.conditional_draws(np.random.default_rng([0, CU.CRT_TAG]), "prot_min", 50, what="level")
    assert np.array_equal(X, L * d.sgn[rows][None, :])
    with pytest.raises(ValueError):
        pa.conditional_draws(np.random.default_rng(0), "prot_min", 5, rows=d.rows_of("sleep"))


def test_chunked_and_batched_nulls_equal_the_single_hypothesis_null(planted):
    a = CU.UnitCRTConfig(B=1500, chunk=1500)
    b = dataclasses.replace(a, chunk=400)
    for h in (("nbr", "load"), ("far", "pv")):
        ra, rb = CU.crt_unit_test(planted, "sleep", *h, a), CU.crt_unit_test(planted, "sleep", *h, b)
        assert ra["p_mscr"] == rb["p_mscr"] and ra["p_signed"] == rb["p_signed"]
    run = CU.run_crt_units(planted, dataclasses.replace(a, B=199), audit=False, families=("ptx",))
    single = CU.crt_unit_test(planted, "ptx", "own", "e", dataclasses.replace(a, B=199))
    batched = next(r for r in run["results"] if (r["relation"], r["kpi"]) == ("own", "e"))
    assert batched["p_mscr"] == single["p_mscr"] and batched["s_obs"] == single["s_obs"]
    assert run["max_joint_err"] == 0.0 and run["max_replica_err"] < 1e-9


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
    recs = CU.synthetic_records(np.random.default_rng(8), n_ep=2, units_per_ep=20)
    clean = CU.build_unit_data(recs)
    for r in recs:
        for k in CU.PRIVILEGED_KEYS:
            r[k] = _Poison(cand={"1": 2})
    d = CU.build_unit_data(recs)
    assert np.array_equal(d.y[("nbr", "load")], clean.y[("nbr", "load")])


def test_carryover_audit_and_power_floor(planted):
    a = CU.carryover_audit(planted, CU.UnitCRTConfig(B=199, audit_B=99), families=("sleep", "ptx"))
    assert a["B"] == 99 and a["n_tested"] > 0
    assert all(r["lag"] == 1 for r in a["results"])
    pf = CU.power_floor(999, 60)
    assert not pf["rank1_ok"] and pf["min_hypotheses_at_floor_to_declare"] == 6
    assert CU.power_floor(9999, 60)["rank1_ok"] and CU.power_floor(9999, 60)["min_B_rank1"] <= 9999
