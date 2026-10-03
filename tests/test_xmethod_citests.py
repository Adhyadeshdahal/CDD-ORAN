"""Tests for the independence-test adapters (cdd_oran/xmethod/methods: mscr, pcorr, pdcor, rcot2, cmi_knn)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.xmethod import api
from cdd_oran.xmethod.methods import _citests_synth as synth
from cdd_oran.xmethod.methods._citests_common import (
    PLACEBO,
    SequentialP,
    by_families,
    pcorr_given_z,
    prepare,
    sequential_p_from_draws,
    standardize,
    tau_from_scores,
)
from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
from cdd_oran.xmethod.methods.mscr import MSCRMethod
from cdd_oran.xmethod.methods.pcorr import PCorrMethod
from cdd_oran.xmethod.methods.pdcor import PDCorMethod
from cdd_oran.xmethod.methods.rcot2 import RCoT2Method, block_perm_sequential

DEV0 = 3_000_000


def _small(kind="planted", n=300, seed=DEV0):
    return synth.make(kind, n, seed)


# ------------------------------------------------------------------------------------------- common core
def test_sequential_p_rules():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=500)
    for obs in (-3.0, 0.0, 1.5, 9.0):
        seq = SequentialP(obs, 500, 20)
        for v in draws:
            if seq.add(v):
                break
        assert (seq.p, seq.k) == sequential_p_from_draws(obs, draws, 20)
    # h reached -> h / k; never reached -> (1 + count) / (B + 1); h=None -> fixed-B Monte Carlo p
    assert sequential_p_from_draws(-1.0, np.zeros(100), 20) == (20 / 20, 20)
    assert sequential_p_from_draws(5.0, np.zeros(100), 20) == (1 / 101, 100)
    assert sequential_p_from_draws(0.5, np.r_[np.ones(30), np.zeros(70)], None) == (31 / 101, 100)


def test_tau_rule_conformal_r29():
    # M = 3: ceil(4 * .95) = 4 > M -> the max; M = 39: ceil(40 * .95) = 38th smallest; M = 0 -> -inf
    assert tau_from_scores([0.1, 0.5, 0.3, np.nan]) == 0.5
    assert tau_from_scores([]) == np.inf and tau_from_scores([np.nan]) == np.inf      # M = 0: +inf
    s = np.arange(1.0, 40.0)
    assert tau_from_scores(s) == 38.0 and int(np.sum(s > tau_from_scores(s))) == 1
    s = np.arange(1.0, 101.0)                                   # ceil(101 * .95) = 96 -> 96th smallest
    assert tau_from_scores(s) == 96.0 and int(np.sum(s > tau_from_scores(s))) == 4


def test_by_families_are_separate():
    p = np.array([1e-4, 0.5, 0.5, 1e-4, 0.5])
    fam = ("action_kpi", "action_kpi", "action_kpi", "kpi_kpi", "kpi_kpi")
    d = by_families(p, fam)
    assert d.tolist() == [True, False, False, True, False]
    # NaN (not scorable) is never declared and does not enter the family size
    assert by_families(np.array([np.nan, 1e-6]), ("action_kpi", "action_kpi")).tolist() == [False, True]


def test_prepare_layout_nan_lag_and_context():
    ds, _ = _small()
    prep = prepare(ds)
    assert prep.S.shape == (ds.n, len(ds.action_names) + len(ds.kpi_names))
    assert prep.family.count("action_kpi") == len(ds.action_names) * len(ds.kpi_names)
    lag = ds.X_kpi_lag.copy()
    lag[:, 1] = np.nan
    ctx = np.random.default_rng(1).normal(size=(ds.n, 1))
    cands = tuple(c for c in ds.candidates if c[0] != "K1")
    ds2 = dataclasses.replace(ds, X_kpi_lag=lag, context=ctx, candidates=cands, meta={"context_names": ("Z",)})
    p2 = prepare(ds2)
    assert "K1" not in p2.S_names and p2.S_names[-1] == "ctx:Z" and p2.n_ctx == 1
    bad = dataclasses.replace(ds2, candidates=ds2.candidates + (("K1", "K0"),))
    with pytest.raises(ValueError):
        prepare(bad)


def test_pcorr_given_z_matches_e1_partial_correlation():
    from cdd_oran.e1slice.discovery_v2 import partial_correlation_scores

    ds, _ = _small()
    prep = prepare(ds)
    xs, ys = standardize(prep.S), standardize(prep.Y)
    rho, df = pcorr_given_z(xs, ys, prep.pairs)
    ref = partial_correlation_scores(xs, ys)
    np.testing.assert_allclose(rho, [ref[j, i] for i, j in prep.pairs], atol=1e-10)
    assert df[0] == ds.n - 2 - (xs.shape[1] - 1)


# ------------------------------------------------------------------------- native mode == frozen code
def test_mscr_native_equals_discover_mscr():
    from cdd_oran.discovery.mscr import MSCRConfig, discover_mscr

    ds, _ = _small(n=400)
    m = MSCRMethod()
    cfg = {**m.native_config(), "n_perm": 199, "n_jobs": 1}
    r = m.run(ds, cfg)
    prep = prepare(ds)
    tested = sorted({i for i, _ in prep.pairs})
    x = prep.S[:, tested + [i for i in range(prep.S.shape[1]) if i not in tested]]
    from cdd_oran.xmethod.methods._citests_common import method_seed

    ref = discover_mscr(x, prep.Y, len(tested), seed=method_seed(ds.seed, m.method_key),
                        config=MSCRConfig(n_perm=199), n_jobs=1, device="cpu")
    pos = {i: c for c, i in enumerate(tested)}
    for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
        assert e.p == ref.pvals[j, pos[i]]
        assert e.score == ref.s_star[j, pos[i]]


def test_pdcor_fast_loop_matches_frozen_loop():
    ds, _ = _small(n=80)
    ds = dataclasses.replace(ds, candidates=ds.candidates[:6] + ds.candidates[-3:])
    m = PDCorMethod()
    cfg = {**m.native_config(), "b_perm": 99}
    prep = prepare(ds)
    fast = m._test(prep, ds, cfg)
    ref = m._test(prep, ds, cfg, _loop="frozen")
    np.testing.assert_array_equal(fast["score"], ref["score"])
    np.testing.assert_allclose(fast["p"], ref["p"], atol=1.5 / 100)    # fp ties may move one count at most
    assert np.mean(fast["p"] == ref["p"]) >= 0.8


def test_rcot2_native_equals_frozen_block_perm():
    from cdd_oran.e2slice.discovery_rcot import rcot_pvalue_block_perm
    from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2

    rng = np.random.default_rng(5)
    z = rng.normal(size=(150, 3))
    x = z[:, 0] + rng.normal(size=150)
    y = 0.5 * x + z[:, 1] + rng.normal(size=150)
    cfg = dataclasses.replace(frozen_config_v2(), block_perm_reps=49)
    for yy in (y, rng.normal(size=150)):
        a = rcot_pvalue_block_perm(x, yy, z, cfg, 123)
        b = block_perm_sequential(x, yy, z, cfg, 123, h=None)
        assert a == b[:3] and b[3] == 49


def test_cmi_knn_native_equals_tigramite():
    from tigramite.independence_tests.cmiknn import CMIknn

    rng = np.random.default_rng(6)
    z = rng.normal(size=(120, 2))
    x = z[:, :1] + rng.normal(size=(120, 1))
    y = 0.3 * x + rng.normal(size=(120, 1))
    m = CMIKnnMethod()
    cfg = {**m.native_config(), "sig_samples": 39, "workers": 1}
    ours = m.make_test(cfg, 77).run_test_raw(x, y, z)
    ref = CMIknn(knn=0.2, shuffle_neighbors=5, significance="shuffle_test", transform="ranks", workers=1,
                 sig_samples=39, seed=77).run_test_raw(x, y, z)
    assert ours == ref


# ----------------------------------------------------------------------------------- Method behaviour
FAST = {"mscr": {"n_perm": 299, "bc_h": 20, "n_jobs": 1}, "pcorr": {}, "pdcor": {"b_perm": 99},
        "rcot2": {"block_perm_reps": 99}, "cmi_knn": {"sig_samples": 49, "workers": 1}}


@pytest.mark.parametrize("cls", [MSCRMethod, PCorrMethod, PDCorMethod, RCoT2Method, CMIKnnMethod])
def test_result_contract(cls):
    ds, _ = _small(n=150 if cls in (PDCorMethod, CMIKnnMethod) else 400)
    if cls in (PDCorMethod, CMIKnnMethod, RCoT2Method):
        ds = dataclasses.replace(ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", PLACEBO, "K3")))
    m = cls()
    r = m.run(ds, FAST[m.name])
    assert isinstance(r, api.Result) and r.method == m.name and r.cpu_s >= 0
    assert [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert all(e.sign in (-1, 0, 1) for e in r.edges)
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)   # analytic lpd4 may give 0 (RCIT clips 1-LPB4 at 0)
    assert r.notes["sign_rule"] == ("native" if m.name == "pcorr" else "pcorr_given_Z")
    assert len(r.notes["declared_native"]) == len(r.edges) and len(r.notes["family_of_edge"]) == len(r.edges)
    # determinism
    r2 = m.run(ds, FAST[m.name])
    assert [(e.score, e.p) for e in r.edges] == [(e.score, e.p) for e in r2.edges]


def test_pcorr_finds_linear_planted_edges_with_signs():
    ds, truth = _small(n=2000)
    r = PCorrMethod().run(ds, {})
    dec = {(e.source, e.target): e.sign for e in r.edges if e.declared}
    assert dec.get(("P0", "K0")) == 1 and dec.get(("P1", "K1")) == -1 and dec.get(("K3", "K3")) == 1
    assert not any(s == PLACEBO for s, _ in dec)


def test_mscr_finds_gated_edge():
    ds, _ = _small(n=2000)
    ds = dataclasses.replace(ds, candidates=tuple(c for c in ds.candidates if c[1] == "K2" and c[0].startswith("P")))
    r = MSCRMethod().run(ds, {"n_perm": 999, "bc_h": 20, "n_jobs": 1})
    dec = {e.source for e in r.edges if e.declared}
    assert {"P2", "P3"} <= dec and PLACEBO not in dec


def test_tune_returns_tau_from_placebo_only():
    devs = [synth.make("null", 300, DEV0 + s)[0] for s in range(3)]
    m = PCorrMethod()
    cfg = m.tune(devs)
    sc = sorted((e.score for d in devs for e in m.run(d, {}).edges if e.source == PLACEBO), reverse=True)
    # R-29: M = 12 scores, ceil(13 * .95) = 13 > M -> tau = the largest
    assert cfg["tau"] == sc[0] and cfg["n_placebo_scores"] == 3 * len(synth.KPIS)
    r = m.run(devs[0], cfg)
    assert len(r.notes["declared_tau"]) == len(r.edges)
    with pytest.raises(ValueError):
        m.tune([devs[0], synth.make("null", 400, DEV0)[0]])


def test_placebo_conf_is_own_family_and_not_a_conditioner():
    ds, _ = _small(n=400)
    rng = np.random.default_rng(9)
    conf = rng.uniform(-1, 1, size=(ds.n, 1))
    names = ds.action_names + ("P_placebo_conf",)
    cands = ds.candidates + tuple(("P_placebo_conf", k) for k in ds.kpi_names)
    ds2 = dataclasses.replace(ds, action_names=names, X_action=np.c_[ds.X_action, conf],
                              designs=ds.designs + (ds.designs[0],), candidates=cands)
    prep = prepare(ds2)
    assert prep.family[-1] == "placebo_conf" and prep.exclude_z == (len(ds.action_names),)
    for m, cfg in ((PCorrMethod(), {}), (MSCRMethod(), {"n_perm": 199, "n_jobs": 1})):
        r1, r2 = m.run(ds, cfg), m.run(ds2, cfg)
        # adding the diagnostic column leaves every original test unchanged
        assert [(e.score, e.p, e.sign, e.declared) for e in r1.edges] == \
            [(e.score, e.p, e.sign, e.declared) for e in r2.edges[: len(r1.edges)]]


def test_rcot2_lpd4_null_and_frozen_hbe_match_momentchi2():
    from momentchi2 import sw

    from cdd_oran.e2slice.discovery_rcot import _hbe_pvalue
    from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2
    from cdd_oran.xmethod.methods.rcot2 import _momentchi2, rcot_pvalue_lpd4

    _, lpb4 = _momentchi2()

    rng = np.random.default_rng(11)
    prodc = rng.normal(size=(400, 25)) @ rng.normal(size=(25, 25)) * 0.1
    prodc -= prodc.mean(0)
    w = np.linalg.eigvalsh(prodc.T @ prodc / 400)
    w = w[w > 1e-12]
    for stat in (0.5, w.sum(), 3 * w.sum()):
        # finding: the frozen e2slice "_hbe_pvalue" is the Satterthwaite-Welch 2-moment gamma (momentchi2 sw)
        assert abs(_hbe_pvalue(stat, prodc) - (1 - sw(w, stat))) < 1e-12
        assert 0 <= 1 - lpb4(w, stat) <= 1
    z = rng.normal(size=(300, 3))
    x = z[:, 0] + rng.normal(size=300)
    st, p, g, how = rcot_pvalue_lpd4(x, 0.8 * x + rng.normal(size=300), z, dataclasses.replace(frozen_config_v2(), dz=100), 5)
    assert not g and how in ("lpb4", "hbe_fallback") and p < 1e-3


def test_pdcor_authors_null_statistic_matches_dcor():
    import dcor

    from cdd_oran.xmethod.methods.pdcor import PDCorMethod

    ds, _ = _small(n=120)
    ds = dataclasses.replace(ds, candidates=(("P0", "K0"), ("P_placebo", "K1")))
    for arm in ("native", "eq"):
        r = PDCorMethod().run(ds, {"b_perm": 99, "arm": arm})
        prep = prepare(ds, arm)
        xs, ys = standardize(prep.S), standardize(prep.Y)
        for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
            ref = float(dcor.partial_distance_correlation(xs[:, i], ys[:, j], np.delete(xs, i, axis=1)))
            assert abs(e.score - ref) < 1e-10
        assert r.edges[0].p < 0.05 and r.config["null"] == "proj_perm" and r.notes["arm"] == arm


# ------------------------------------------------------------------------------------------- arms (R-17 / R-18)
def test_arm_defaults():
    for cls in (MSCRMethod, PCorrMethod, PDCorMethod, RCoT2Method, CMIKnnMethod):
        assert cls().native_config()["arm"] == "native"
    ds, _ = _small(n=200)
    assert PCorrMethod().run(ds, {}).notes["arm"] == "eq"          # primary = equal information
    with pytest.raises(ValueError):
        prepare(ds, "bogus")


def test_eq_arm_layout_on_harness():
    from cdd_oran.xmethod.covariates import design_covariates
    from cdd_oran.xmethod.worlds import generate_dataset

    def canon(nm):          # helper names -> Prepared.S_names of the R-3 columns
        return nm.removeprefix("concurrent:").removeprefix("lag_kpi:")

    for w, reg in (("E2", "R2"), ("E4", "R3"), ("E4", "R4"), ("E5", "R1")):
        ds, _ = generate_dataset(w, reg, 300, DEV0 + 2)
        nat, eq = prepare(ds, "native"), prepare(ds, "eq")
        _, _, mask = design_covariates(ds)
        assert eq.S.shape[0] == mask.sum() == eq.Y.shape[0] and eq.S_names[:nat.S.shape[1]] == nat.S_names
        np.testing.assert_array_equal(eq.S[:, :nat.S.shape[1]], nat.S[mask])
        assert eq.pairs == nat.pairs
        conf = {c for c, nm in enumerate(eq.S_names) if "P_placebo_conf" in nm}
        for i in sorted({i for i, _ in nat.pairs}):
            nm_i = nat.S_names[i]
            if nm_i not in ds.action_names:
                continue
            # R-18: native = helper R-3 set (concurrent "all"), minus the diagnostic P_placebo_conf (R-10)
            cn, _, _ = design_covariates(ds, False, False, focal=nm_i, concurrent="all")
            want = {canon(c) for c in cn} - ({"P_placebo_conf"} if nm_i != "P_placebo_conf" else set())
            assert {nat.S_names[c] for c in nat.z_cols(i)} == want
            # R-25: eq = the WHOLE helper set (concurrent "designed"), P_placebo_conf columns only for itself
            cn, _, _ = design_covariates(ds, focal=nm_i, concurrent="designed")
            want = {canon(c) for c in cn}
            if nm_i != "P_placebo_conf":
                want = {c for c in want if "P_placebo_conf" not in c}
            assert {eq.S_names[c] for c in eq.z_cols(i)} == want
            if nm_i != "P_placebo_conf":
                assert not conf & set(eq.z_cols(i))
    eq2 = prepare(generate_dataset("E2", "R2", 300, DEV0)[0], "eq")
    assert "sp:P0" in {eq2.S_names[c] for c in eq2.z_cols(0)}               # R2: the focal setpoint is in Z


def test_eq_arm_pcorr_matches_ols():
    from scipy import stats

    from cdd_oran.xmethod.worlds import generate_dataset

    ds, _ = generate_dataset("E2", "R2", 400, DEV0 + 3)
    ds = dataclasses.replace(ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", "P_placebo", "K1")))
    r = PCorrMethod().run(ds, {"arm": "eq"})
    prep = prepare(ds, "eq")
    for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
        X = np.column_stack([np.ones(prep.S.shape[0]), prep.S[:, i], prep.S[:, prep.z_cols(i)]])
        beta, *_ = np.linalg.lstsq(X, prep.Y[:, j], rcond=None)
        res = prep.Y[:, j] - X @ beta
        dof = X.shape[0] - X.shape[1]
        se = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[1, 1])
        p = 2 * stats.t.sf(abs(beta[1] / se), dof)
        assert abs(e.p - p) <= 1e-8 * max(p, 1e-300) + 1e-14 and e.sign == np.sign(beta[1])
    assert r.notes["n_rows_dropped"] == 2 and r.notes["covariates_helper"] == "cdd_oran.xmethod.covariates"


# ------------------------------------------------------------------------------------------- R-22 exact fits
def test_r22_deterministic_world_not_testable():
    from cdd_oran.xmethod.worlds import generate_dataset

    ds, _ = generate_dataset("E1", "R1", 300, DEV0 + 4)
    for arm in ("eq", "native"):
        r = PCorrMethod().run(ds, {"arm": arm})
        assert set(r.notes["not_testable"].values()) == {"exact fit: deterministic world under Z"}
        assert len(r.notes["not_testable_edges"]) == len(ds.candidates)
        assert all(e.p is None and np.isnan(e.score) and not e.declared and e.sign == 0 for e in r.edges)
    ds2, _ = generate_dataset("E2", "R1", 300, DEV0 + 4)
    r2 = PCorrMethod().run(ds2, {})
    assert "not_testable" not in r2.notes and r2.notes["exact_fit_rr_min_testable"] > 1e-3


def test_r22_partial_degenerate_alignment():
    ds, _ = _small(n=150)
    Y = ds.Y.copy()
    Y[:, 0] = 0.7 * ds.X_action[:, 0] - 0.2 * ds.X_kpi_lag[:, 1]          # K0 an exact linear fit
    ds = dataclasses.replace(ds, Y=Y, candidates=(("P0", "K0"), ("P0", "K1"), ("P_placebo", "K0"), ("K3", "K3")))
    r = PDCorMethod().run(ds, {"b_perm": 99})
    dead = [e.target == "K0" for e in r.edges]
    assert [e.p is None for e in r.edges] == dead and [list(x) for x in r.notes["not_testable_edges"]] == \
        [["P0", "K0"], ["P_placebo", "K0"]]
    assert len(r.notes["mc_draws"]) == len(r.edges) and len(r.notes["signed_pdcor"]) == len(r.edges)
    assert [v is None for v in r.notes["mc_draws"]] == dead and len(r.notes["declared_native"]) == len(r.edges)
    assert r.edges[3].p is not None and r.edges[3].p < 0.05


# ------------------------------------------------------------------------------------------- cmi_knn torch backend
@pytest.mark.parametrize("world,regime", [("E2", "R2"), ("E4", "R3")])
def test_cmi_knn_torch_backend_equals_tigramite(world, regime):
    from cdd_oran.xmethod.worlds import generate_dataset

    ds, _ = generate_dataset(world, regime, 260, DEV0 + 8, kappa=0.25)
    prep = prepare(ds, "eq")
    m = CMIKnnMethod()
    for i, j in prep.pairs[:2]:
        res = []
        for backend in ("cpu", "torch"):
            t = m.make_test({**m.default_config(), "sig_samples": 30, "bc_h": None, "backend": backend,
                             "device": "cpu", "workers": 1, "chunk_rows": 100}, 5)
            res.append(t.run_test_raw(prep.S[:, [i]], prep.Y[:, [j]], prep.z_of(prep.S, i)))
        assert res[0] == res[1]                                    # statistic and p bit-identical
    assert t.knn_counts.hits > 0                                   # the exact cache was used


def test_eq_arm_equals_classic_cond_set():
    """R-25 / R-28 / R-33 / R-37: citests' eq and eq_min sets == classic's audited cond_set exactly."""
    from cdd_oran.xmethod.methods._classic_common import cond_set
    from cdd_oran.xmethod.worlds import generate_dataset

    def canon(nm):
        return nm.removeprefix("concurrent:").removeprefix("lag_kpi:")

    for w, reg in (("E2", "R2"), ("E4", "R3"), ("E4", "R4"), ("E5", "R1")):
        ds, _ = generate_dataset(w, reg, 300, DEV0 + 9, kappa=0.25)
        for arm in ("eq", "eq_min"):
            prep = prepare(ds, arm)
            for i in sorted({i for i, _ in prep.pairs}):
                nm = prep.S_names[i]
                if nm in ds.action_names:
                    cs = cond_set(ds, "action", ds.action_names.index(nm), arm)
                else:
                    cs = cond_set(ds, "kpi", ds.kpi_names.index(nm), arm)
                got = {prep.S_names[c] for c in prep.z_cols(i)}
                assert got == {canon(c) for c in cs.names} - set(prep.eq_dropped), (w, reg, arm, nm)


# ------------------------------------------------------------------------------------------- arm eq_min (R-33)
def test_eq_min_layout():
    from cdd_oran.xmethod.worlds import generate_dataset

    for w, reg in (("E2", "R2"), ("E2", "R1"), ("E4", "R3"), ("E5", "R2")):
        ds, _ = generate_dataset(w, reg, 300, DEV0 + 11, kappa=0.25)
        nat, em = prepare(ds, "native"), prepare(ds, "eq_min")
        assert em.S.shape[0] == ds.n and em.pairs == nat.pairs and em.arm == "eq_min"
        for i in sorted({i for i, _ in nat.pairs}):
            nm = nat.S_names[i]
            want = {nat.S_names[c] for c in nat.z_cols(i)}
            if nm in ds.action_names and ds.designs[ds.action_names.index(nm)].kind == "dither":
                want |= {f"sp:{nm}"}                       # R2: the focal setpoint only
            assert {em.S_names[c] for c in em.z_cols(i)} == want, (w, reg, nm)
            if nm != "P_placebo_conf":
                assert not any("P_placebo_conf" in em.S_names[c] for c in em.z_cols(i))   # R-37


@pytest.mark.parametrize("cls", [MSCRMethod, PCorrMethod, PDCorMethod, RCoT2Method, CMIKnnMethod])
def test_eq_min_runs(cls):
    from cdd_oran.xmethod.worlds import generate_dataset

    ds, _ = generate_dataset("E2", "R2", 200, DEV0 + 12, kappa=0.25)
    ds = dataclasses.replace(ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", PLACEBO, "K1")))
    r = cls().run(ds, {**FAST[cls.name], "arm": "eq_min"})
    assert r.notes["arm"] == "eq_min" and [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)
    if cls is PCorrMethod:                                       # == OLS t-test with Z = native + sp:<focal>
        from scipy import stats

        prep = prepare(ds, "eq_min")
        for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
            X = np.column_stack([np.ones(ds.n), prep.S[:, i], prep.S[:, prep.z_cols(i)]])
            beta, *_ = np.linalg.lstsq(X, prep.Y[:, j], rcond=None)
            res = prep.Y[:, j] - X @ beta
            dof = X.shape[0] - X.shape[1]
            se = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[1, 1])
            assert abs(e.p - 2 * stats.t.sf(abs(beta[1] / se), dof)) < 1e-9


def test_tau_is_the_shared_placebo_tau():
    """R-29 / audit L1: one implementation (score.placebo_tau); tune() and tau_from_scores both use it."""
    from cdd_oran.xmethod import score
    from cdd_oran.xmethod.methods._citests_common import tau_from_results

    devs = [synth.make("null", 200, DEV0 + s)[0] for s in range(2)]
    res = [PCorrMethod().run(d, {"arm": "native"}) for d in devs]
    sc = [e.score for r in res for e in r.edges if e.source == PLACEBO]
    out = tau_from_results(res)
    assert out["tau"] == score.placebo_tau(res, alpha=0.05) == tau_from_scores(sc, 0.05)
    assert out["n_placebo_scores"] == len(sc) and out["tau_rule"] == score.tau_rule_name(0.05)
    assert tau_from_scores(np.arange(1.0, 20.0)) == 19.0        # (19 + 1) * .95 = 19 exactly: the 19th smallest


# ------------------------------------------------------------------------------------------- audit L2 / L3 / L4
def _stub(kind):
    from cdd_oran.xmethod.methods._citests_common import CITestBase

    class Stub(CITestBase):
        name, version, method_key = "stub", "0", 9

        def _test(self, prep, data, cfg):
            m = len(prep.pairs)
            if kind == "stop":
                raise ValueError("rank-deficient source design")
            p = np.full(m, 1e-6)
            p[0] = np.nan
            return {"score": np.ones(m), "p": p, "not_testable": {0: "guard: test"}}

    return Stub()


def test_guard_and_stop_listed_as_not_testable():
    ds, _ = _small(n=200)
    r = _stub("guard").run(ds, {"arm": "native"})
    first = "{}->{}".format(*ds.candidates[0])
    assert r.notes["not_testable"] == {first: "guard: test"} and r.notes["not_testable_edges"] == [list(ds.candidates[0])]
    e0 = r.edges[0]
    assert e0.p is None and np.isnan(e0.score) and e0.sign == 0 and not e0.declared
    assert all(e.declared for e in r.edges[1:] if r.notes["family_of_edge"][r.edges.index(e)] == "action_kpi")
    r = _stub("stop").run(ds, {"arm": "native"})
    assert len(r.notes["not_testable"]) == len(ds.candidates) and r.notes["stop"].startswith("rank")
    assert all(v.startswith("STOP: rank") for v in r.notes["not_testable"].values())
    assert all(e.p is None and not e.declared for e in r.edges)


def test_mscr_bank_stream_per_column_group(monkeypatch):
    """audit L4: group 0 keeps the frozen [seed, j, n_perm] stream; the R3 / R4 P_placebo_conf group gets its own."""
    import cdd_oran.xmethod.methods.mscr as mscr_mod
    from cdd_oran.xmethod.worlds import generate_dataset

    keys = []
    real = np.random.default_rng

    class _NP:
        def __getattr__(self, k):
            return getattr(np, k)

    class _Random:
        def __getattr__(self, k):
            return getattr(np.random, k)

        def default_rng(self, key=None):
            keys.append(list(key) if isinstance(key, list) else key)
            return real(key)

    fake = _NP()
    fake.random = _Random()
    monkeypatch.setattr(mscr_mod, "np", fake)
    ds, _ = generate_dataset("E4", "R3", 300, DEV0 + 13, kappa=0.25)
    MSCRMethod().run(ds, {**FAST["mscr"], "arm": "eq"})
    lists = [k for k in keys if isinstance(k, list)]
    assert any(len(k) == 3 for k in lists) and any(len(k) == 4 and k[3] == 1 for k in lists)


def test_cmi_knn_production_backend_is_torch():
    """F2-GPU passed (Kaggle T4, 6 worlds, exact): torch is the production default; native_config = tigramite."""
    from cdd_oran.xmethod.methods.cmi_knn import default_device

    m = CMIKnnMethod()
    assert m.default_config()["backend"] == "torch" and m.native_config()["backend"] == "cpu"
    ds, _ = _small(n=150)
    ds = dataclasses.replace(ds, candidates=(("P0", "K0"),))
    r = m.run(ds, {**FAST["cmi_knn"], "arm": "native"})
    assert r.notes["backend"] == "torch" and r.notes["device"] == default_device()
    r_cpu = m.run(ds, {**FAST["cmi_knn"], "arm": "native", "backend": "cpu"})
    assert (r.edges[0].score, r.edges[0].p) == (r_cpu.edges[0].score, r_cpu.edges[0].p)
