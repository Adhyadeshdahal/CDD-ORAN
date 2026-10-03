"""pcorr_hac adapter (ruling R-32) and the eq_min covariate set (ruling R-33).

F2: the adapter's HAC test == statsmodels ``OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L, 'use_correction': True},
use_t=True)`` on the same design; the Andrews (1991) AR(1) bandwidth; R-22 not-testable format; arms; families.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import statsmodels.api as sm

from cdd_oran.xmethod import covariates as C
from cdd_oran.xmethod import score as S
from cdd_oran.xmethod.methods import _classic_synth as SYN
from cdd_oran.xmethod.methods import pcorr_hac as H
from cdd_oran.xmethod.methods._classic_common import (
    columns,
    cond_set,
    pcorr,
    resolve,
    source_column,
)
from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.worlds import REGIMES_OF, generate_dataset

SEED = 3_000_000
CELLS = [(w, r) for w, rs in REGIMES_OF.items() for r in rs]


# ---------------------------------------------------------------------------------------------- eq_min (R-33)
@pytest.mark.parametrize("world,regime", CELLS)
def test_eq_min_is_r3_plus_focal_setpoint(world, regime):
    ds, _ = generate_dataset(world, regime, 200, SEED, lam=1.0)
    for ai, a in enumerate(ds.action_names):
        r3n, r3M, r3m = C.design_covariates(ds, False, False, focal=ai, concurrent="all")
        n, M, m = C.eq_min_covariates(ds, a)
        assert m.all() and r3m.all()
        if ds.designs[ai].kind == "dither":
            j = n.index(f"sp:{a}")
            assert [x for x in n if not x.startswith("sp:")] == list(r3n) and sum(x.startswith("sp:") for x in n) == 1
            assert np.array_equal(M[:, j], ds.designs[ai].fixed_part)
            assert np.array_equal(np.delete(M, j, axis=1), r3M)
        else:                                       # no setpoint (R1 / R3 / R4): eq_min IS the R-3 set
            assert n == r3n and np.array_equal(M, r3M)
        assert C.eq_min_covariates(ds, ai)[0] == n


def test_eq_min_order_and_errors():
    ds, _ = generate_dataset("E2", "R2", 200, SEED)
    n, _, _ = C.eq_min_covariates(ds, "P3")
    assert n[:7] == tuple(f"lag_kpi:{k}" for k in ds.kpi_names) + ("sp:P3",)
    assert n[7:] == tuple(f"concurrent:{a}" for a in ds.action_names if a != "P3")
    with pytest.raises(ValueError):
        C.design_covariates(ds, setpoints="focal")              # needs a focal action
    with pytest.raises(ValueError):
        C.design_covariates(ds, focal=0, setpoints="some")
    # default unchanged: every setpoint
    assert sum(x.startswith("sp:") for x in C.design_covariates(ds, focal=0)[0]) == len(ds.action_names)


def test_eq_min_info_order_rows():
    """Shuffled rows: eq_min columns stay aligned with the Dataset's row order."""
    ds, _ = generate_dataset("E1", "R2", 150, SEED)
    perm = np.random.default_rng(1).permutation(ds.n)
    des = tuple(dataclasses.replace(d, random_part=d.random_part[perm], fixed_part=d.fixed_part[perm])
                for d in ds.designs)
    p = dataclasses.replace(ds, X_action=ds.X_action[perm], X_kpi_lag=ds.X_kpi_lag[perm], Y=ds.Y[perm],
                            designs=des, time_index=ds.time_index[perm])
    n0, M0, _ = C.eq_min_covariates(ds, 2)
    n1, M1, _ = C.eq_min_covariates(p, 2)
    assert n0 == n1 and np.array_equal(M1, M0[perm])


# ---------------------------------------------------------------------------------------------- HAC test (F2)
def _statsmodels(y, x, Z, L):
    X = np.column_stack([H.hac_design(Z), x])
    f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True}, use_t=True)
    return f.params[-1], f.bse[-1], f.pvalues[-1], f.df_resid


@pytest.mark.parametrize("rho", [0.0, 0.6])
@pytest.mark.parametrize("L", [None, 0, 4])
def test_hac_equals_statsmodels_synthetic(rho, L):
    r = np.random.default_rng(5)
    n = 700
    Z = r.normal(size=(n, 4))
    Z = np.column_stack([Z, Z[:, 0] + Z[:, 1], np.full(n, 3.0)])     # a collinear and a constant column
    e = np.zeros(n)
    w = r.normal(size=n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + w[t]
    x = Z[:, 2] + e
    y = 0.3 * Z[:, 0] + 0.1 * x + r.normal(size=n)
    out = H.hac_test(y, x, Z, L)
    assert out["status"] == H.OK and out["k"] == 6                  # 1 + 4 independent Z + x
    g, se, p, df = _statsmodels(y, x, Z, out["maxlags"])
    assert out["g"] == pytest.approx(g, rel=1e-10) and out["se"] == pytest.approx(se, rel=1e-10)
    assert out["p"] == pytest.approx(p, rel=1e-8, abs=1e-14) and out["df"] == df


@pytest.mark.parametrize("world,regime,arm", [("E2", "R2", "native"), ("E2", "R2", "eq_min"), ("E4", "R3", "native"),
                                              ("E5", "R1", "eq_min"), ("E3", "R2", "eq_min")])
def test_adapter_equals_statsmodels_on_world(world, regime, arm):
    ds, _ = generate_dataset(world, regime, 600, SEED, lam=1.0, kappa=0.25)
    res = METHODS["pcorr_hac"]().run(ds, {"arm": arm})
    cols, order = columns(ds), C.info_order(ds)
    by = {(e.source, e.target): e for e in res.edges}
    for s, t in ds.candidates[::3]:
        fam, j = resolve(ds, cols, s)
        x, src = source_column(ds, cols, fam, j)
        cs = cond_set(ds, fam, src, arm)
        L = res.notes["maxlags"][f"{s}->{t}"]
        g, _, p, _ = _statsmodels(cols.Y[order, ds.kpi_names.index(t)], x[order], cs.Z[order], L)
        assert by[(s, t)].p == pytest.approx(p, rel=1e-7, abs=1e-14), (s, t)
        assert by[(s, t)].sign == int(np.sign(g))
        rho = pcorr(x, cols.Y[:, ds.kpi_names.index(t)], cs.Z)
        assert by[(s, t)].sign == int(np.sign(rho))                  # R-4: sign of pcorr given the same Z


def test_andrews_bandwidth():
    v = np.zeros(4000)
    r = np.random.default_rng(0)
    for t in range(1, len(v)):
        v[t] = 0.5 * v[t - 1] + r.normal()
    L, rho, s_t = H.andrews_lags(v)
    a = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    assert rho == pytest.approx(0.5, abs=0.05)
    assert s_t == pytest.approx(1.1447 * (a * 4000) ** (1 / 3)) and L == int(np.floor(s_t))
    assert H.andrews_lags(np.zeros(50))[0] == 0                      # white / degenerate score: no lags
    assert H.andrews_lags(np.arange(30.0))[0] <= 28                  # capped at n - 2


def test_time_order_not_row_order():
    """The HAC lags follow time_index: a row-shuffled copy gives the same p-values."""
    ds, _ = generate_dataset("E2", "R2", 500, SEED, kappa=0.25)
    perm = np.random.default_rng(3).permutation(ds.n)
    des = tuple(dataclasses.replace(d, random_part=d.random_part[perm], fixed_part=d.fixed_part[perm])
                for d in ds.designs)
    p = dataclasses.replace(ds, X_action=ds.X_action[perm], X_kpi_lag=ds.X_kpi_lag[perm], Y=ds.Y[perm],
                            designs=des, time_index=ds.time_index[perm])
    a = METHODS["pcorr_hac"]().run(ds, {"arm": "eq_min"})
    b = METHODS["pcorr_hac"]().run(p, {"arm": "eq_min"})
    assert [e.p for e in a.edges] == pytest.approx([e.p for e in b.edges], rel=1e-9)


# ---------------------------------------------------------------------------------------------- adapter contract
def test_contract_families_and_determinism():
    ds, tr = generate_dataset("E4", "R4", 400, SEED, lam=1.5, kappa=0.25)
    m = METHODS["pcorr_hac"]()
    r = m.run(ds)
    assert r.config["arm"] == "native" and r.notes["arm"] == "native"
    assert [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert r.notes["family"]["P_placebo_conf->K0"] == "diagnostic" and r.notes["family"]["P0->K0"] == "action"
    assert r.notes["family"]["K0->K0"] == "kpi" and "per family" in r.notes["declare_rule"]
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)
    r2 = m.run(ds)
    assert [(e.p, e.sign, e.declared) for e in r.edges] == [(e.p, e.sign, e.declared) for e in r2.edges]
    S.score(r, tr)                                                    # scorer accepts the result
    with pytest.raises(ValueError):
        m.run(ds, {"arm": "eq"})


def test_not_testable_exact_fit_kappa0():
    """Deterministic E1 at kappa 0: Y_k is an exact fit of its parents (in Z + source) -> not testable, R-22."""
    ds, tr = generate_dataset("E1", "R1", 300, SEED, kappa=0.0)
    r = METHODS["pcorr_hac"]().run(ds)
    nt = r.notes["not_testable_edges"]
    assert nt and r.notes["not_testable_reason"].startswith("exact fit") and "not_testable" not in r.notes
    by = {f"{e.source}->{e.target}": e for e in r.edges}
    assert all(np.isnan(by[k].score) and by[k].p is None and not by[k].declared for k in nt)
    assert "P0->K0" in nt
    sc = S.score(r, tr)
    assert sc["n_not_testable_overridden"] == 0 and sc["n_not_testable_true"] >= 1


def test_collinear_source_not_testable():
    data, _ = SYN.make(300, SEED)
    X = data.X_action.copy()
    X[:, 2] = 0.5                                                   # constant action: collinear with the intercept
    r = METHODS["pcorr_hac"]().run(dataclasses.replace(data, X_action=X))
    assert set(r.notes["not_testable_collinear"]) == {f"A2->{k}" for k in data.kpi_names}


def test_planted_edges_found_and_global_null():
    data, truth = SYN.make(3000, SEED + 1, b=1.0)
    r = METHODS["pcorr_hac"]().run(data)
    dec = {(e.source, e.target) for e in r.edges if e.declared}
    assert set(truth.edges) <= dec
    for (s, t), sign in truth.signs.items():
        assert next(e for e in r.edges if (e.source, e.target) == (s, t)).sign == sign
    null, _ = SYN.make(1000, SEED + 2, b=0.0)
    assert not any(e.declared for e in METHODS["pcorr_hac"]().run(null).edges)


def test_eq_min_arm_uses_focal_setpoint():
    ds, _ = generate_dataset("E2", "R2", 400, SEED, kappa=0.25)
    nat = METHODS["pcorr_hac"]().run(ds, {"arm": "native"})
    eqm = METHODS["pcorr_hac"]().run(ds, {"arm": "eq_min"})
    assert eqm.notes["z_size"]["P0->K0"] == nat.notes["z_size"]["P0->K0"] + 1
    assert eqm.notes["z_size"]["K0->K1"] == nat.notes["z_size"]["K0->K1"]          # KPI source: same set
    kk = [i for i, (s, _) in enumerate(ds.candidates) if s in ds.kpi_names]
    assert [nat.edges[i].p for i in kk] == [eqm.edges[i].p for i in kk]
    assert eqm.config["arm"] == "eq_min" and "eq_min" in eqm.notes["label"]


# ---------------------------------------------------------------------------------------------- fixed-b (R-38)
@pytest.mark.parametrize("b", [0.02, 0.1, 0.5, 1.0])
def test_fixedb_reproduces_kv2005_table1(b):
    """F3: at KV (2005) Table I Bartlett critical values cv(b), the simulated fixed-b two-sided p is the nominal one
    (90 / 95 / 97.5 / 99 % -> .20 / .10 / .05 / .02) within Monte Carlo + cubic-fit error (10 % relative)."""
    from cdd_oran.xmethod.methods import _fixedb as F
    for pc, *_ in F.KV_TABLE1_BARTLETT:
        target = 2 * (1 - pc)
        assert F.fixedb_p(F.kv_cv(b, pc), b) == pytest.approx(target, rel=0.10), (b, pc)


def test_fixedb_shape():
    from scipy import stats

    from cdd_oran.xmethod.methods import _fixedb as F
    assert F.fixedb_p(1.96, 0.0) == pytest.approx(2 * stats.norm.sf(1.96))
    ps = [F.fixedb_p(t, 0.1) for t in (0.5, 1.0, 2.0, 3.0, 5.0)]
    assert all(a > c for a, c in zip(ps, ps[1:], strict=False)) and ps[-1] > 0     # decreasing, positive tail
    assert F.fixedb_p(-2.0, 0.1) == F.fixedb_p(2.0, 0.1)
    assert F.fixedb_p(2.0, 0.3) > F.fixedb_p(2.0, 0.1) > F.fixedb_p(2.0, 0.0)          # larger b, heavier tail
    eps = 1e-9                                                                         # continuous at B0
    assert F.fixedb_p(2.0, F.B0 - eps) == pytest.approx(F.fixedb_p(2.0, F.B0 + eps), abs=1e-6)
    assert F.fixedb_p(2.0, 0.1234) == pytest.approx(0.6 * F.fixedb_p(2.0, 0.123) + 0.4 * F.fixedb_p(2.0, 0.124),
                                                    rel=1e-9)
    assert F.fixedb_p(float("inf"), 0.2) == 0.0


def test_adapter_fixed_b_variant():
    from scipy import stats

    from cdd_oran.xmethod.methods import _fixedb as F
    ds, _ = generate_dataset("E2", "R2", 500, SEED, kappa=0.25)
    m = METHODS["pcorr_hac"]()
    rt, rb = m.run(ds, {"arm": "eq_min"}), m.run(ds, {"arm": "eq_min", "inference": "fixed_b"})
    assert rt.config["inference"] == "t" and rb.config["inference"] == "fixed_b" and m.version == "1.1"
    assert "fixed-b" in rb.notes["hac"]["reference"] and rt.notes["t_df_b"] == rb.notes["t_df_b"]
    for et, eb in zip(rt.edges, rb.edges, strict=True):
        if et.p is None:
            assert eb.p is None
            continue
        t, df, b = rt.notes["t_df_b"][f"{et.source}->{et.target}"]
        L = rt.notes["maxlags"][f"{et.source}->{et.target}"]
        assert b == pytest.approx((L + 1) / ds.n) and et.sign == eb.sign
        assert et.p == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-9)
        assert eb.p == pytest.approx(F.fixedb_p(t, b), rel=1e-12)
    with pytest.raises(ValueError):
        m.run(ds, {"inference": "normal"})
