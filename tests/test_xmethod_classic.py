"""xm-classic adapters (pc, notears, shap_dag, two_tower, corr, granger) on synthetic api.Datasets.

Covers the api.Method contract, the CONTRACT sec 5 declaration rules with the orchestrator rulings (BY per family
action | kpi, placebo 2nd-largest tau, pcorr_given_Z sign), and the F2 equivalences (adapter == reference on
identical inputs). Synthetic data only (``_classic_synth``); no study world.
Needs the ``baselines`` dependency group (shap, xgboost): ``uv sync --group baselines``, then
``uv run --group baselines pytest tests/test_xmethod_classic.py``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.xmethod import api
from cdd_oran.xmethod.methods import _classic_synth as SYN
from cdd_oran.xmethod.methods._classic_common import (
    PLACEBO,
    Scored,
    by_declare,
    columns,
    declare,
    placebo_tau,
    source_kind,
)
from cdd_oran.xmethod.methods.classic import METHODS

FAST = {"two_tower": {"epochs": 30}}


def _run(name, data, **cfg):
    return METHODS[name]().run(data, dict(FAST.get(name, {}), **cfg))


@pytest.fixture(scope="module")
def syn():
    return SYN.make(800, 3_000_000, b=1.0)


@pytest.mark.parametrize("name", list(METHODS))
def test_result_contract(name, syn):
    data, _ = syn
    r = _run(name, data)
    assert isinstance(r, api.Result) and r.method == name and r.version
    assert [(e.source, e.target) for e in r.edges] == list(data.candidates)
    assert all(e.sign in (-1, 0, 1) for e in r.edges)
    assert all(isinstance(e.declared, bool) for e in r.edges)
    assert set(r.notes["family"].values()) <= {"action", "kpi"}
    assert r.notes["family"][f"{PLACEBO}->Y0"] == "action" and r.notes["family"]["Y0->Y0"] == "kpi"
    assert r.cpu_s >= 0 and r.notes["threads"] == 1
    if METHODS[name].uses_p:
        assert all(e.p is None or 0.0 <= e.p <= 1.0 for e in r.edges)
    else:                                       # score-only and untuned: nothing declared
        assert not any(e.declared for e in r.edges)


@pytest.mark.parametrize("name", ["pc", "notears", "corr", "granger"])
def test_planted_strong_edges_rank_top(name):
    data, truth = SYN.make(3000, 3_000_001, b=1.0)
    r = _run(name, data)
    sc = {(e.source, e.target): e.score for e in r.edges}
    for (s, t), sign in truth.signs.items():
        nulls = [sc[(a, t)] for a in data.action_names if (a, t) not in truth.edges and np.isfinite(sc[(a, t)])]
        assert sc[(s, t)] > max(nulls), (s, t)
        e = next(e for e in r.edges if (e.source, e.target) == (s, t))
        assert e.sign == sign, (name, s, t)


def test_placebo_tau_rule():
    """R-29 conformal tau through ClassicBase.tune: the ceil((M+1)(1-.05))-th smallest DEV placebo score."""
    import math
    m = METHODS["notears"]()
    dev = [SYN.make(500, 3_000_000 + i, b=1.0)[0] for i in range(10)]
    cfg = m.tune(dev)
    pl = sorted(e.score for d in dev for e in m.run(d, {}).edges if e.source == PLACEBO)
    k = math.ceil((len(pl) + 1) * 0.95 - 1e-9)
    assert len(pl) == 30 and k == 30 and cfg["tau"] == pytest.approx(pl[min(k, len(pl)) - 1])
    assert cfg["tau_rule"] == "placebo_conformal_0.05" and cfg["n_placebo_scores"] == 30
    n_pl = sum(e.declared for d in dev for e in m.run(d, cfg).edges if e.source == PLACEBO)
    assert n_pl == 0                                                      # M = 30 < 39: tau = max
    assert placebo_tau([])["tau"] == float("inf")


def test_tune_respects_arm():
    """audit-classic2 A: tune(config={'arm': ...}) tunes that arm (pc eq and native taus differ); the default is the
    method's first arm; a native-only method gives the same tau whichever arm is requested."""
    dev = [SYN.make(300, 3_000_020 + i, b=1.0, regime="R2")[0] for i in range(3)]
    pc = METHODS["pc"]()
    eq, nat, dflt = pc.tune(dev, config={"arm": "eq"}), pc.tune(dev, config={"arm": "native"}), pc.tune(dev)
    assert (eq["arm"], nat["arm"], dflt["arm"]) == ("eq", "native", "eq")
    assert dflt["tau"] == eq["tau"] and eq["tau"] != nat["tau"]
    assert nat["tau"] == placebo_tau([pc.run(d, {"arm": "native"}) for d in dev])["tau"]
    nt = METHODS["notears"]()
    assert nt.tune(dev, config={"arm": "eq"})["tau"] == nt.tune(dev, config={"arm": "native"})["tau"]


def test_tune_refuses_truth():
    with pytest.raises(ValueError):
        METHODS["corr"]().tune([], truth_free=False)


def test_by_is_per_family():
    s = [Scored("A0", "Y0", "action", 5.0, 1, 1e-4), Scored("A1", "Y0", "action", 0.1, 1, 0.8),
         Scored("Y0", "Y0", "kpi", 3.0, 1, 0.03), Scored("Y1", "Y0", "kpi", 0.1, 1, 0.9)]
    dec, notes = declare(s, {"q": 0.05}, uses_p=True)
    # kpi family alone: m = 2, c_m = 1.5 -> threshold for rank 1 = .05 / 3 = .0167 < .03 -> not declared
    assert dec == [True, False, False, False]
    pooled = by_declare([1e-4, 0.8, 0.03, 0.9])
    assert list(pooled) == [True, False, False, False]
    s[2] = dataclasses.replace(s[2], p=0.01)
    assert declare(s, {"q": 0.05}, uses_p=True)[0][2]
    assert "per family" in notes["declare_rule"]


def test_source_name_resolution(syn):
    data, _ = syn
    assert source_kind(data, "A1") == ("action", 1)
    for nm in ("Y2", "Y2_lag", "Y2[t]", "lag_Y2"):
        assert source_kind(data, nm) == ("kpi", 2)
    with pytest.raises(KeyError):
        source_kind(data, "nope")


@pytest.mark.parametrize("name", ["pc", "notears", "corr", "granger"])
def test_world_without_lags(name):
    data, _ = SYN.make(400, 3_000_002)
    data = dataclasses.replace(data, X_kpi_lag=np.full_like(data.X_kpi_lag, np.nan))
    r = _run(name, data)
    kpi = [e for e in r.edges if r.notes["family"][f"{e.source}->{e.target}"] == "kpi"]
    assert kpi and all(np.isnan(e.score) and not e.declared for e in kpi)
    assert any(np.isfinite(e.score) for e in r.edges)


@pytest.mark.parametrize("name", ["pc", "notears"])
def test_context_column_runs(name):
    data, _ = SYN.make(400, 3_000_003)
    ctx = np.random.default_rng(0).normal(size=(data.n, 2))
    r = _run(name, dataclasses.replace(data, context=ctx))
    assert len(r.edges) == len(data.candidates)


def test_pcorr_sign_rule(syn):
    data, _ = syn
    for name in ("pc", "shap_dag", "two_tower"):
        r = _run(name, data)
        assert r.notes["sign_rule"] == "pcorr_given_Z"
        sg = {(e.source, e.target): e.sign for e in r.edges}
        assert sg[("A0", "Y0")] == 1 and sg[("A1", "Y1")] == -1
    from cdd_oran.xmethod.methods._classic_common import pcorr
    rng = np.random.default_rng(1)
    z = rng.normal(size=5000)
    x = z + rng.normal(size=5000)
    y = 2 * z - 0.5 * x + rng.normal(size=5000)          # marginal corr > 0, partial corr given z < 0
    assert np.corrcoef(x, y)[0, 1] > 0 and pcorr(x, y, z[:, None]) < 0


# ------------------------------------------------------------------------------------------------ F2 equivalences
def test_f2_notears_bk_equals_vendored_without_knowledge():
    from cdd_oran.xmethod.methods._vendor.notears.linear import notears_linear
    from cdd_oran.xmethod.methods.notears import notears_linear_bk
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 5))
    X[:, 2] += 1.5 * X[:, 0]
    X[:, 4] -= X[:, 2]
    W1 = notears_linear(X, lambda1=0.1, loss_type="l2")
    W2 = notears_linear_bk(X, lambda1=0.1, loss_type="l2")
    assert np.array_equal(W1, W2)
    fb = np.zeros((5, 5), dtype=bool)
    fb[0, 2] = True
    W3 = notears_linear_bk(X, lambda1=0.1, loss_type="l2", w_threshold=0.0, forbid=fb)
    assert W3[0, 2] == 0.0


def test_f2_pc_equals_causallearn_pc():
    """The adapter's PC pipeline == causal-learn ``pc()`` with the same background knowledge."""
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from cdd_oran.xmethod.methods.pc import run_pc
    data, _ = SYN.make(1500, 3_000_004, b=1.0)
    cols = columns(data)
    D = np.column_stack([cols.actions, cols.lags, cols.Y])
    names = [f"A:{a}" for a in data.action_names] + [f"L:{k}" for k in cols.lag_names] + [f"Y:{k}" for k in
                                                                                        data.kpi_names]
    kinds = [nm[0] for nm in names]
    tiers = [1 if k == "Y" else 0 for k in kinds]
    forbidden = {(i, j) for i, ki in enumerate(kinds) for j, kj in enumerate(kinds) if i != j and
                 ((kj == "A" and ki in "LYA") or (ki == "A" and kj in "LC"))}
    cg1, rec = run_pc(D, names, tiers, forbidden, 0.05, "fisherz", True, 0, 2)
    nodes = CausalGraph(D.shape[1], names).G.nodes
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], t)
    for i, j in forbidden:
        bk.add_forbidden_by_node(nodes[i], nodes[j])
    cg2 = pc(D, 0.05, "fisherz", stable=True, uc_rule=0, uc_priority=2, background_knowledge=bk,
             show_progress=False, node_names=names)
    assert np.array_equal(cg1.G.graph, cg2.G.graph)
    assert rec.n_tests > 0 and rec.fallbacks == 0


@pytest.mark.parametrize("arm", ["eq", "native"])
def test_f2_pc_native_matches_score_at_alpha(syn, arm):
    data, _ = syn
    r = _run("pc", data, arm=arm)
    nat = set(r.notes["native_declared"])
    for e in r.edges:
        if np.isfinite(e.score):
            assert (f"{e.source}->{e.target}" in nat) == (e.score >= -np.log10(0.05) - 1e-12), (e.source, e.target)


def test_f2_granger_equals_statsmodels():
    from statsmodels.tsa.stattools import grangercausalitytests
    burn, n = 50, 600
    data, _ = SYN.make(n, 3_000_005, b=1.0, burn=burn)
    r = _run("granger", data, arm="native")
    assert r.notes["label"].startswith("pairwise Granger")
    p = {(e.source, e.target): e.p for e in r.edges}
    # rows are consecutive: K series = [X_kpi_lag[0], Y[0], Y[1], ...]
    K = np.vstack([data.X_kpi_lag[:1], data.Y])
    for src, tgt in (("Y1", "Y2"), ("Y0", "Y1"), ("Y2", "Y0")):
        s, t = data.kpi_names.index(src), data.kpi_names.index(tgt)
        res = grangercausalitytests(np.column_stack([K[:, t], K[:, s]]), maxlag=[1])
        assert p[(src, tgt)] == pytest.approx(res[1][0]["ssr_ftest"][1], rel=1e-8)
    # action source: same restricted / full OLS F-test via statsmodels OLS
    import statsmodels.api as sm
    y, own, x = data.Y[:, 0], data.X_kpi_lag[:, 0], data.X_action[:, 0]
    full = sm.OLS(y, sm.add_constant(np.column_stack([own, x]))).fit()
    assert p[("A0", "Y0")] == pytest.approx(float(full.f_test("x2 = 0").pvalue), rel=1e-8)


def test_f2_corr_equals_numpy(syn):
    data, _ = syn
    r = _run("corr", data)
    e = next(e for e in r.edges if (e.source, e.target) == ("A1", "Y1"))
    assert e.score == pytest.approx(abs(np.corrcoef(data.X_action[:, 1], data.Y[:, 1])[0, 1]))


def test_f2_shap_dag_hgbdt_option_wraps_script(syn):
    """Sensitivity option "hgbdt" == the earlier E2 port's function at n <= 10 000 (where sklearn's 'auto' early
    stopping is off anyway)."""
    from scripts.e2_baseline_shap_dag import fit_shap_importances
    data, _ = SYN.make(300, 3_000_006, b=1.0)
    r = _run("shap_dag", data, seed=0, regressor="hgbdt")
    imp, _ = fit_shap_importances(data.X_action, data.Y[:, 1], seed=0)
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([sc[(a, "Y1")] for a in data.action_names])
    assert np.allclose(got, imp / np.std(data.Y[:, 1]))
    assert all(np.isnan(sc[(k, "Y1")]) for k in data.kpi_names)          # RCPs only (paper)
    assert r.notes["regressor"] == "hgbdt"


def test_f2_shap_dag_default_is_xgboost_package_defaults():
    """Default regressor = XGBRegressor at the package defaults (paper's choice, R-13), seeded, no early stopping."""
    import shap
    import xgboost
    data, _ = SYN.make(300, 3_000_006, b=1.0)
    r = _run("shap_dag", data, seed=0)
    assert r.notes["regressor"] == "xgboost" and r.notes["packages"]["xgboost"] == xgboost.__version__
    model = xgboost.XGBRegressor(random_state=0, n_jobs=1).fit(data.X_action, data.Y[:, 1])
    assert model.get_params()["early_stopping_rounds"] is None
    imp = np.abs(shap.TreeExplainer(model).shap_values(data.X_action)).mean(0)
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([sc[(a, "Y1")] for a in data.action_names])
    assert np.allclose(got, imp / np.std(data.Y[:, 1]), rtol=1e-6)
    assert max(sc[(a, "Y1")] for a in data.action_names) == sc[("A1", "Y1")]   # planted A1 -> Y1 ranks top


def test_shap_dag_no_early_stopping_at_any_n():
    """Neither regressor early-stops (sklearn's HistGBR default 'auto' would switch it on at n > 10 000)."""
    from cdd_oran.xmethod.methods.shap_dag import make_regressor
    assert make_regressor("hgbdt", 0).get_params()["early_stopping"] is False
    assert make_regressor("xgboost", 0).get_params()["early_stopping_rounds"] is None
    with pytest.raises(ValueError):
        make_regressor("rf", 0)


def test_f2_two_tower_wraps_script():
    """Raw gate == the script's S; score = row share S[k, p] / sum_p S[k, p] (F7 fix); native = relative rule on S."""
    import torch

    import scripts.e2_baseline_gnn as G
    from cdd_oran.xmethod.methods.two_tower import LABEL, _sized
    data, _ = SYN.make(300, 3_000_007, b=1.0)
    r = _run("two_tower", data, seed=0, epochs=20)
    torch.set_num_threads(1)
    with _sized(G, num_params=4, num_kpis=3, d=16, r=8, hidden=32):
        S, _ = G.fit_two_tower(data.X_action, data.Y, epochs=20, lr=0.01, l1=1e-3, seed=0)
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([[sc[(a, k)] for a in data.action_names] for k in data.kpi_names])
    gate = np.array([[r.notes["gate"][k][a] for a in data.action_names] for k in data.kpi_names])
    assert np.allclose(gate, S, atol=1e-6)
    assert np.allclose(got, S / S.sum(1, keepdims=True), atol=1e-6)
    assert np.allclose(got.sum(1), 1.0)
    nat = {f"{a}->{k}" for i, k in enumerate(data.kpi_names) for j, a in enumerate(data.action_names)
           if S[i, j] >= 0.10 * S[i].max()}
    assert set(r.notes["native_declared"]) == nat
    assert r.notes["label"] == LABEL and "not the published" in LABEL


def test_pc_notes_fallback_rate_and_granger_label(syn):
    data, _ = syn
    r = _run("pc", data)
    assert r.notes["fisherz_pinv_fallback_rate"] == pytest.approx(r.notes["fisherz_pinv_fallbacks"]
                                                                 / r.notes["n_ci_tests"])
    assert "pMax" in r.notes["score"]
    assert _run("granger", data, arm="native").notes["label"].startswith("pairwise Granger")
    assert _run("granger", data).notes["label"].startswith("conditional Granger")


# ------------------------------------------------------------------------------------------------ arms (R-17, R-18, R-25)
def test_cond_set_from_shared_helper():
    """R-25: the whole conditioning set comes from design_covariates; eq = focal set (concurrent designed), native =
    R-3 set (concurrent all); a lagged-KPI source drops its own lag and conditions on every action."""
    from cdd_oran.xmethod.covariates import design_covariates
    from cdd_oran.xmethod.methods._classic_common import cond_set
    data, _ = SYN.make(300, 3_000_009, regime="R2")
    data = dataclasses.replace(data, context=np.random.default_rng(0).normal(size=(data.n, 1)))
    eq = cond_set(data, "action", 0, "eq")
    names, M, mask = design_covariates(data, focal=0)
    assert eq.names == names and np.array_equal(eq.Z, M) and np.array_equal(eq.mask, mask)
    assert int(eq.mask.sum()) == data.n - 2 and not eq.mask[:2].any()
    assert {f"sp:{a}" for a in data.action_names} <= set(eq.names)            # setpoints incl. focal + placebo
    assert {f"{a}@t-{k}" for a in data.action_names for k in (1, 2)} <= set(eq.names)
    assert "concurrent:A0" not in eq.names and "concurrent:A1" in eq.names and "ctx:c0" in eq.names
    nat = cond_set(data, "action", 0, "native")
    assert nat.names == design_covariates(data, False, False, focal=0, concurrent="all")[0] and nat.mask.all()
    assert not any(nm.startswith(("sp:", "A1@")) for nm in nat.names)
    kp = cond_set(data, "kpi", 1, "eq")
    assert "lag_kpi:Y1" not in kp.names and "lag_kpi:Y0" in kp.names
    assert {f"concurrent:{a}" for a in data.action_names} <= set(kp.names)
    r1 = cond_set(SYN.make(300, 3_000_009, regime="R1")[0], "action", 0, "eq")
    assert not any(nm.startswith("sp:") for nm in r1.names)                    # R1: no setpoints


def test_f2_granger_eq_equals_statsmodels_ols():
    """Equal-information arm == statsmodels OLS F-test of the source given intercept + its helper set (R-25)."""
    import statsmodels.api as sm

    from cdd_oran.xmethod.methods._classic_common import cond_set
    data, _ = SYN.make(500, 3_000_010, b=1.0, regime="R2")
    r = _run("granger", data)
    assert r.notes["arm"] == "eq" and r.notes["n_rows_used"] == data.n - 2
    p = {(e.source, e.target): e.p for e in r.edges}
    for fam, src, name, tgt, x in (("action", 0, "A0", "Y0", data.X_action[:, 0]),
                                   ("action", 2, "A2", "Y1", data.X_action[:, 2]),
                                   ("kpi", 1, "Y1", "Y2", data.X_kpi_lag[:, 1])):
        cs = cond_set(data, fam, src, "eq")
        m = cs.mask
        X = sm.add_constant(np.column_stack([cs.Z[m], x[m]]))
        fit = sm.OLS(data.Y[m, data.kpi_names.index(tgt)], X).fit()
        R = np.zeros((1, X.shape[1]))
        R[0, -1] = 1.0
        assert p[(name, tgt)] == pytest.approx(float(fit.f_test(R).pvalue), rel=1e-6), (name, tgt)


def test_pc_eq_arm_exogenous_design_nodes():
    data, _ = SYN.make(800, 3_000_011, b=1.0, regime="R2")
    r = _run("pc", data)
    assert r.notes["arm"] == "eq" and r.notes["n_rows_used"] == data.n - 2
    assert "sp:A0" in r.notes["eq_covariates"] and "A0@t-1" in r.notes["eq_covariates"]
    assert r.notes["n_vars"] == (len(data.action_names) + 2 * len(data.kpi_names) + len(r.notes["eq_covariates"])
                                 - len(r.notes["eq_dropped"]) - len(r.notes["dropped_constant"]))
    assert [(e.source, e.target) for e in r.edges] == list(data.candidates)   # design nodes are never candidates
    rn = _run("pc", data, arm="native")
    assert rn.notes["arm"] == "native" and "eq_covariates" not in rn.notes
    assert rn.notes["n_vars"] == len(data.action_names) + 2 * len(data.kpi_names)


@pytest.mark.parametrize("name", ["corr", "notears", "shap_dag", "two_tower"])
def test_native_only_methods_note_no_interface(name, syn):
    data, _ = syn
    r = _run(name, data, arm="eq")
    assert r.notes["arm"] == "native" and r.notes["arm_note"] == "no conditioning interface"
    assert r.notes["arm_requested"] == "eq"
    with pytest.raises(ValueError):
        _run(name, data, arm="both")


def test_meta_families_and_diagnostic_by_family():
    data, _ = SYN.make(600, 3_000_008, b=1.0)
    cands = data.candidates
    meta = {"primary_candidates": tuple(c for c in cands if c[0] in data.action_names and c[0] != "A2"),
            "secondary_candidates": tuple(c for c in cands if c[0] in data.kpi_names),
            "diagnostic_candidates": tuple(c for c in cands if c[0] == "A2")}
    r = _run("corr", dataclasses.replace(data, meta=meta))
    fam = r.notes["family"]
    assert fam["A2->Y0"] == "diagnostic" and fam["A0->Y0"] == "action" and fam["Y0->Y0"] == "kpi"
    sc = [Scored(e.source, e.target, fam[f"{e.source}->{e.target}"], e.score, e.sign, e.p) for e in r.edges]
    for f in ("action", "kpi", "diagnostic"):
        idx = [i for i, x in enumerate(sc) if x.family == f]
        assert [r.edges[i].declared for i in idx] == list(by_declare([sc[i].p for i in idx]))


@pytest.mark.parametrize("name", ["granger", "pc"])
def test_exact_fit_candidates_are_not_testable(name):
    """Deterministic target (no noise), eq arm: a null source cannot add to an exact fit -> NOT TESTABLE (score NaN,
    p None, never declared, counted in notes; orchestrator Q-F1), not p = 1; a true source stays testable."""
    data, _ = SYN.make(300, 3_000_012, b=1.0, noise=0.0, regime="R2")
    r = _run(name, data)
    by = {(e.source, e.target): e for e in r.edges}
    for k in data.kpi_names:
        e = by[(PLACEBO, k)]
        assert np.isnan(e.score) and e.p is None and not e.declared
    assert r.notes["not_testable_reason"] == "exact fit: deterministic world under Z" and "not_testable" not in r.notes
    assert r.notes["n_not_testable"] == len(r.notes["not_testable_edges"]) > 0
    assert f"{PLACEBO}->Y0" in r.notes["not_testable_edges"] and "A0->Y0" not in r.notes["not_testable_edges"]
    assert np.isfinite(by[("A0", "Y0")].score)
    if name == "granger":
        assert by[("A0", "Y0")].p < 1e-10
    clean = _run(name, SYN.make(300, 3_000_012, b=1.0, regime="R2")[0])
    assert "not_testable_edges" not in clean.notes and "not_testable_reason" not in clean.notes


@pytest.mark.parametrize("name", ["granger", "pc"])
def test_score_accepts_not_testable_notes(name):
    """fix-classic2 Q1: the harness score() reads the adapters' not-testable listing (edges in not_testable_edges,
    reason in not_testable_reason) instead of raising on a reason string; listed edges count as not declared."""
    from cdd_oran.xmethod.score import score
    data, truth = SYN.make(300, 3_000_012, b=1.0, noise=0.0, regime="R2")
    r = _run(name, data)
    s = score(r, truth, candidates=data.candidates, kpi_sources=data.kpi_names)
    listed = sorted(r.notes["not_testable_edges"])
    assert listed and s["not_testable_edges"] == listed and s["n_not_testable_overridden"] == 0
    assert s["placebo_not_testable"] == len(data.kpi_names)
    assert not set(listed) & set(s["declared_edges"])


@pytest.mark.parametrize("arm", ["eq", "native"])
def test_granger_collinear_source_is_not_testable(arm):
    """audit-classic2 B: a source collinear with the restricted model ("degenerate", e.g. a lag that copies another
    lag) is listed as not testable ('source collinear with Z'), not left as an unlisted NaN."""
    data, _ = SYN.make(300, 3_000_030, b=1.0, regime="R2")
    lag = data.X_kpi_lag.copy()
    lag[:, 2] = lag[:, 1]
    r = _run("granger", dataclasses.replace(data, X_kpi_lag=lag), arm=arm)
    nan = sorted(f"{e.source}->{e.target}" for e in r.edges if np.isnan(e.score))
    assert nan and sorted(r.notes["not_testable_edges"]) == nan == sorted(r.notes["not_testable_collinear"])
    assert r.notes["not_testable_reason"] == "source collinear with Z" and r.notes["n_not_testable"] == len(nan)
    assert all(e.p is None and not e.declared for e in r.edges if np.isnan(e.score))


# ---------------------------------------------------------------------------------------------- R-37
def _conf_scrambled(ds):
    """Same dataset with the diagnostic P_placebo_conf column (and its logged table) replaced by unrelated values."""
    j = ds.action_names.index("P_placebo_conf")
    X = ds.X_action.copy()
    X[:, j] = np.random.default_rng(11).permutation(X[:, j])[::-1]
    des = list(ds.designs)
    if des[j].propensity is not None:
        des[j] = dataclasses.replace(des[j], propensity=des[j].propensity[::-1].copy())
    return dataclasses.replace(ds, X_action=X, designs=tuple(des))


@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_cond_set_never_holds_placebo_conf(regime):
    """R-37: no conditioning set of another source carries P_placebo_conf; its own set is the full helper set."""
    from cdd_oran.xmethod.covariates import design_covariates
    from cdd_oran.xmethod.methods._classic_common import cond_set, is_diagnostic_column
    from cdd_oran.xmethod.worlds import generate_dataset
    ds, _ = generate_dataset("E4", regime, 300, 3_000_000, lam=1.5, kappa=0.25)
    conf = ds.action_names.index("P_placebo_conf")
    for arm in ("eq", "native", "eq_min"):
        for ai in range(len(ds.action_names)):
            names = cond_set(ds, "action", ai, arm).names
            if ai == conf:
                assert "concurrent:P0" in names or arm == "eq"          # eq: R4 P0 is not designed
                ref = (design_covariates(ds, focal=ai) if arm == "eq" else
                       design_covariates(ds, False, False, focal=ai, concurrent="all"))[0]
                assert names == ref
            else:
                assert not any(is_diagnostic_column(nm) or "P_placebo_conf" in nm for nm in names), (arm, ai)
        assert not any("P_placebo_conf" in nm for nm in cond_set(ds, "kpi", 0, arm).names)
    assert is_diagnostic_column("P_placebo_conf@t-2:missing") and not is_diagnostic_column("concurrent:P_placebo")


@pytest.mark.parametrize("name,arm", [("pc", "eq"), ("pc", "native"), ("granger", "eq"), ("pcorr_hac", "native"),
                                      ("pcorr_hac", "eq_min"), ("notears", "native"), ("shap_dag", "native"),
                                      ("two_tower", "native")])
@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_placebo_conf_does_not_change_primary(name, arm, regime):
    """R-37: scrambling P_placebo_conf leaves every non-diagnostic candidate unchanged (score, p, sign, declared);
    its own candidates are still scored (pc: from the second, full-node graph)."""
    from cdd_oran.xmethod.worlds import generate_dataset
    ds, _ = generate_dataset("E4", regime, 400, 3_000_001, lam=1.5, kappa=0.25)
    a, b = _run(name, ds, arm=arm), _run(name, _conf_scrambled(ds), arm=arm)

    def rows(r):
        return repr([(e.source, e.target, e.score, e.p, e.sign, e.declared) for e in r.edges
                     if e.source != "P_placebo_conf"])                        # repr: NaN == NaN
    assert rows(a) == rows(b)
    diag = [e for e in a.edges if e.source == "P_placebo_conf"]
    assert diag and all(np.isfinite(e.score) for e in diag)
    if name == "pc":
        assert a.notes["diagnostic_graph"]["n_vars"] > a.notes["n_vars"]
        assert not any("P_placebo_conf" in x for x in a.notes.get("eq_covariates", []))
    if name in ("notears", "shap_dag", "two_tower"):                     # R-38: R-37 extended to these three
        assert "R-37" in a.notes["diagnostic_fit"]["rule"]
    if name == "two_tower":
        assert all("P_placebo_conf" not in g for g in a.notes["gate"].values())
