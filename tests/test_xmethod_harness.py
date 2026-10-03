"""Tests of the cross-method study harness: worlds / regimes, truth, scorer, runner (cdd_oran/xmethod)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows
from cdd_oran.envs.v2.e2 import E2V2Env
from cdd_oran.xmethod import api, runner
from cdd_oran.xmethod.score import apply_threshold, placebo_tau, score
from cdd_oran.xmethod.worlds import REGIMES_OF, dataset_hash, e4_logged, generate_dataset, truth_for
from cdd_oran.xmethod.worlds import generate as gen

SEED = 3_000_000
CELLS = [(w, r) for w, rs in REGIMES_OF.items() for r in rs]


def _slope(y, *xs):
    X = np.column_stack([np.ones(len(y)), *xs])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


# ------------------------------------------------------------------------------------------------ worlds
@pytest.mark.parametrize("world,regime", CELLS)
def test_regeneration_is_byte_identical_and_seed_dependent(world, regime):
    a, _ = generate_dataset(world, regime, 300, SEED, lam=1.5)
    b, _ = generate_dataset(world, regime, 300, SEED, lam=1.5)
    c, _ = generate_dataset(world, regime, 300, SEED + 1, lam=1.5)
    assert dataset_hash(a) == dataset_hash(b)
    assert a.X_action.tobytes() == b.X_action.tobytes() and a.Y.tobytes() == b.Y.tobytes()
    assert dataset_hash(a) != dataset_hash(c)


@pytest.mark.parametrize("world,regime", CELLS)
def test_shapes_candidates_truth(world, regime):
    ds, tr = generate_dataset(world, regime, 200, SEED)
    p, k = ds.X_action.shape[1], ds.Y.shape[1]
    conf = regime in ("R3", "R4")
    tail = (gen.PLACEBO, gen.PLACEBO_CONF) if conf else (gen.PLACEBO,)
    assert ds.action_names[-len(tail):] == tail and len(ds.action_names) == p and len(ds.kpi_names) == k
    assert ds.X_kpi_lag.shape == ds.Y.shape == (200, k) and len(ds.designs) == p
    assert np.array_equal(ds.Y[:-1], ds.X_kpi_lag[1:])                    # row alignment
    assert np.all(np.diff(ds.time_index) == 1)
    cands = set(ds.candidates)
    assert len(cands) == len(ds.candidates) == p * k + k * k
    assert tr.edges | tr.null_edges == cands and not (tr.edges & tr.null_edges)
    assert all((nm, t) in tr.null_edges for t in ds.kpi_names for nm in tail)
    prim, sec, diag = (ds.meta[f"{x}_candidates"] for x in ("primary", "secondary", "diagnostic"))
    assert set(prim) | set(sec) | set(diag) == cands and len(prim) == (p - len(tail) + 1) * k
    assert all(s in ds.kpi_names for s, _ in sec) and all(s == gen.PLACEBO_CONF for s, _ in diag)
    assert len(diag) == (k if conf else 0) and ds.meta["placebo_conf"] == (gen.PLACEBO_CONF if conf else None)
    assert (gen.PLACEBO, ds.kpi_names[0]) in prim
    assert all(np.isfinite(a).all() for a in (ds.X_action, ds.X_kpi_lag, ds.Y))
    assert (ds.context is not None) == (regime == "R3")
    kind = {"R1": "iid", "R2": "dither", "R3": "logged", "R4": "none"}[regime]
    if conf:                                       # ruling R-10: tuning placebo is i.i.d. from P0's marginal
        assert ds.designs[0].kind == ds.designs[2].kind == kind and ds.designs[1].kind == "iid"
    else:
        assert all(d.kind == kind for d in ds.designs)


def _env(world):
    cls = gen._WORLDS[world].env_cls
    return cls(env_seed=0) if world != "E4" else cls(env_seed=0, mode="do")


@pytest.mark.parametrize("world", list(REGIMES_OF))
def test_truth_matches_scm_by_intervention(world):
    """Every non-placebo candidate: replacing the source by another row's value changes the target somewhere
    iff the candidate is a true edge (exact zero effect for nulls); signed edges move only in the truth sign."""
    ds, tr = generate_dataset(world, "R1", 3000, SEED)
    env = _env(world)
    pr = env.num_params
    z = np.random.default_rng(1).standard_normal(len(ds.Y))
    perm = np.random.default_rng(2).permutation(len(ds.Y))

    def f(a, kl, zi):
        if world in ("E4", "E5"):
            env.prev_Z = zi
        return env._update_kpis(a, kl)

    for src, tgt in ds.candidates:
        if src == gen.PLACEBO:
            continue
        j = int(tgt[1:])
        diffs = []
        for i in range(len(ds.Y)):
            a, kl = ds.X_action[i, :pr].copy(), ds.X_kpi_lag[i].copy()
            a2, kl2 = a.copy(), kl.copy()
            if src.startswith("P"):
                a2[int(src[1:])] = ds.X_action[perm[i], int(src[1:])]
                dsrc = a2[int(src[1:])] - a[int(src[1:])]
            else:
                kl2[int(src[1:])] = ds.X_kpi_lag[perm[i], int(src[1:])]
                dsrc = kl2[int(src[1:])] - kl[int(src[1:])]
            d = f(a2, kl2, z[i])[j] - f(a, kl, z[i])[j]
            diffs.append(np.sign(d) * np.sign(dsrc))
        diffs = np.array(diffs)
        moved = np.any(diffs != 0)
        assert moved == ((src, tgt) in tr.edges), (world, src, tgt)
        if (src, tgt) in tr.signs:
            assert set(np.unique(diffs[diffs != 0])) == {tr.signs[(src, tgt)]}, (world, src, tgt)


def test_e2_r1_roller_reproduces_e2slice_rows():
    """The generic roller is the e2slice row logic: same env, same draws, two primes -> identical rows."""
    cfg = E2DatasetConfig(n_rows_per_seed=300, seed=7)
    ref = generate_rows(cfg)
    env = E2V2Env(env_seed=7, episode=7)
    env.reset(episode=7)
    rng = np.random.default_rng(np.random.SeedSequence(entropy=0, spawn_key=(7,)))

    def inject(_i):
        for i, (lo, hi) in enumerate(E2V2Env.id_ranges):
            env.apply_action(i, float(rng.uniform(lo, hi)))

    xa, xk, y, _, _ = gen._roll(env, inject, 300, warmup=2)
    assert np.array_equal(xa, ref.x_params) and np.array_equal(xk, ref.x_kpis) and np.array_equal(y, ref.y_kpis)


# ------------------------------------------------------------------------------------------------ E4
@pytest.mark.parametrize("lam", gen.E4_LAMBDAS)
def test_e4_r1_sign_correct_at_every_lambda_and_lambda_inert(lam):
    ds, _ = generate_dataset("E4", "R1", 4000, SEED, lam=lam)
    b = _slope(ds.Y[:, 0], ds.X_action[:, 0])
    assert -1.3 < b < -0.7                                              # alpha = -1, randomised A
    base, _ = generate_dataset("E4", "R1", 4000, SEED, lam=1.0)
    assert np.array_equal(ds.X_action, base.X_action) and np.array_equal(ds.Y, base.Y)


def test_e4_r4_and_r3_marginal_sign_wrong_at_lambda_1_5():
    r4, _ = generate_dataset("E4", "R4", 4000, SEED, lam=1.5)
    assert _slope(r4.Y[:, 0], r4.X_action[:, 0]) > 0.5                  # sign reversal (frozen contract)
    r3, _ = generate_dataset("E4", "R3", 4000, SEED, lam=1.5)
    assert _slope(r3.Y[:, 0], r3.X_action[:, 0]) > 0.5                  # marginal: wrong sign
    b_adj = _slope(r3.Y[:, 0], r3.X_action[:, 0], r3.context[:, 0])     # adjust for observed Z: identified
    assert abs(b_adj + 1.0) < 1e-8                                      # K = -A + 2.5 Z exactly (noiseless)


@pytest.mark.parametrize("lam", gen.E4_LAMBDAS)
def test_e4_r3_design_is_redrawable_and_matches_frozen_behaviour(lam):
    r3, _ = generate_dataset("E4", "R3", 2000, SEED, lam=lam)
    r4, _ = generate_dataset("E4", "R4", 2000, SEED, lam=lam)
    # R3 action = grid rounding of the frozen R4 behaviour action (same Z / eta tape), outcome mechanism frozen
    assert np.array_equal(r3.X_action[:, 0], e4_logged.GRID[e4_logged.grid_index(r4.X_action[:, 0])])
    assert r3.action_names == ("P0", gen.PLACEBO, gen.PLACEBO_CONF)
    for j in (0, 2):                                                    # the Z-policy columns
        name, d = r3.action_names[j], r3.designs[j]
        tab = d.propensity
        assert tab.shape == (2000, 101) and np.allclose(tab.sum(axis=1), 1.0) and (tab > 0).all()
        assert d.dist["name"] == "categorical_rows" and d.dist["values"] == e4_logged.GRID.tolist()
        k = e4_logged.grid_index(r3.X_action[:, j])
        assert np.allclose(tab[np.arange(2000), k], r3.meta["realised_propensity"][name])
        assert np.allclose(r3.meta["realised_propensity"][name],
                           e4_logged.policy_prob_of(k, r3.context[:, 0], lam))
    # calibration: mean realised grid index vs the table's expectation
    exp_k = (r3.designs[2].propensity * np.arange(101)).sum(axis=1)
    k_p = e4_logged.grid_index(r3.X_action[:, 2])
    assert abs((k_p - exp_k).mean()) < 4 * (k_p - exp_k).std() / math.sqrt(2000)
    if lam == 0.0:
        assert np.allclose(r3.designs[0].propensity, r3.designs[0].propensity[0])


# ------------------------------------------------------------------------------------------------ R2
@pytest.mark.parametrize("world", list(REGIMES_OF))
def test_r2_serial_dependence_and_dither_split(world):
    ds, _ = generate_dataset(world, "R2", 4000, SEED)
    block = ds.meta["block"]
    assert len(np.unique(block)) == gen.DITHER_BLOCKS and np.all(np.diff(block) >= 0)
    for j, d in enumerate(ds.designs):
        x = ds.X_action[:, j]
        assert np.array_equal(x, d.fixed_part + d.random_part)
        for b in range(gen.DITHER_BLOCKS):                              # setpoint constant within a block
            assert np.ptp(d.fixed_part[block == b]) == 0
        ac = np.corrcoef(x[:-1], x[1:])[0, 1]
        ac_r = np.corrcoef(d.random_part[:-1], d.random_part[1:])[0, 1]
        assert ac > 0.5, (world, j, ac)                                 # setpoint -> strong lag-1 dependence
        assert abs(ac_r) < 4 / math.sqrt(4000)                          # dither is i.i.d.
        lo, hi = d.dist["lo"], d.dist["hi"]
        assert d.random_part.min() >= lo and d.random_part.max() <= hi


# ------------------------------------------------------------------------------------------------ placebo
@pytest.mark.parametrize("world,regime", CELLS)
def test_placebo_does_not_touch_anything_else(world, regime):
    a, _ = generate_dataset(world, regime, 300, SEED)
    b, _ = generate_dataset(world, regime, 300, SEED, _with_placebo=False)
    assert b.action_names == a.action_names[:b.X_action.shape[1]]
    assert np.array_equal(a.X_action[:, :b.X_action.shape[1]], b.X_action)
    assert all(nm.startswith("P_placebo") for nm in a.action_names[b.X_action.shape[1]:])
    assert np.array_equal(a.X_kpi_lag, b.X_kpi_lag) and np.array_equal(a.Y, b.Y)


@pytest.mark.parametrize("world", list(REGIMES_OF))
def test_placebo_uncorrelated_with_everything_r1(world):
    ds, _ = generate_dataset(world, "R1", 8000, SEED)
    assert ds.action_names[-1] == gen.PLACEBO
    pl = ds.X_action[:, -1]
    others = np.column_stack([ds.X_action[:, :-1], ds.X_kpi_lag, ds.Y])
    r = np.array([np.corrcoef(pl, others[:, c])[0, 1] for c in range(others.shape[1])])
    assert np.all(np.abs(r) < 4.5 / math.sqrt(8000)), r


@pytest.mark.parametrize("world", list(REGIMES_OF))
def test_placebo_dither_uncorrelated_r2(world):
    ds, _ = generate_dataset(world, "R2", 8000, SEED)
    rp = ds.designs[-1].random_part
    others = np.column_stack([ds.X_action[:, :-1], ds.X_kpi_lag, ds.Y])
    r = np.array([np.corrcoef(rp, others[:, c])[0, 1] for c in range(others.shape[1])])
    assert np.all(np.abs(r) < 4.5 / math.sqrt(8000)), r


@pytest.mark.parametrize("regime", ["R3", "R4"])
@pytest.mark.parametrize("lam", [0.5, 1.5])
def test_r10_tuning_placebo_independent_and_marginal_conf_placebo_confounded(regime, lam):
    """Ruling R-10: in R3 / R4, P_placebo is i.i.d. from P0's marginal and independent of everything (Z included);
    P_placebo_conf follows the Z policy (so it is associated with K through Z) and is not in the primary set."""
    n = 8000
    ds, tr = generate_dataset("E4", regime, n, SEED, lam=lam)
    a, pl, cf = ds.X_action[:, 0], ds.X_action[:, 1], ds.X_action[:, 2]
    z = ds.context[:, 0] if regime == "R3" else None
    others = [a, cf, ds.X_kpi_lag[:, 0], ds.Y[:, 0]] + ([z] if z is not None else [])
    r = np.array([np.corrcoef(pl, o)[0, 1] for o in others])
    assert np.all(np.abs(r) < 4.5 / math.sqrt(n)), r
    # same marginal as P0 (two-sample: mean, sd, P(A in {0, 1}))
    for stat in (np.mean, np.std, lambda x: np.mean((x == 0) | (x == 1))):
        assert abs(stat(pl) - stat(a)) < 0.03, (stat, stat(pl), stat(a))
    d = ds.designs[1]
    assert d.kind == "iid"
    if regime == "R3":
        p = np.array(d.dist["p"])
        assert np.isclose(p.sum(), 1.0) and d.dist["values"] == e4_logged.GRID.tolist()
        mean_k = (p * np.arange(101)).sum()
        assert abs(e4_logged.grid_index(pl).mean() - mean_k) < 4 * 30 / math.sqrt(n)
        # the marginal table is the Z-average of the policy table
        zz = np.random.default_rng(0).standard_normal(20000)
        assert np.allclose(e4_logged.policy_probs(zz, lam).mean(axis=0), p, atol=3e-3)
    else:
        assert d.dist["name"] == "clipped_normal" and np.isclose(d.dist["sd"], math.sqrt(lam ** 2 + 0.25))
    # confounded diagnostic placebo: wrong-sign association with K, outside the primary family
    assert np.corrcoef(cf, ds.Y[:, 0])[0, 1] > 0.1
    assert all(c[0] != gen.PLACEBO_CONF for c in ds.meta["primary_candidates"])
    assert set(ds.meta["diagnostic_candidates"]) == {(gen.PLACEBO_CONF, "K0")} <= tr.null_edges


def test_e5_observation_noise_level():
    ds, _ = generate_dataset("E5", "R1", 4000, SEED)
    # K_mid (K2) latent = P0 exactly, so the observed residual is the added noise
    resid = ds.Y[:, 2] - ds.X_action[:, 0]
    assert abs(resid.std() / (gen.KAPPA_E5 * gen.E5_SIGMA[2]) - 1) < 0.05


# ------------------------------------------------------------------------------------------------ score
def _res(edges):
    return api.Result("m", "0", tuple(api.EdgeResult(s, t, sc, None, sg, d) for s, t, sc, sg, d in edges), 0.0, {})


def test_score_definitions():
    tr = truth_for("E1", "R1")
    ds, _ = generate_dataset("E1", "R1", 50, SEED)
    res = _res([("P0", "K0", 0.9, 1, True), ("P1", "K1", 0.8, -1, True), ("P0", "K1", 0.7, 1, True),
                ("P_placebo", "K2", 0.6, 1, True), ("K0", "K2", 0.5, 1, True), ("K3", "K3", 0.1, 0, False)])
    s = score(res, tr, candidates=ds.candidates, kpi_sources=ds.kpi_names)
    assert s["placebo_conf_declared"] == 0 and math.isnan(s["placebo_conf_fpr"])
    assert (s["n_declared"], s["tp"], s["fp"]) == (3, 2, 1)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 4)
    assert s["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5)) and s["fdp"] == pytest.approx(1 / 3)
    assert s["null_fpr"] == pytest.approx(1 / 12) and s["sign_acc"] == 0.5 and s["sign_n"] == 2
    assert s["placebo_declared"] == 1 and s["placebo_fpr"] == pytest.approx(1 / 4)
    assert s["n_missing"] == 16 - 3 and s["secondary"]["tp"] == 1 and s["secondary"]["recall"] == 0.5
    assert s["all"]["tp"] == 3 and s["all"]["n_candidates"] == 32
    empty = score(_res([]), tr, candidates=ds.candidates)
    assert math.isnan(empty["precision"]) and empty["fdp"] == 0 and empty["recall"] == 0
    with pytest.raises(ValueError):
        score(_res([("P9", "K0", 1.0, 0, True)]), tr, candidates=ds.candidates)
    d4, t4 = generate_dataset("E4", "R4", 50, SEED, lam=1.5)
    s4 = score(_res([("P_placebo_conf", "K0", 1.0, 1, True), ("P0", "K0", 1.0, 1, True)]), t4,
               candidates=d4.candidates, kpi_sources=d4.kpi_names)
    assert (s4["placebo_conf_declared"], s4["placebo_conf_fpr"], s4["placebo_declared"]) == (1, 1.0, 0)
    assert (s4["n_declared"], s4["tp"], s4["fp"], s4["n_candidates"]) == (1, 1, 0, 1)
    assert s4["all"]["n_candidates"] == 2 and s4["sign_acc"] == 0.0


def test_score_not_testable_r23():
    """R-23: listed not-testable candidates count as NOT declared; counts reported per block."""
    tr = truth_for("E1", "R1")
    ds, _ = generate_dataset("E1", "R1", 50, SEED)
    nan = float("nan")
    edges = [("P0", "K0", 0.9, 1, True), ("P1", "K1", nan, 0, True),        # true edge, not testable (flag ignored)
             ("P2", "K2", nan, 0, False),                                   # true edge, not testable
             ("P0", "K1", nan, 0, True),                                    # null, not testable: no false positive
             ("P_placebo", "K3", nan, 0, False), ("K0", "K2", nan, 0, False)]
    base = _res(edges)
    listed = ["P1->K1", "P2->K2", "P0->K1", "P_placebo->K3", "K0->K2"]
    res = api.Result("m", "0", base.edges, 0.0, {}, notes={"not_testable_edges": listed})
    s = score(res, tr, candidates=ds.candidates, kpi_sources=ds.kpi_names)
    assert (s["n_declared"], s["tp"], s["fp"]) == (1, 1, 0)
    assert s["recall"] == pytest.approx(1 / 4)                              # the 2 not-testable true edges = misses
    assert (s["n_not_testable_true"], s["n_not_testable_null"]) == (2, 1)
    assert (s["secondary"]["n_not_testable_true"], s["secondary"]["n_not_testable_null"]) == (1, 0)
    assert (s["all"]["n_not_testable_true"], s["all"]["n_not_testable_null"]) == (3, 1)
    assert s["placebo_not_testable"] == 1 and s["n_not_testable_overridden"] == 2
    assert s["not_testable_edges"] == sorted(listed) and "P0->K1" not in s["declared_edges"]
    # R-22 form: notes["not_testable"] as a dict keyed by edge (reasons), and (src, tgt) pairs
    for notes in ({"not_testable": {k: "exact fit" for k in listed}},
                  {"not_testable_edges": [tuple(k.split("->")) for k in listed]}):
        s2 = score(api.Result("m", "0", base.edges, 0.0, {}, notes=notes), tr, candidates=ds.candidates,
                   kpi_sources=ds.kpi_names)
        assert {k: s2[k] for k in ("tp", "fp", "n_not_testable_true", "n_not_testable_null")} == \
            {"tp": 1, "fp": 0, "n_not_testable_true": 2, "n_not_testable_null": 1}
    # nothing listed: every record still carries zero counts
    s0 = score(_res(edges[:1]), tr, candidates=ds.candidates, kpi_sources=ds.kpi_names)
    assert (s0["n_not_testable_true"], s0["n_not_testable_null"], s0["n_not_testable_overridden"]) == (0, 0, 0)
    for bad in ({"not_testable_edges": "P0->K0"}, {"not_testable_edges": ["P0K0"]},
                {"not_testable_edges": ["P9->K0"]}):
        with pytest.raises(ValueError):
            score(api.Result("m", "0", base.edges, 0.0, {}, notes=bad), tr, candidates=ds.candidates)


def test_merge_summary_reports_not_testable(tmp_path):
    spec = {"methods": ["dummy"], "worlds": ["E1"], "regimes": ["R1"], "ns": [100], "seeds": [SEED, SEED + 1],
            "kappas": [0]}
    out = str(tmp_path / "r.jsonl")
    runner.run_part(spec, 0, 1, out, log=lambda s: None)
    summ = runner.merge(spec, [out], str(tmp_path / "m.jsonl"))
    cell = summ["per_cell"]["dummy|E1|R1|k0|n100"]
    assert cell["n_not_testable_true_total"] == 0 and cell["n_not_testable_null_total"] == 0


def test_placebo_tau_rule():
    """Superseded R-2 rule (``max_declarations``): smallest tau with <= k placebo declarations."""
    rs = [_res([("P_placebo", "K0", 0.3, 0, False), ("P_placebo", "K1", 0.9, 0, False), ("P0", "K0", 5, 0, 0)]),
          _res([("P_placebo", "K0", 0.5, 0, False), ("P_placebo", "K1", float("nan"), 0, False)])]
    tau = placebo_tau(rs, max_declarations=1)
    assert tau == 0.5                                                   # 2nd largest placebo score
    n_pl = sum(e.declared for r in rs for e in apply_threshold(r, tau).edges if e.source == "P_placebo")
    assert n_pl == 1
    assert placebo_tau(rs[:1], max_declarations=2) == -math.inf


@pytest.mark.parametrize("m", [1, 3, 18, 19, 20, 39, 40, 200])
def test_placebo_tau_conformal_r29(m):
    """R-29: tau = ceil((M+1)(1-.05))-th smallest of the M finite placebo scores (largest if the index exceeds M);
    NaN ignored; other sources ignored; M = 0 -> +inf."""
    sc = list(np.random.default_rng(m).permutation(m) + 1.0)                    # scores 1..M
    rs = [_res([("P_placebo", f"K{i % 3}", v, 0, False) for i, v in enumerate(sc)] +
               [("P_placebo", "K0", float("nan"), 0, False), ("P0", "K0", 1e9, 0, False)])]
    k = math.ceil((m + 1) * 0.95 - 1e-9)
    assert placebo_tau(rs) == (float(k) if k <= m else float(m))
    if m == 19:
        assert placebo_tau(rs) == 19.0                                    # (19 + 1) * .95 = 19 exactly
    if m == 39:
        assert placebo_tau(rs) == 38.0                                    # ceil(38) = 38: 1 of 39 above tau
    frac = sum(e.declared for e in apply_threshold(rs[0], placebo_tau(rs)).edges if e.source == "P_placebo") / m
    assert frac <= 0.05 or m < 20                                         # capped case: none above the max
    assert placebo_tau([_res([("P_placebo", "K0", float("nan"), 0, False)])]) == math.inf
    assert placebo_tau(rs, alpha=0.5) == float(math.ceil((m + 1) * 0.5 - 1e-9) if math.ceil((m + 1) * 0.5 - 1e-9) <= m
                                               else m)


# ------------------------------------------------------------------------------------------------ runner
def test_runner_end_to_end_resume_and_merge(tmp_path):
    spec = {"methods": ["dummy"], "worlds": ["E1", "E4"], "regimes": ["R1", "R3"], "ns": [200],
            "seeds": [SEED, SEED + 1], "lams": [0.0, 1.5], "kappas": [0.1]}
    n_jobs = 2 * 1 + 2 * 2 * 2                                          # E1: R1; E4: R1, R3 x 2 lambdas
    assert len(runner.all_keys(spec)) == n_jobs
    outs = [str(tmp_path / f"res_{i}.jsonl") for i in range(2)]
    assert sum(runner.run_part(spec, i, 2, outs[i], log=lambda s: None) for i in range(2)) == n_jobs
    assert runner.run_part(spec, 0, 2, outs[0], log=lambda s: None) == 0          # resumable
    summ = runner.merge(spec, outs, str(tmp_path / "merged.jsonl"))
    assert summ["n_records"] == n_jobs and not summ["missing"] and not summ["errors"]
    recs = [json.loads(line) for line in open(tmp_path / "merged.jsonl")]
    r = recs[0]
    for k in ("dataset_sha256", "cpu_s", "gen_cpu_s", "peak_rss_mb", "scores", "edges"):
        assert k in r
    # identical dataset hash across reruns of the same job
    ds, _ = generate_dataset(r["job"]["world"], r["job"]["regime"], 200, r["job"]["seed"],
                             lam=r["job"]["lam"] if r["job"]["lam"] is not None else 1.0, kappa=r["job"]["kappa"])
    assert dataset_hash(ds) == r["dataset_sha256"]


def test_runner_rejects_non_dev_seeds():
    with pytest.raises(ValueError, match="DEV block"):
        runner.dataset_groups({"methods": ["dummy"], "worlds": ["E1"], "regimes": ["R1"], "ns": [100],
                               "seeds": [3_100_000, 3_100_001], "kappas": [0]})


def test_runner_records_errors_without_stopping(tmp_path):
    class Boom:
        name, version = "boom", "0"

        def run(self, data, config):
            raise RuntimeError("boom")

    ds, tr = generate_dataset("E1", "R1", 50, SEED)
    rec = runner.run_one(Boom(), ds, tr, {}, 0.0, "boom|x")
    assert "RuntimeError" in rec["error"]
