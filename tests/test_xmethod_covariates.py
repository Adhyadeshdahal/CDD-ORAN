"""Shared design-covariate builder (cdd_oran/xmethod/covariates.py, ruling R-17), its use in pmrt_core (bit-identical
to the pre-helper code; R-19 option) and F7 finding 7 (no structure hints in the Dataset)."""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from cdd_oran.xmethod import covariates as C
from cdd_oran.xmethod.methods import pmrt_core as PC
from cdd_oran.xmethod.worlds import REGIMES_OF, generate_dataset, roles_for

SEED = 3_000_000
CELLS = [(w, r) for w, rs in REGIMES_OF.items() for r in rs]
ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- legacy reference (pmrt_core.py @ feat/v2 a25b0a4)
def _legacy_finite_block(M):
    if M is None:
        return []
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    cols = []
    for j in range(M.shape[1]):
        c = M[:, j]
        ok = np.isfinite(c)
        if not ok.any():
            continue
        if ok.all():
            cols.append(c)
        else:
            cols += [np.where(ok, c, 0.0), (~ok).astype(float)]
    return cols


def _legacy_covariates(data, order, hist_lags):
    n = data.n
    cols = _legacy_finite_block(data.X_kpi_lag[order] if data.X_kpi_lag is not None else None)
    if data.context is not None:
        cols += _legacy_finite_block(np.asarray(data.context)[order])
    for d in data.designs:
        if d.kind == "dither" and d.fixed_part is not None:
            cols += _legacy_finite_block(np.asarray(d.fixed_part, float)[order])
    if hist_lags > 0:
        A = np.asarray(data.X_action, float)[order]
        t = None if data.time_index is None else np.asarray(data.time_index)[order]
        for lag in range(1, min(hist_lags, n - 1) + 1):
            ok = np.zeros(n, bool)
            ok[lag:] = True if t is None else (t[lag:] - t[:-lag]) == lag
            L = np.zeros_like(A)
            L[lag:] = A[:-lag]
            L[~ok] = 0.0
            cols += [L[:, j] for j in range(A.shape[1])]
            if t is not None and not ok[lag:].all():
                cols.append((~ok).astype(float))
    return np.column_stack(cols) if cols else np.zeros((n, 0))


def _perturb(ds, seed=7, time_index=True):
    """Shuffled rows (information order != row order), time gaps, partly missing lagged KPIs."""
    rng = np.random.default_rng(seed)
    n = ds.n
    t = np.arange(n) + np.cumsum(rng.random(n) < 0.03) * 3
    perm = rng.permutation(n)
    xk = ds.X_kpi_lag.copy()
    xk[rng.random(xk.shape) < 0.02] = np.nan

    def P(a):
        return None if a is None else np.asarray(a)[perm]
    des = tuple(dataclasses.replace(d, random_part=P(d.random_part), fixed_part=P(d.fixed_part),
                                    propensity=P(d.propensity)) for d in ds.designs)
    return dataclasses.replace(ds, X_action=ds.X_action[perm], X_kpi_lag=xk[perm], Y=ds.Y[perm], designs=des,
                               time_index=t[perm] if time_index else None, context=P(ds.context))


def _variants(world, regime, n=300):
    ds, _ = generate_dataset(world, regime, n, SEED, lam=1.5)
    return [ds, _perturb(ds), _perturb(ds, time_index=False)]


# ---------------------------------------------------------------------------------------------- helper == legacy
@pytest.mark.parametrize("world,regime", CELLS)
@pytest.mark.parametrize("lags", [0, 1, 2, 3])
def test_helper_equals_legacy_pmrt_covariates(world, regime, lags):
    for ds in _variants(world, regime):
        order = C.info_order(ds)
        names, M, mask = C.design_covariates(ds, lags=lags)
        ref = _legacy_covariates(ds, order, lags)
        assert M.shape == ref.shape and np.array_equal(M[order], ref)
        assert len(names) == M.shape[1] and len(set(names)) == len(names) and mask.shape == (ds.n,)
        assert np.array_equal(PC.covariates(ds, order, lags), ref)               # pmrt_core's wrapper


def test_names_blocks_and_encoding():
    ds, _ = generate_dataset("E2", "R2", 300, SEED)
    names, M, mask = C.design_covariates(ds)
    acts = ds.action_names
    assert names == tuple([f"lag_kpi:{k}" for k in ds.kpi_names] + [f"sp:{a}" for a in acts]
                          + [f"{a}@t-{L}" for L in (1, 2) for a in acts])
    j = names.index("sp:P3")
    assert np.array_equal(M[:, j], ds.designs[3].fixed_part)
    assert np.array_equal(M[1:, names.index("P0@t-1")], ds.X_action[:-1, 0]) and M[0, names.index("P0@t-1")] == 0
    assert np.array_equal(mask, np.arange(300) >= 2)                              # first rows: no true lags
    # R3: context, no setpoints (logged designs have no fixed part)
    d3, _ = generate_dataset("E4", "R3", 300, SEED, lam=1.5)
    n3, M3, _ = C.design_covariates(d3)
    assert n3[:2] == ("lag_kpi:K0", "ctx:Z") and not any(x.startswith("sp:") for x in n3)
    assert np.array_equal(M3[:, 1], d3.context[:, 0])
    # gaps and partly missing values get indicators; row_mask marks rows without a true previous step
    p = _perturb(ds)
    npn, Mp, mp = C.design_covariates(p)
    assert "gap@t-1" in npn and "gap@t-2" in npn and any(x.endswith(":missing") for x in npn)
    t = p.time_index
    order = C.info_order(p)
    ts = t[order]
    ok1 = np.r_[False, np.diff(ts) == 1]
    ok2 = np.r_[False, False, (ts[2:] - ts[:-2]) == 2]
    assert np.array_equal(mp[order], ok1 & ok2)
    assert np.array_equal(Mp[order][:, npn.index("gap@t-1")], (~ok1).astype(float))
    assert C.design_covariates(p, include_lagged_actions=False)[2].all()


def test_r3_set_and_flags():
    ds, _ = generate_dataset("E4", "R3", 200, SEED, lam=1.0)
    names, M, mask = C.design_covariates(ds, include_setpoints=False, include_lagged_actions=False)
    assert names == ("lag_kpi:K0", "ctx:Z") and mask.all()
    nameless = dataclasses.replace(ds, meta={k: v for k, v in ds.meta.items() if k != "context_names"})
    assert C.design_covariates(nameless, include_setpoints=False, include_lagged_actions=False)[0][1] == "ctx:c0"
    d2, _ = generate_dataset("E1", "R2", 200, SEED)
    n_no_sp = C.design_covariates(d2, include_setpoints=False)[0]
    assert not any(x.startswith("sp:") for x in n_no_sp) and any(x.endswith("@t-2") for x in n_no_sp)
    assert C.design_covariates(d2, include_kpi_lag=False, include_lagged_actions=False)[0] == tuple(
        f"sp:{a}" for a in d2.action_names)


def test_concurrent_rules():
    r4, _ = generate_dataset("E4", "R4", 200, SEED, lam=1.5)              # P0 none, P_placebo iid, conf none
    assert C.concurrent_indices(r4, 0) == [] and C.concurrent_indices(r4, 1) == []
    assert C.concurrent_indices(r4, 1, "all") == [0, 2]
    r3, _ = generate_dataset("E4", "R3", 200, SEED, lam=1.5)
    names, M, _ = C.design_covariates(r3, focal="P0")
    assert names[-2:] == ("concurrent:P_placebo", "concurrent:P_placebo_conf")
    assert np.array_equal(M[:, -2:], r3.X_action[:, 1:])
    assert C.design_covariates(r3, focal=0)[0] == names
    d, _ = generate_dataset("E2", "R1", 200, SEED)
    for ai in range(len(d.action_names)):
        assert C.concurrent_indices(d, ai) == PC.concurrent_columns(d, ai) == [j for j in range(9) if j != ai]
    with pytest.raises(ValueError):
        C.concurrent_indices(d, 0, "some")


# ---------------------------------------------------------------------------------------------- pmrt_core
CFG = {"B": 199, "seq_h": None}


def _edges(res):
    return repr([(e.source, e.target, e.score, e.p, e.sign, e.declared) for e in res.edges])


@pytest.mark.parametrize("world,regime", [("E2", "R2"), ("E4", "R3"), ("E5", "R1")])
def test_pmrt_core_bit_identical_with_legacy_covariates(world, regime, monkeypatch):
    """The refactored run() with the shared helper == run() fed the legacy covariate builder, bit for bit
    (full production-config proof on 16 datasets: scratchpad/xmethod/results/cov_equiv.jsonl)."""
    for ds in _variants(world, regime, n=400)[:2]:
        new = PC.PmrtCore().run(ds, CFG)

        def legacy(data, include_setpoints=True, include_lagged_actions=True, lags=2, **kw):
            assert include_setpoints and include_lagged_actions and not kw
            M_info = _legacy_covariates(data, C.info_order(data), lags)
            inv = np.empty(data.n, np.int64)
            inv[C.info_order(data)] = np.arange(data.n)
            return tuple(f"c{j}" for j in range(M_info.shape[1])), M_info[inv], np.ones(data.n, bool)
        monkeypatch.setattr(PC, "design_covariates", legacy)
        old = PC.PmrtCore().run(ds, CFG)
        monkeypatch.undo()
        assert _edges(new) == _edges(old)
        assert repr(new.notes["z"]) == repr(old.notes["z"]) and repr(new.notes["adjust"]) == repr(old.notes["adjust"])


def test_pmrt_core_r3_option():
    ds, _ = generate_dataset("E2", "R2", 400, SEED)
    assert PC.PmrtCoreConfig().covariates == "eq"
    eq = PC.PmrtCore().run(ds, CFG)
    r3 = PC.PmrtCore().run(ds, {**CFG, "covariates": "r3"})
    assert eq.notes["covariates"]["set"] == "eq" and r3.notes["covariates"]["set"] == "r3"
    base = r3.notes["covariates"]["base"]
    assert base == [f"lag_kpi:{k}" for k in ds.kpi_names]                   # R-3: no setpoints, no lagged actions
    assert any(x.startswith("sp:") for x in eq.notes["covariates"]["base"])
    assert eq.config["covariates"] == "eq" and r3.config["covariates"] == "r3"
    adj_eq, adj_r3 = eq.notes["adjust"]["P0"], r3.notes["adjust"]["P0"]
    assert adj_r3["d"] == len(base) + len(adj_r3["concurrent"]) < adj_eq["d"]
    assert eq.notes["z"] != r3.notes["z"]
    with pytest.raises(ValueError):
        PC.PmrtCore().run(ds, {**CFG, "covariates": "full"})


# ---------------------------------------------------------------------------------------------- F7 finding 7
@pytest.mark.parametrize("world,regime", CELLS)
def test_no_structure_hints_in_dataset(world, regime):
    ds, _ = generate_dataset(world, regime, 50, SEED)
    assert "roles" not in ds.meta
    assert set(ds.meta) <= {"generator", "warmup", "env_seed", "episode", "placebo", "placebo_conf", "block",
                            "dither", "context_names", "realised_propensity", "obs_noise", "lam",
                            "primary_candidates", "secondary_candidates", "diagnostic_candidates"}


def test_roles_harness_private_and_unread_by_methods():
    assert roles_for("E5")["P1"] == "G1 gate axis" and roles_for("E1") == {}
    srcs = list((ROOT / "cdd_oran" / "xmethod" / "methods").glob("*.py")) + [ROOT / "cdd_oran" / "xmethod" /
                                                                             "covariates.py"]
    assert srcs
    for f in srcs:
        text = f.read_text(encoding="utf-8")
        assert "roles" not in text, f
    cov_src = (ROOT / "cdd_oran" / "xmethod" / "covariates.py").read_text(encoding="utf-8")
    assert cov_src.count(".meta") == 1 and '.get("context_names")' in cov_src      # context names only
