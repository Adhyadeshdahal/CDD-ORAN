"""Experiment B driver (cdd_oran/xmethod/exp_b.py, e6_bridge.py) end to end on a SYNTHETIC unit table in the
xm-e6units/1 schema (no E6 data needed): counts -> run -> merge -> table. Planted: sleep -> nbr KPIs (+)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cdd_oran.xmethod import e6_bridge as B

pytest.importorskip("cdd_oran.xmethod.exp_b")
from cdd_oran.xmethod import exp_b as X  # noqa: E402

try:
    X._campaign()
    HAVE_CAMPAIGN = True
except ImportError:
    HAVE_CAMPAIGN = False


def fake_table(path: str, stage: str, seeds, per_ep: int = 12, effect: float = 2.0, rng_seed: int = 0) -> None:
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in seeds:
        for fi in range(len(B.FAMILIES)):
            t0 = np.sort(rng.integers(90, 630, per_ep))
            for t in t0:
                rows.append((s, fi, t))
    n = len(rows)
    seed, fam, t0 = (np.array(c) for c in zip(*rows, strict=True))
    sgn = rng.choice([-1.0, 1.0], n)
    mode = rng.choice([0, 2], n)                                # accept / reject (v4 pi0 .5 / .5)
    level = (mode == 0).astype(float)
    probs = np.tile([0.5, 0.0, 0.5, 0.0], (n, 1))
    pre = rng.normal(10, 2, (n, 15))
    y = rng.normal(0, 1, (n, 15))
    if stage == "eval":                                         # placebo logs: accept applied, no effect
        nbr = [j for j, (r, _) in enumerate(B.TARGETS) if r == "nbr"]
        sl = fam == B.FAMILIES.index("sleep")
        y[np.ix_(sl, nbr)] += effect * (sgn * level)[sl, None]
    np.savez_compressed(path, schema=np.array(B.E6UNITS_SCHEMA), seed=seed.astype(np.int64),
                        episode=seed.astype(np.int64), family=fam.astype(np.int8), t0=t0.astype(np.int32),
                        c=rng.integers(0, 24, n).astype(np.int16), sgn=sgn.astype(np.float32),
                        mode=mode.astype(np.int8), level=level.astype(np.float32), p1=np.full(n, 0.5),
                        probs=probs, pre=pre, y=y, ctx=rng.normal(size=(n, 3)), ctx_keys=np.array(["a", "b", "c"]),
                        hist=rng.normal(size=(n, 8)), tfrac=t0 / 720.0, families=np.array(B.FAMILIES),
                        targets=np.array([f"{r}|{k}" for r, k in B.TARGETS]), stage=np.array(stage))


def fake_gt(path: str) -> None:
    cells = []
    for f in B.FAMILIES:
        for r in B.RELATIONS:
            for k in B.KPIS:
                st = "TRUE" if (f == "sleep" and r == "nbr") else ("NULL" if r != "far" else "INDET")
                cells.append({"family": f, "relation": r, "kpi": k, "status": st, "sign": 1,
                              "ci": [-1.0, 1.0] if st == "NULL" else [1.0, 2.0]})
    json.dump({"gt": {"cells": cells}}, open(path, "w"))


def test_dataset_mapping(tmp_path):
    p = tmp_path / "e6units_eval.npz"
    fake_table(str(p), "eval", range(188100, 188110))
    t = B.Table(str(p))
    ds = B.make_dataset(t, "sleep", range(188100, 188110), 20, "s60_0")
    assert ds.n == 120 and ds.X_action.shape == (120, 2) and ds.Y.shape == (120, 15)
    assert set(np.unique(ds.X_action[:, 0])) <= {-1.0, 0.0, 1.0}
    assert np.allclose(ds.designs[0].propensity.sum(1), 1.0)
    sg = ds.context[:, -2]                                      # zero propensity on the impossible sign
    assert np.all(ds.designs[0].propensity[sg > 0, 0] == 0) and np.all(ds.designs[0].propensity[sg < 0, 2] == 0)
    assert len(ds.candidates) == 30 and ds.seed == B.dataset_seed(20, "sleep")
    assert np.all(np.diff(ds.time_index) != 0)


def test_context_encoding_and_lag_identity(tmp_path):
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    part = np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n))
    zero_obs = np.where(rng.random(n) < 0.3, np.nan, 0.0)           # observed only as 0: = 0 * (1 - indicator)
    M = np.column_stack([x, np.full(n, np.nan), np.ones(n), 2 * x + 1, part, zero_obs])
    C, names = B.encode_context(M, ["x", "allnan", "const", "dup", "part", "zero_obs"])
    assert names == ("x", "part", "part:missing", "zero_obs:missing")
    assert np.isfinite(C).all() and C.shape == (n, 4)
    # E6 load identity: own + nbr + far load = const -> the far-load lag is blanked, the target kept
    p = tmp_path / "e6units_eval.npz"
    fake_table(str(p), "eval", range(188100, 188110))
    z = dict(np.load(p))
    li = [j for j, (_, k) in enumerate(B.TARGETS) if k == "load"]
    z["pre"][:, li[2]] = 27000.0 - z["pre"][:, li[0]] - z["pre"][:, li[1]]
    np.savez_compressed(p, **z)
    ds = B.make_dataset(B.Table(str(p)), "ptx", range(188100, 188110), 20, "s60_0")
    assert ds.meta["lag_dropped_collinear"] == ["K_far_load"]
    assert np.isnan(ds.X_kpi_lag[:, li[2]]).all() and np.isfinite(ds.Y).all()


@pytest.mark.skipif(not HAVE_CAMPAIGN, reason="campaign.py not in this checkout (set XM_CAMPAIGN_FILE)")
def test_end_to_end(tmp_path):
    tabs = tmp_path / "tables"
    tabs.mkdir()
    fake_table(str(tabs / "e6units_eval.npz"), "eval", range(188100, 188220), per_ep=6)   # s60_0, s60_1
    fake_table(str(tabs / "e6units_placebo.npz"), "placebo", range(188700, 188740), per_ep=6, rng_seed=1)
    fake_gt(str(tmp_path / "gt.json"))
    spec = {"name": "t", "budget_cpu_s": 600, "slices": ["60", "placebo"], "gt": str(tmp_path / "gt.json"),
            "counts": str(tmp_path / "counts.json"),
            "arms": {"corr": {"ref": "cdd_oran.xmethod.methods.corr:Corr", "config": {"arm": "native"},
                              "declare": "by"},
                     "pcorr_eq": {"ref": "cdd_oran.xmethod.methods.pcorr:PCorrMethod", "config": {"arm": "eq"},
                                  "declare": "by"},
                     "pmrt_eq": {"ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore",
                                 "config": {"covariates": "eq"}, "declare": "by"}}}
    c = X.counts_from_tables(spec, str(tabs))                   # s60_2..9 have no data here: 0 units, skipped
    json.dump({"counts": c}, open(spec["counts"], "w"))
    assert c["s60_0"]["sleep"] == 360 and c["placebo"]["sleep"] == 240 and c["s60_5"]["sleep"] == 0
    units = X.expand(spec, c)
    assert len(units) == 3 * 4 * 3 and len(X.slices(spec)) == 11
    out = str(tmp_path / "res.jsonl")
    C = X._campaign()
    loader = X.Loader(spec, str(tabs), spec["gt"])
    old = C.generate_dataset
    C.generate_dataset = loader
    try:
        C.run_units(X.campaign_spec(spec), units, out, 600, isolate=False)
    finally:
        C.generate_dataset = old
    recs = [json.loads(x) for x in open(out)]
    assert len(recs) == len(units) and all(r["status"] == "ok" for r in recs), [r.get("error") for r in recs][:1]
    assert {r["job"]["seed"] for r in recs} == {u.seed for u in units}
    m = str(tmp_path / "merged.jsonl")
    summ = X.merge(spec, [out], m)
    assert summ["n_ok"] == len(units) and not summ["errors"] and not summ["missing"]
    t = X.table(spec, m, spec["gt"], str(tmp_path / "tab"))
    cell = t["cells"]["pmrt_eq|60|raw"]
    assert cell["rec"]["rate"] > 0.8 and cell["null"]["n"] > 0
    assert t["cells"]["pmrt_eq|placebo|raw"]["plac_log"]["n"] == 4 * 15
    assert os.path.exists(tmp_path / "tab" / "EXP_B_TABLES.md")
