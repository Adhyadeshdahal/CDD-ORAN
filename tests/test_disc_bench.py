"""disc_bench (cdd_oran.decision.disc_bench): the streaming cache + loader rebuild crt_units.UnitData bit for bit (any
H / H_pre), MSCR-CRT v2 gives identical p-values on both, subsets are disjoint / seeded / use the v3 folds at n = 300,
the scoring (chain hits, premise, indirect / overall) and the placebo check. Synthetic records only (fast)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import crt_units_v2 as V2
from cdd_oran.decision import disc_bench as DB
from cdd_oran.decision.edge_score import HYPOTHESES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _write(recs, path, header=True):
    with open(path, "w") as fh:
        if header:
            fh.write(json.dumps({"kind": "header", "schema": "e6p-disc-rec/1"}) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    rng = np.random.default_rng(3)
    recs = CU.synthetic_records(rng, n_ep=12, units_per_ep=50, effect=60.0, family="sleep")
    for r in recs[::3]:                          # a non-numeric / missing ctx value, and a shuffled seed order
        r["units"][0]["ctx"]["own_prb_util"] = None
        r["units"][1]["ctx"].pop("nbr_mean_prb_util")
    recs = recs[::-1]
    f = d / "res_0.jsonl"
    _write(recs[:7], f)
    f2 = d / "res_1.jsonl"
    _write(recs[7:] + [recs[0]], f2, header=False)     # a duplicate episode (deduped within a file only)
    c1, c2 = d / "a.npz", d / "b.npz"
    DB.build_cache([str(f)], str(c1), stages={"dev"}, src="synthA")
    DB.build_cache([str(f2)], str(c2), stages={"dev"}, src="synthB")
    return {"recs": recs, "cache": [str(c1), str(c2)], "dir": d}


def _same(a, b):
    assert a.n == b.n
    for k in ("episode", "seed", "fold", "family", "c", "t0", "mode", "applied", "p", "probs", "step", "sgn", "prev"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert list(a.xapp) == list(b.xapp)
    for dd in ("y", "pre", "ycell", "rel_mask"):
        A, B = getattr(a, dd), getattr(b, dd)
        assert set(A) == set(B)
        assert all(np.array_equal(A[k], B[k]) for k in A), dd
    assert list(a.z) == list(b.z)
    assert all(np.array_equal(a.z[k], b.z[k], equal_nan=True) for k in a.z)
    assert a.meta["dropped"]["window"] == b.meta["dropped"]["window"]


@pytest.mark.parametrize("H,H_pre", [(90, 90), (30, 30), (150, 60), (60, 150)])
def test_loader_rebuilds_build_unit_data_bit_for_bit(synth, H, H_pre):
    recs = sorted(synth["recs"], key=lambda r: (r["seed"], r.get("sub", "")))
    a = CU.build_unit_data(recs, H=H, H_pre=H_pre)
    b = DB.load_pool(synth["cache"], H=H, H_pre=H_pre).unit_data()
    _same(a, b)


def test_subset_and_v2_pvalues_identical(synth):
    recs = sorted(synth["recs"], key=lambda r: (r["seed"], r.get("sub", "")))
    pool = DB.load_pool(synth["cache"])
    ids = [e for e in range(pool.n_eps) if pool.eps["seed"][e] % 2 == 0]
    a = CU.build_unit_data([r for r in recs if r["seed"] % 2 == 0])
    b = pool.unit_data(ids)
    _same(a, b)
    cfg = V2.UnitCRTConfigV2(B=299, min_episodes=1)
    ra, rb = V2.run_crt_units_v2(a, cfg, 5), V2.run_crt_units_v2(b, cfg, 5)
    assert [r["p"] for r in ra["results"]] == pytest.approx([r["p"] for r in rb["results"]], nan_ok=True, abs=0)


def test_cache_dedupes_and_skips_other_stages(synth, tmp_path):
    recs = synth["recs"][:3]
    f = tmp_path / "res_x.jsonl"
    other = dict(recs[0], stage="gt", key=["gt", "", 1])
    _write(recs + [recs[1], other], f)
    s = DB.build_cache([str(f)], str(tmp_path / "x.npz"), stages={"dev"})
    assert s["episodes"] == 3 and s["stages"] == ["dev"]
    with pytest.raises(ValueError):
        DB.build_cache([str(f)], str(tmp_path / "y.npz"), stages={"eval"})
    both = [str(tmp_path / "x.npz"), synth["cache"][0]]                  # cross-file duplicates kept once
    assert DB.load_pool(both).n_eps == len({r["seed"] for r in synth["recs"][:7]} | {r["seed"] for r in recs})
    assert DB.load_pool(both, stages={"eval"}).n_eps == 0 and DB.load_pool(both, stages={"dev"}).n_eps > 0


def _fake_pool(n_v3=1200, n_v2=480):
    """Pool with only episode metadata (for draw_subsets)."""
    n = n_v3 + n_v2
    eps = {"seed": np.arange(n), "sub": np.array(["v3"] * n_v3 + ["v2"] * n_v2), "stage": np.array(["eval"] * n),
           "fold": np.concatenate([np.repeat(np.arange(4), n_v3 // 4), np.repeat(np.arange(4), n_v2 // 4)])}
    return DB.Pool(H=90, H_pre=90, n_cells=4, kpis=CU.KPIS, eps=eps, u={}, ctx_keys=[])


def test_draw_subsets_disjoint_seeded_and_v3_folds():
    pool = _fake_pool()
    for n, R in ((60, 10), (120, 10)):
        s = DB.draw_subsets(pool, n, R, seed=0)
        allv = np.concatenate([x["episodes"] for x in s])
        assert len(s) == R and all(len(x["episodes"]) == n for x in s) and len(np.unique(allv)) == n * R
        s2 = DB.draw_subsets(pool, n, R, seed=0)
        assert all(np.array_equal(x["episodes"], y["episodes"]) for x, y in zip(s, s2, strict=True))
        assert not np.array_equal(DB.draw_subsets(pool, n, R, seed=1)[0]["episodes"], s[0]["episodes"])
    s = DB.draw_subsets(pool, 300, 5)
    assert [x["name"] for x in s] == ["v3fold0", "v3fold1", "v3fold2", "v3fold3", "rand0"]
    assert [x["split"] for x in s[:4]] == [1, 2, 3, 4]
    assert set(pool.eps["sub"][s[4]["episodes"]]) == {"v2"}
    assert len(np.unique(np.concatenate([x["episodes"] for x in s]))) == 1500
    with pytest.raises(ValueError):
        DB.draw_subsets(_fake_pool(0, 100), 60, 2)


def _ref():
    ref = {h: {"status": "NULL", "sign": 0} for h in HYPOTHESES}
    for f, r, k, s in DB.CHAIN:
        ref[(f, r, k)] = {"status": "TRUE", "sign": s}
    ref[("carrier", "own", "e")] = {"status": "TRUE", "sign": 1}
    ref[("ptx", "far", "rlf")] = {"status": "INDET", "sign": 0}
    return ref


def test_subset_metrics():
    ref = _ref()
    decl = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    decl[("sleep", "nbr", "pv")] = {"declared": True, "sign": 1}
    decl[("sleep", "nbr", "load")] = {"declared": True, "sign": -1}         # wrong sign: TP, not a chain hit
    decl[("carrier", "nbr", "v")] = {"declared": True, "sign": 1}           # FP (NULL)
    decl[("ptx", "far", "rlf")] = {"declared": True, "sign": 1}             # INDET: excluded
    m = DB.subset_metrics(decl, ref)
    assert m["chain_true"] == 4 and m["chain_hits"] == 1 and m["premise_hit"] == 1.0
    assert m["ind_tp"] == 2 and m["ind_fp"] == 1
    assert m["ind_recall"] == pytest.approx(0.5) and m["ind_precision"] == pytest.approx(2 / 3)
    assert m["ov_precision"] == pytest.approx(2 / 3) and m["sign_acc"] == pytest.approx(0.5)
    assert m["n_declared"] == 4 and m["far_declared"] == 1


def test_bench_and_placebo_check_run_end_to_end(synth, tmp_path):
    pool = DB.load_pool(synth["cache"])
    calls = []

    def method(data, split):
        calls.append((data.n, split))
        return {h: {"declared": h == DB.PREMISE, "sign": 1, "p": 0.01 if h == DB.PREMISE else 0.5}
                for h in HYPOTHESES}
    res = DB.bench(pool, method, _ref(), sizes=(3, 4), R={3: 3, 4: 2}, placebos={"pl": pool.unit_data([0, 1])})
    assert res["sizes"]["3"]["R"] == 3 and res["sizes"]["4"]["R"] == 2 and len(calls) == 6
    assert res["sizes"]["3"]["summary"]["premise_hit"]["mean"] == 1.0
    assert res["sizes"]["3"]["summary"]["ind_precision"]["sd"] == 0.0
    pc = res["placebo"]["pl"]
    assert pc["n_declared"] == 1 and pc["n_p"] == 60 and pc["rate"] == pytest.approx(1 / 60)


def test_mscr_method_detects_the_planted_edge(synth):
    pool = DB.load_pool(synth["cache"])
    m = DB.method_mscr_v2(B=999, min_episodes=1)
    d = m(pool.unit_data(), 0)
    assert d[("sleep", "nbr", "load")]["p"] < 0.01 and d[("sleep", "nbr", "load")]["sign"] == 1


def test_baseline_methods_and_gt_reference_roundtrip(synth, tmp_path):
    dev = DB.load_pool(synth["cache"]).unit_data()
    from cdd_oran.decision import baselines_disc as BD
    sup = BD.Support(min_episodes=1)
    assert BD.supported(dev, "sleep", sup)
    for name in ("corr", "granger", "granger_by"):
        m = DB.method_baseline(name, dev)
        d = m(dev, 0)
        assert set(d) >= set(HYPOTHESES)
    g = {"ref": _ref(), "counts": {"TRUE": 5}, "n_episodes": 0}
    DB.save_ref(g, str(tmp_path / "r.json"))
    assert DB.load_ref(str(tmp_path / "r.json"))["ref"] == _ref()
