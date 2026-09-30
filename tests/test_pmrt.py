"""PMRT (cdd_oran.decision.pmrt; formerly crt_units_plus): the per-bin features rebuild the v2 target exactly, the
history / running-centre / weights are PREDICTABLE (dropping later units or changing later outcomes never changes an
earlier unit's weight), the running-centred linear statistic is unbiased on a MODE-DEPENDENT skeleton where the
realised-set (episode x sgn) FE statistic is not, the CRT is chunk-invariant and calibrated under a planted null,
the exact max-combination p is >= the p of the arm attaining the max, and a planted nbr effect learned on training
episodes is detected on fresh episodes. Synthetic records only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import disc_bench as DB
from cdd_oran.decision import pmrt as PM
from cdd_oran.decision.crt_units_v2 import residualise

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::UserWarning")

FAST = PM.PmrtConfig(B=499, cv_reps=1, gb=False)


def _pd(tmp_path, recs, name="a"):
    f = str(tmp_path / f"{name}.npz")
    PM.cache_records(recs, f)
    pool = PM.load_pmrt_pool([f])
    return pool, PM.pmrt_data(pool)


@pytest.fixture(scope="module")
def train_test(tmp_path_factory):
    d = tmp_path_factory.mktemp("plus")
    tr = CU.synthetic_records(np.random.default_rng(11), n_ep=16, units_per_ep=70, effect=40.0, family="sleep",
                              seed0=5000)
    te = CU.synthetic_records(np.random.default_rng(12), n_ep=12, units_per_ep=70, effect=40.0, family="sleep",
                              seed0=6000)
    nul = CU.synthetic_records(np.random.default_rng(13), n_ep=12, units_per_ep=70, effect=0.0, family="sleep",
                               seed0=7000)
    _, ptr = _pd(d, tr, "tr")
    _, pte = _pd(d, te, "te")
    _, pnu = _pd(d, nul, "nu")
    params = PM.fit_pmrt(ptr, FAST)
    return {"tr": ptr, "te": pte, "nu": pnu, "params": params, "recs_te": te, "dir": d}


# ------------------------------------------------------------------------------------------------ features
def test_bins_rebuild_the_v2_target_and_unit_table(train_test, tmp_path):
    pd = train_test["te"]
    ud0 = DB.load_pool([PM.cache_records(train_test["recs_te"], str(tmp_path / "x.npz"))], H=90).unit_data()
    for k in ("episode", "c", "t0", "mode", "sgn", "family"):
        assert np.array_equal(getattr(pd.ud, k), getattr(ud0, k))
    for rel in pd.ud.relations:
        for j, kpi in enumerate(pd.ud.kpis):
            yy = ((pd.post[:, j, :, :3].sum(-1) - pd.pre[:, j, :, :3].sum(-1)) * pd.ud.rel_mask[rel]).sum(1)
            assert np.allclose(yy, pd.ud.y[(rel, kpi)], rtol=1e-4, atol=1e-2 * (1 + np.abs(pd.ud.y[(rel, kpi)]).mean()))
    late = pd.ud.t0 + 150 > 720
    assert np.all(np.isnan(pd.post[late][..., 3])) and np.all(np.isfinite(pd.post[~late][..., 3]))


def test_hist_is_past_only(train_test):
    pd = train_test["te"]
    ud = pd.ud
    i = int(np.nonzero(np.abs(pd.hist).sum(1) > 0)[0][0])
    same_ep = np.nonzero(ud.episode == ud.episode[i])[0]
    v = PM.v_design(ud.mode, ud.probs, ud.sgn)
    fi = ud.family
    exp_own = [(ud.c[j] == ud.c[i]) and 0 < ud.t0[i] - ud.t0[j] <= PM.HIST_WIN for j in same_ep]
    for f in range(len(CU.FAMILIES)):
        want = sum(v[j] for j, ok in zip(same_ep, exp_own, strict=True) if ok and fi[j] == f)
        # kept units only here; hist also counts window-dropped units of the episode (none before t0 >= 120 here)
        assert np.isclose(pd.hist[i, f], want)


def test_slot_matrix_ranks_relation_cells():
    rel = np.array([[True, True, True, False], [False, True, True, True]])
    score = np.array([[0, 1, 5, 9], [0, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]], float)
    A = PM.slot_matrix(np.array([0, 1]), rel, score, 3)
    assert A[0, 0].tolist() == [0, 0, 1, 0] and A[0, 1].tolist() == [0, 1, 0, 0] and A[0, 2].tolist() == [1, 0, 0, 0]
    assert A[1, 0].tolist() == [0, 1, 0, 0] and A[1, 1].tolist() == [0, 0, 1, 0] and A[1, 2].tolist() == [0, 0, 0, 1]
    assert np.array_equal(A.sum(1), rel.astype(float))


# ------------------------------------------------------------------------------------------------ predictability
def _naive_center(k, ep, sg, n0, n1, prior):
    out = np.zeros(len(k))
    for u in range(len(k)):
        g = 1 if sg[u] > 0 else -1
        pp = [k[j] for j in range(u) if (1 if sg[j] > 0 else -1) == g]
        pm = (sum(pp) + n1 * prior[g][0]) / (len(pp) + n1)
        ss = [k[j] for j in range(u) if ep[j] == ep[u] and (1 if sg[j] > 0 else -1) == g]
        out[u] = k[u] - (sum(ss) + n0 * pm) / (len(ss) + n0)
    return out


def test_running_center_matches_naive_and_is_predictable():
    rng = np.random.default_rng(1)
    n = 200
    k, ep, sg = rng.normal(size=n), np.sort(rng.integers(0, 6, n)), rng.choice([-1.0, 1.0], n)
    prior = {1: [0.3], -1: [-0.2]}
    r = PM.running_center(k, ep, sg, 7.0, 13.0, prior)
    assert np.allclose(r, _naive_center(k, ep, sg, 7.0, 13.0, prior))
    k2 = k.copy()
    k2[120:] += 1e6
    assert np.array_equal(PM.running_center(k2, ep, sg, 7.0, 13.0, prior)[:120], r[:120])
    assert np.array_equal(PM.running_center(k[:90], ep[:90], sg[:90], 7.0, 13.0, prior), r[:90])


def test_family_weights_ignore_later_units(train_test):
    """Dropping every unit after an information-order cut leaves the earlier units' weights (all arms) unchanged."""
    pd, params = train_test["te"], train_test["params"]
    ud = pd.ud
    order = np.lexsort((ud.c, ud.t0, ud.episode))
    cut = order[: len(order) * 2 // 3]
    sub = PM.subset_pmrt(pd, np.sort(cut))
    for f in ("sleep", "prot_min"):
        rows, W, _, _ = PM.family_columns(pd, f, params, FAST)
        rows_s, W_s, _, _ = PM.family_columns(sub, f, params, FAST)
        m = len(rows_s)
        assert m > 10
        key = list(zip(ud.episode[rows[:m]], ud.t0[rows[:m]], ud.c[rows[:m]], strict=True))
        key_s = list(zip(sub.ud.episode[rows_s], sub.ud.t0[rows_s], sub.ud.c[rows_s], strict=True))
        assert key == key_s
        assert np.allclose(W[:m], W_s, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------------------------------------ validity
P = np.array([0.5, 0.2, 0.3])
LV = np.array([1.0, 1.0, 0.0])


def _episode(rng, T=720, H=90, thr=0.8, dwell=150, rho=0.995):
    """Mode-dependent skeleton (as tests/test_eprocess_units.py): reject -> the request re-opens a unit 60 s later,
    applied -> dwell; the outcome series is fixed before any mode (sharp null)."""
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
    return [(t0, g, m, Y[t0:t0 + H].sum() - Y[t0 - H:t0].sum()) for t0, g, m in units]


def test_running_centre_unbiased_where_realised_fe_is_not():
    rng = np.random.default_rng(7)
    m0 = float(P @ LV)
    var = float(P @ LV ** 2) - m0 ** 2
    z_fe, z_rc = [], []
    for _ in range(300):
        rows = [(e,) + u for e in range(20) for u in _episode(rng)]
        a = np.array(rows, float).reshape(-1, 5)
        ep, sg, mode, y = a[:, 0].astype(int), a[:, 2], a[:, 3].astype(int), a[:, 4]
        v = sg * (LV[mode] - m0)
        r_fe, _ = residualise(y, None, ep * 2 + (sg > 0))
        r_rc = PM.running_center(y, ep, sg, 20.0, 50.0, {1: [0.0], -1: [0.0]})
        z_fe.append(v @ r_fe / np.sqrt(var * (r_fe @ r_fe)))
        z_rc.append(v @ r_rc / np.sqrt(var * (r_rc @ r_rc)))
    se = 1.0 / np.sqrt(len(z_rc))
    assert abs(np.mean(z_rc)) < 3 * se * 1.2
    assert abs(np.mean(z_fe)) > abs(np.mean(z_rc)) + 2 * se         # the realised-set FE statistic is biased here


def test_crt_calibrated_on_planted_null_and_chunk_invariant(train_test):
    pd, params = train_test["nu"], train_test["params"]
    run = PM.run_pmrt(pd, params, FAST)
    ps = [r["arms"][a]["p"] for r in run["results"] if r["status"] == "tested" for a in ("best", "plain", "pred")]
    assert np.mean(np.array(ps) <= 0.05) <= 0.15
    assert sum(e["declared"] for e in PM.edges_pmrt(run, "best").values()) <= 1
    run2 = PM.run_pmrt(pd, params, dataclasses.replace(FAST, max_chunk_bytes=50_000))
    for r1, r2 in zip(run["results"], run2["results"], strict=True):
        if r1["status"] == "tested":
            assert {a: v["p"] for a, v in r1["arms"].items()} == {a: v["p"] for a, v in r2["arms"].items()}


def test_max_combination_p_dominates_its_argmax_arm(train_test):
    run = PM.run_pmrt(train_test["te"], train_test["params"], FAST)
    for r in run["results"]:
        if r["status"] != "tested":
            continue
        for c, arms in PM.COMBOS.items():
            via = r["arms"][c]["via"]
            assert via in arms
            assert r["arms"][c]["p"] >= r["arms"][via]["p"] - 1e-12
            assert r["arms"][c]["p"] >= min(r["arms"][a]["p"] for a in arms) - 1e-12


def test_planted_effect_detected_on_fresh_episodes(train_test):
    run = PM.run_pmrt(train_test["te"], train_test["params"], dataclasses.replace(FAST, B=4999))
    e = PM.edges_pmrt(run, "best")
    for h in (("sleep", "nbr", "load"), ("sleep", "nbr", "pv")):
        assert e[h]["declared"] and e[h]["sign"] == 1, (h, e[h])
    far = [h for h, v in e.items() if v["declared"] and h[1] == "far"]
    assert not far
