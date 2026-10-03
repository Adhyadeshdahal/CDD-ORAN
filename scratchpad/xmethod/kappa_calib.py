"""Calibration of the observation-noise level kappa_E6 (ruling R-24) from EXISTING E6-P discovery v4 DEV data.

  uv run python scratchpad/xmethod/kappa_calib.py build --src DIR[,DIR] --out NPZ          # records -> design matrices
  uv run python scratchpad/xmethod/kappa_calib.py fit --data NPZ --out DIR [--B 200] [--workers 4] [--quick]

DATA. Episode records "e6p-disc-rec/1" of stage dev_v4 (seeds 188000-188019, sub "v4"; Kaggle kernel
bishalpanta/e6p-disc-v4dev-1-a, res_*.jsonl), read only. Panel grid = the obs-only 10 s panel (record "panel",
step_s 10): step k covers the plant window (t_k, t_k + 10]. KPI targets = the per-cell KPI series of the study
(crt_units.KPIS: pv, v, e, rlf, load = lab field "ue") summed over the window's seconds = lab_series rows
t_k .. t_k + 9 (row i = second (i, i + 1]). Targets only on fully scored windows (t_k >= warm-up 120 s).

PREDICTORS (primary set "base"): the cell's own 5 KPIs at lags 1..L_KPI (= 6 steps, 60 s; lags may reach into the
warm-up, which is observed); control-parameter states of the cell and of its CIO neighbours = every panel column of
kind knob_own / knob_nbr (state in force at the START of the window; lag 0 = "current") at lags 0..L_KNOB (= 3);
is_macro. Variants (descriptive, all keep "base"): "lead" + the knob state at the END of the window (start of k+1:
includes changes made during the window; possible reverse leakage, xApps react to KPIs); "nbr" + mean neighbour
(obs_static neighbours) KPIs at lags 1..3 + nbr_act_ue_lag; "time" + t_k; "all" = lead + nbr + time.

MODEL. sklearn HistGradientBoostingRegressor (defaults, random_state 0), one model per KPI pooled over cells; fit on
the even-j episodes (j = seed - 188000), evaluated on the odd-j episodes (primary); "swap" = the reverse; "xfit" =
both directions, out-of-sample predictions pooled. u_k = 1 - R^2_oos (R^2 = 1 - SSE / SST, SST about the test-set
mean); kappa_k = sqrt(u_k / (1 - u_k)) (u >= 1 -> inf). Descriptive "within": SST about per-(episode, cell) test means
(removes between-cell level differences, which lagged KPIs predict trivially).
CI: episode bootstrap, B reps: resample train episodes (within the train half) and test episodes (within the test
half) with replacement, refit, re-evaluate -> percentile 95 % CI of u_k, kappa_k and of the KPI summaries
(median, mean). RNG default_rng([7824, b]) (scratch tag, R-24).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

KPIS = ("pv", "v", "e", "rlf", "load")
FIELD = {"pv": "pv", "v": "v", "e": "e", "rlf": "rlf", "load": "ue"}
STEP = 10
L_KPI, L_KNOB, L_NBR = 6, 3, 3
SEED0, N_DEV = 188000, 20
VARIANTS = ("base", "lead", "nbr", "time", "all")
TAG = 7824


def log(s=""):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ------------------------------------------------------------------------------------------------ build
def _lag(a, L):
    """a (K, C) -> a shifted down by L steps along axis 0 (NaN fill); L < 0 = lead."""
    out = np.full_like(a, np.nan, dtype=float)
    if L > 0:
        out[L:] = a[:-L]
    elif L < 0:
        out[:L] = a[-L:]
    else:
        out[:] = a
    return out


def episode_design(rec):
    from cdd_oran.decision.collect_p import dec
    from cdd_oran.decision.features import panel_from_rec
    ls = rec["lab_series"]
    D = dec(ls["data"]).astype(np.float64)                         # (T, F, C)
    scored = dec(ls["scored"]).astype(bool)
    T, _, C = D.shape
    p = panel_from_rec(rec["panel"])
    tk = np.unique(p.t)
    K = len(tk)
    assert p.n == K * C and np.array_equal(p.t, np.repeat(tk, C)) and np.array_equal(p.cell, np.tile(np.arange(C), K))
    ok = np.array([t + STEP <= T and scored[t:t + STEP].all() for t in tk])
    Y = np.stack([np.stack([D[t:t + STEP, list(ls["fields"]).index(FIELD[k]), :].sum(0) if t + STEP <= T
                            else np.full(C, np.nan) for t in tk]) for k in KPIS], 2)       # (K, C, 5)
    knob_cols = [c for c in p.data if p.kind[c] in ("knob_own", "knob_nbr")]
    knob = {c: p.data[c].reshape(K, C) for c in knob_cols}
    nb = rec["obs_static"]["neighbours"]
    feats = {}
    for i, k in enumerate(KPIS):
        for L in range(1, L_KPI + 1):
            feats[f"base|{k}@-{L}"] = _lag(Y[:, :, i], L)
    for c in knob_cols:
        for L in range(0, L_KNOB + 1):
            feats[f"base|{c}@-{L}"] = _lag(knob[c], L)
        feats[f"lead|{c}@+1"] = _lag(knob[c], -1)
    feats["base|is_macro"] = p.data["is_macro"].reshape(K, C)
    for i, k in enumerate(KPIS):
        nm = np.stack([Y[:, n, i].mean(1) if n else np.full(K, np.nan) for n in nb], 1)
        for L in range(1, L_NBR + 1):
            feats[f"nbr|nbr_{k}@-{L}"] = _lag(nm, L)
    if "nbr_act_ue_lag" in p.data:
        feats["nbr|nbr_act_ue_lag"] = p.data["nbr_act_ue_lag"].reshape(K, C)
    feats["time|t"] = np.repeat(tk[:, None].astype(float), C, 1)
    keep = ok[:, None] & np.ones((K, C), bool)
    names = list(feats)
    X = np.stack([feats[n][keep] for n in names], 1)
    y = Y[keep]
    cell = np.repeat(np.arange(C)[None], K, 0)[keep]
    return names, X, y, cell, int(keep.sum())


def cmd_build(a):
    from cdd_oran.decision.disc_bench import iter_records
    paths = []
    for s in a.src.split(","):
        paths += sorted(glob.glob(os.path.join(s, "**", "res_*.jsonl"), recursive=True)) if os.path.isdir(s) else [s]
    eps, names = {}, None
    for rec in iter_records(paths, stages={"dev"}):
        if rec.get("sub") != "v4" or rec.get("smoke") or not SEED0 <= int(rec["seed"]) < SEED0 + N_DEV:
            continue
        nm, X, y, cell, n = episode_design(rec)
        names = names or nm
        assert nm == names, "feature set differs across episodes"
        eps[int(rec["seed"])] = (X, y, cell)
        log(f"seed {rec['seed']}: rows {n}")
    seeds = sorted(eps)
    assert seeds == list(range(SEED0, SEED0 + N_DEV)), f"expected the 20 DEV-v4 episodes, got {seeds}"
    X = np.concatenate([eps[s][0] for s in seeds]).astype(np.float32)
    y = np.concatenate([eps[s][1] for s in seeds])
    cell = np.concatenate([eps[s][2] for s in seeds])
    ep = np.concatenate([np.full(len(eps[s][1]), s - SEED0) for s in seeds])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, X=X, y=y, cell=cell, ep=ep, names=np.array(names), kpis=np.array(KPIS),
                        sources=np.array(paths))
    log(f"wrote {a.out}: X {X.shape}, y {y.shape}, episodes {len(seeds)}, data sha256 {data_sha(X, y)}")


def data_sha(X, y):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(X, np.float32).tobytes()
                          + np.ascontiguousarray(y, np.float64).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ fit
def _cols(names, variant):
    groups = {"base": {"base"}, "lead": {"base", "lead"}, "nbr": {"base", "nbr"}, "time": {"base", "time"},
              "all": {"base", "lead", "nbr", "time"}}[variant]
    return [i for i, n in enumerate(names) if n.split("|")[0] in groups]


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=0)


def _r2(y, yh, grp=None):
    sse = float(((y - yh) ** 2).sum())
    if grp is None:
        sst = float(((y - y.mean()) ** 2).sum())
    else:
        _, inv = np.unique(grp, return_inverse=True)
        mu = np.bincount(inv, y) / np.bincount(inv)
        sst = float(((y - mu[inv]) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def kappa_of(u):
    u = np.asarray(u, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(u >= 1, np.inf, np.sqrt(np.clip(u, 0, None) / (1 - u)))


def fit_eval(X, Y, ep, cell, tr_eps, te_eps, cols):
    """Fit one HGB per KPI on rows of tr_eps (episode list, repeats = bootstrap weights), evaluate on te_eps.
    Returns {kpi: (yte, yhat, grp)}; grp = test (episode-draw, cell) id for the within-R^2."""
    def rows(eplist):
        idx, g = [], []
        for d, e in enumerate(eplist):
            r = np.flatnonzero(ep == e)
            idx.append(r)
            g.append(d * 1000 + cell[r])
        return np.concatenate(idx), np.concatenate(g)
    itr, _ = rows(tr_eps)
    ite, gte = rows(te_eps)
    out = {}
    for i, k in enumerate(KPIS):
        m = _model().fit(X[itr][:, cols], Y[itr, i])
        out[k] = (Y[ite, i], m.predict(X[ite][:, cols]), gte)
    return out


def _summ(us):
    u = np.array([us[k] for k in KPIS])
    kap = kappa_of(u)
    return {"u": dict(zip(KPIS, u.tolist())), "kappa": dict(zip(KPIS, kap.tolist())),
            "kappa_median": float(np.median(kap)), "kappa_mean": float(np.mean(kap)),
            "kappa_from_median_u": float(kappa_of(np.median(u))), "u_median": float(np.median(u))}


def _point(X, Y, ep, cell, A, Bh, cols):
    """Primary split A -> Bh, swap, cross-fit; pooled and within R^2."""
    res = {}
    f1 = fit_eval(X, Y, ep, cell, A, Bh, cols)
    f2 = fit_eval(X, Y, ep, cell, Bh, A, cols)
    for name, parts in (("primary", [f1]), ("swap", [f2]), ("xfit", [f1, f2])):
        for kind in ("pooled", "within"):
            us = {}
            for k in KPIS:
                y = np.concatenate([p[k][0] for p in parts])
                yh = np.concatenate([p[k][1] for p in parts])
                g = np.concatenate([p[k][2] + 10 ** 6 * j for j, p in enumerate(parts)])
                us[k] = 1.0 - _r2(y, yh, g if kind == "within" else None)
            res[f"{name}/{kind}"] = _summ(us)
    return res


def _boot_one(args):
    X, Y, ep, cell, A, Bh, cols, b = args
    rng = np.random.default_rng([TAG, b])
    tr = list(rng.choice(A, len(A), replace=True))
    te = list(rng.choice(Bh, len(Bh), replace=True))
    f = fit_eval(X, Y, ep, cell, tr, te, cols)
    out = {}
    for kind in ("pooled", "within"):
        out[kind] = {k: 1.0 - _r2(f[k][0], f[k][1], f[k][2] if kind == "within" else None) for k in KPIS}
    return b, out


def cmd_fit(a):
    import sklearn
    t0 = time.time()
    Z = np.load(a.data, allow_pickle=False)
    X, Y, ep, cell, names = Z["X"].astype(np.float64), Z["y"], Z["ep"], Z["cell"], list(Z["names"])
    A = [e for e in range(N_DEV) if e % 2 == 0]
    Bh = [e for e in range(N_DEV) if e % 2 == 1]
    os.makedirs(a.out, exist_ok=True)
    rep = {"schema": "kappa-calib/1", "data": os.path.basename(a.data), "data_sha256": data_sha(Z["X"], Y),
           "n_rows": int(len(Y)),
           "n_episodes": N_DEV, "train_eps": A, "test_eps": Bh, "seeds": [SEED0 + e for e in range(N_DEV)],
           "kpis": KPIS, "model": "HistGradientBoostingRegressor(defaults, random_state=0)",
           "sklearn": sklearn.__version__, "numpy": np.__version__, "L_KPI": L_KPI, "L_KNOB": L_KNOB, "L_NBR": L_NBR,
           "features": {v: [names[i] for i in _cols(names, v)] for v in VARIANTS},
           "target_sd": {k: float(Y[:, i].std()) for i, k in enumerate(KPIS)},
           "target_frac0": {k: float((Y[:, i] == 0).mean()) for i, k in enumerate(KPIS)}, "point": {}}
    variants = ("base",) if a.quick else VARIANTS
    for v in variants:
        tv = time.time()
        rep["point"][v] = _point(X, Y, ep, cell, A, Bh, _cols(names, v))
        pr = rep["point"][v]["primary/pooled"]
        log(f"{v}: kappa {json.dumps({k: round(x, 3) for k, x in pr['kappa'].items()})} median "
            f"{pr['kappa_median']:.3f} ({time.time() - tv:.1f}s)")
    B = a.B
    cols = _cols(names, "base")
    boots = {}
    tb = time.time()
    jobs = [(X, Y, ep, cell, A, Bh, cols, b) for b in range(B)]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for b, o in ex.map(_boot_one, jobs, chunksize=4):
                boots[b] = o
    else:
        for j in jobs:
            b, o = _boot_one(j)
            boots[b] = o
            if b % 10 == 0:
                log(f"boot {b}/{B}")
    rep["boot_secs"] = round(time.time() - tb, 1)
    ci = {}
    for kind in ("pooled", "within"):
        U = np.array([[boots[b][kind][k] for k in KPIS] for b in range(B)])
        K_ = kappa_of(U)
        q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
        ci[kind] = {"u": {k: q(U[:, i]) for i, k in enumerate(KPIS)},
                    "kappa": {k: q(K_[:, i]) for i, k in enumerate(KPIS)},
                    "kappa_median": q(np.median(K_, 1)), "kappa_mean": q(np.mean(K_, 1)),
                    "kappa_from_median_u": q(kappa_of(np.median(U, 1)))}
    rep["boot"] = {"B": B, "rng": f"default_rng([{TAG}, b])", "scheme": "episodes resampled within each half, refit",
                   "ci95": ci}
    rep["secs"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "kappa_calib.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    np.savez_compressed(os.path.join(a.out, "boot_u.npz"),
                        **{kind: np.array([[boots[b][kind][k] for k in KPIS] for b in range(B)])
                           for kind in ("pooled", "within")})
    log(f"done in {rep['secs']} s -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--B", type=int, default=200)
    f.add_argument("--workers", type=int, default=1)
    f.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    {"build": cmd_build, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
