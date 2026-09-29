"""Analysis per PILOT_DESIGN.md (no tuning).
  python scratchpad/decision_stack/req_pilot/analyze.py --in a.jsonl[,b.jsonl,...]   (pulled Kaggle shard files)"""
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HS = ("10", "30", "90")
paths = sys.argv[sys.argv.index("--in") + 1].split(",") if "--in" in sys.argv else [os.path.join(HERE, "labels.jsonl")]
recs = {}
for p in paths:
    for l in open(p):
        try:
            q = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "rows" in q:
            recs[(q["seed"], q["k"])] = q              # dedupe resumed / re-pulled lines
recs = [recs[x] for x in sorted(recs)]
print("numpy versions:", Counter(q.get("numpy") for q in recs))
rows = [dict(r, seed=q["seed"], k=q["k"]) for q in recs for r in q["rows"]]
print(f"decision seconds {len(recs)}, episodes {len({q['seed'] for q in recs})}, labelled requests {len(rows)}")
print("xApp x knob type x dir counts:", Counter((r["xapp"], r["knob"][0], int(np.sign(r['prop'] - r['cur']))) for r in rows))

XA, KT = ("MRO", "TS", "ES", "SLICE"), ("cio", "hys", "ttt", "ll_ratio", "sleep", "carrier")
NUM = ["dir", "mag_q", "cur", "is_rev_cio", "u_c", "u_n", "u_diff", "u_nbr", "ll_slice_c", "rsv_idle_c",
       "ll_ratio_delay", "ll_nan", "act_ue_c", "carriers_c", "ll_ratio_c", "embb_p5_c", "is_macro_c", "asleep_c",
       "since_change", "dwell_ok", "n_same_cell", "n_same_knob", "n_pending", "dir_x_ll", "dir_x_u"]
CELLS = sorted({(r["xapp"], r["knob"][0], int(np.sign(r["prop"] - r["cur"]))) for r in rows})


def X_full(r):
    f = r["feat"]
    return [float(f["xapp"] == x) for x in XA] + [float(f["ktype"] == k) for k in KT] + [float(f[c]) for c in NUM]


def X_base(r):
    key = (r["xapp"], r["knob"][0], int(np.sign(r["prop"] - r["cur"])))
    return [float(key == c) for c in CELLS]


Xf = np.array([X_full(r) for r in rows])
Xb = np.array([X_base(r) for r in rows])
G = np.array([r["seed"] for r in rows])
Y = {h: np.array([r["dJ"][h] for r in rows]) for h in HS}
rng = np.random.default_rng(0)


def cv_pred(make, X, y, g, proba=True):
    pr = np.full(len(y), np.nan)
    nf = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(nf).split(X, y, g):
        if proba and len(np.unique(y[tr])) < 2:
            pr[te] = y[tr].mean()
            continue
        m = make().fit(X[tr], y[tr])
        pr[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return pr


def auc_ci(y, p, g):
    ug = np.unique(g)
    bs = []
    for _ in range(1000):
        s = rng.choice(ug, len(ug))
        ix = np.concatenate([np.where(g == e)[0] for e in s])
        if len(np.unique(y[ix])) == 2:
            bs.append(roc_auc_score(y[ix], p[ix]))
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
hgbc = lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=10)
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
hgbr = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=10)

# reliability
rep = [r for r in rows if "dJ_nudged" in r]
rel = {}
print(f"\n== reliability (1e-6 m nudge), n pairs {len(rep)}")
for h in HS:
    a = np.array([r["dJ"][h] for r in rep]); b = np.array([r["dJ_nudged"][h] for r in rep])
    pool = np.r_[a, b]
    rel[h] = 1 - (np.var(a - b) / 2) / np.var(pool) if np.var(pool) > 0 else np.nan
    nz = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)
    print(f"H={h:>2}: reliability {rel[h]:.3f}  pearson {np.corrcoef(a, b)[0, 1]:.3f}  sign agree (both nz, n={nz.sum()}) "
          f"{np.mean(np.sign(a[nz]) == np.sign(b[nz])) if nz.any() else np.nan:.2f}  zero-status agree "
          f"{np.mean((np.abs(a) > 1e-6) == (np.abs(b) > 1e-6)):.2f}  noise sd {np.std(a - b) / np.sqrt(2):.2f} vs sd {np.std(pool):.2f}")

verdict = {}
for h in HS:
    y = Y[h]
    nz = np.abs(y) > 1e-6
    print(f"\n== H={h}: n {len(y)}, frac exactly 0 {1 - nz.mean():.2f}, +(reject hurts) {np.mean(y > 1e-6):.2f}, "
          f"-(reject helps) {np.mean(y < -1e-6):.2f}; |dJ| nz quantiles 10/50/90/max "
          f"{np.round(np.percentile(np.abs(y[nz]), [10, 50, 90, 100]), 2).tolist() if nz.any() else []}")
    for c in CELLS:
        m = np.array([(r["xapp"], r["knob"][0], int(np.sign(r["prop"] - r["cur"]))) == c for r in rows])
        yy = y[m]
        print(f"   {str(c):28s} n {m.sum():3d} zero {np.mean(np.abs(yy) <= 1e-6):.2f} pos {np.mean(yy > 1e-6):.2f} "
              f"neg {np.mean(yy < -1e-6):.2f} mean {yy.mean():7.2f} median|nz| "
              f"{np.median(np.abs(yy[np.abs(yy) > 1e-6])) if (np.abs(yy) > 1e-6).any() else 0:6.2f}")
    res = {}
    if nz.sum() >= 10 and len(np.unique(y[nz] > 0)) == 2:
        s, g = (y[nz] > 0).astype(int), G[nz]
        for nm, mk, X in (("logistic", logit, Xf[nz]), ("hgb", hgbc, Xf[nz]), ("baseline", logit, Xb[nz])):
            p = cv_pred(mk, X, s, g)
            a, lo, hi = auc_ci(s, p, g)
            res[nm] = a
            print(f"   sign AUC (nz, n={nz.sum()}) {nm:9s} {a:.3f} [{lo:.3f},{hi:.3f}]  acc {np.mean((p > 0.5) == s):.2f} "
                  f"(majority {max(s.mean(), 1 - s.mean()):.2f})")
    if len(np.unique(nz)) == 2:
        z = nz.astype(int)
        for nm, mk, X in (("logistic", logit, Xf), ("hgb", hgbc, Xf), ("baseline", logit, Xb)):
            p = cv_pred(mk, X, z, G)
            print(f"   nonzero-vs-zero AUC {nm:9s} {roc_auc_score(z, p):.3f}")
    for nm, mk, X in (("ridge", ridge, Xf), ("hgb", hgbr, Xf), ("baseline", ridge, Xb)):
        p = cv_pred(mk, X, y, G, proba=False)
        pn = cv_pred(mk, X[nz], y[nz], G[nz], proba=False) if nz.sum() >= 10 else None
        print(f"   magnitude CV R2 {nm:9s} all {1 - np.var(y - p) / np.var(y):.3f}"
              + (f"  nz-only {1 - np.var(y[nz] - pn) / np.var(y[nz]):.3f}" if pn is not None else ""))
    best = max(res.get("logistic", 0), res.get("hgb", 0))
    verdict[h] = (best, rel[h], int(nz.sum()), best >= 0.70 and rel[h] >= 0.70 and nz.sum() >= 30)
print("\n== decision rule per H (bestAUC, reliability, n_nonzero, pass):", verdict)
print("VERDICT:", "PROMISING" if any(v[3] for v in verdict.values()) else "DEAD for E6")
