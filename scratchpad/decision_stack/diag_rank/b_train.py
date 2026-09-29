"""Diag B: v3 training labels -- noise vs treatment effect; model coefficient scale."""
import glob, os, pickle, time
import numpy as np
from cdd_oran.decision import collect as CO, effect_model as EM, rank_eval as RE
from cdd_oran.decision.trace import Trace

t0 = time.time()
paths = sorted(p for p in glob.glob("scratchpad/e6_dev/runs/e6-policy-v3/npz_all/v3_*.npz")
               if CO.dev_stratum(int(os.path.basename(p).rsplit("_s", 1)[1][:-4]))["split"] == "fit")
eps, meta = [], []
for p in paths:
    tr = Trace.from_npz(p)
    e = EM.episode_from_trace(tr, 90)
    e["stratum"] = f"{tr.meta['cfg']['scenario']}-{tr.meta['cfg']['load']}"
    e["elig"] = tr.arrays["pol_x_elig"][: len(e["code"])]
    e["nbr_dev"] = tr.arrays["pol_nbr_n_dev"][: len(e["code"])]
    eps.append(e)
print("episodes", len(eps), "load s %.0f" % (time.time() - t0))
d = EM.PolicyEffectModel.stack(eps)
M, R = d["code"].shape
cw = EM.PolicyEffectModel.cost_weights(1e4, 5.0)
Y = d["y"] @ cw                                   # (M, R) priced region outcome
print("rows M*R", M * R, "slots", M, "regions", R)
print("priced region y: mean %.1f sd %.1f" % (Y.mean(), Y.std()))
elig = np.concatenate([e["elig"] for e in eps])   # (M, R, 4)
strat = np.concatenate([[e["stratum"]] * len(e["code"]) for e in eps])
np.save("scratchpad/decision_stack/diag_rank/_Y.npy", Y)
# context-only fit: residual sd (5-fold by episode)
own = np.nan_to_num(d["own"]); glob_ = np.repeat(np.nan_to_num(d["glob"]), R, 0)
ridx = np.tile(np.arange(R), M)
Xc = np.hstack([own.reshape(M * R, -1), glob_, np.eye(R)[ridx]])
lagY = None
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
y = Y.ravel(); ep = np.repeat(d["ep"], R)
A = EM.policy_features(d["code"]).reshape(M * R, -1)
dev = (d["code"].ravel() != 0)
def cv(X, y, model="ridge", k=5):
    f = ep % k; pr = np.zeros_like(y)
    for i in range(k):
        tr_, te = f != i, f == i
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr_], y[tr_]) if model == "ridge" else \
            HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=40).fit(X[tr_], y[tr_])
        pr[te] = m.predict(X[te])
    return pr
for nm, X in (("context", Xc), ("context+policy", np.hstack([Xc, A]))):
    for mdl in ("ridge", "hgb"):
        pr = cv(X, y, mdl)
        print("CV %-15s %-5s R2 %.3f resid sd %.1f" % (nm, mdl, 1 - np.var(y - pr) / np.var(y), np.std(y - pr)))
# control rows only: residual noise at fixed policy (accept-all)
ctrl = ~dev
pr = cv(Xc, y, "hgb")
res = y - pr
print("resid sd, control rows (code 0): %.1f ; treated rows %.1f" % (res[ctrl].std(), res[dev].std()))
# policy main effects from OLS on residuals (context-adjusted), with SEs
from numpy.linalg import lstsq
Z = np.hstack([np.ones((len(y), 1)), A])
b, *_ = lstsq(Z, res, rcond=None)
e_ = res - Z @ b
cov = np.linalg.pinv(Z.T @ Z) * e_.var()
se = np.sqrt(np.diag(cov))
print("context-adjusted policy main effects (per region, priced):")
for n, bb, s in zip(EM.POLICY_NAMES, b[1:], se[1:]):
    print("  %-12s %+7.2f (se %.2f, z %+.1f, n %d)" % (n, bb, s, bb / s, int(A[:, EM.POLICY_NAMES.index(n)].sum())))
# per region SD of priced y by stratum
print("ep-level between-episode sd of mean region y: %.1f" % np.std([Y[d['ep'] == i].mean() for i in np.unique(d['ep'])]))
