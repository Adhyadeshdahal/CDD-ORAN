"""Diag D: is the CATE of fixed candidates predictable from observable context when labels are PAIRED oracle
contrasts (noise-free)? Context = the accept-all ref trace of the same seed (identical to the panel main run)."""
import json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from cdd_oran.decision import collect as CO, effect_model as EM
from cdd_oran.decision.trace import Trace

S = [json.loads(l) for l in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl")]
S = [r for r in S if r.get("type") == "slot"]
ctx = {}
for seed in sorted({r["seed"] for r in S}):
    st = CO.dev_stratum(seed)
    tr = Trace.from_npz(f"scratchpad/e6_dev/runs/e6-policy-v3/npz_all/ref_{st['scenario']}_{st['load']}_s{seed}.npz")
    e = EM.episode_from_trace(tr, 90)
    for i, t in enumerate(e["t"]):
        own, g = e["own"][i], e["glob"][i]
        ctx[(seed, int(t))] = (own, g, (e["y"][i] @ EM.PolicyEffectModel.cost_weights(1e4, 5.0)))
names = [c["src"][0] for c in S[0]["cands"][1:15]]
X, Yd, ep, keep = [], [], [], []
for r in S:
    own, g, yreg = ctx[(r["seed"], r["t"])]
    assert abs(yreg.sum() - r["oracle"]["J"][0]) < 1e-6 * max(1, abs(r["oracle"]["J"][0])), (yreg.sum(), r["oracle"]["J"][0])
    own = np.nan_to_num(own)
    f = np.concatenate([own.mean(0), own.max(0), own.min(0), g, [r["load"] == "high", r["scenario"] == "surge",
                        r["scenario"] == "mistune", r["k"]], own.ravel()])
    X.append(f)
    J = np.array(r["oracle"]["J"]); Yd.append(J[1:15] - J[0]); ep.append(r["seed"])
X, Yd, ep = np.array(X, float), np.array(Yd), np.array(ep)
print("region labels sum == oracle accept-all J: OK (all slots)")
u = np.unique(ep); fold = np.array([np.where(u == e)[0][0] % 5 for e in ep])
print("target               sd     R2_ridge  R2_hgb  corr(learned 'all' pred, oracle) across slots")
P = np.zeros_like(Yd)
for j, nm in enumerate(names):
    y = Yd[:, j]
    out = {}
    for mdl in ("ridge", "hgb"):
        pr = np.zeros_like(y)
        for k in range(5):
            m = (RidgeCV(alphas=np.logspace(-2, 4, 13)) if mdl == "ridge" else
                 HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=15, learning_rate=0.05)).fit(X[fold != k], y[fold != k])
            pr[fold == k] = m.predict(X[fold == k])
        out[mdl] = (1 - np.var(y - pr) / np.var(y), pr)
    P[:, j] = out["ridge"][1] if out["ridge"][0] > out["hgb"][0] else out["hgb"][1]
    lp = np.array([r["models"]["all"]["pred"][j + 1] for r in S])
    c = np.corrcoef(lp, y)[0, 1] if np.std(y) > 0 else np.nan
    print("%-20s %6.1f  %7.3f  %7.3f   %+.3f" % (nm, y.std(), out["ridge"][0], out["hgb"][0], c))
# ranking among the 15 fixed candidates using CV predictions trained on 288 paired slots
rho_p, rho_l, imp_p, imp_l, imp_o = [], [], [], [], []
for i, r in enumerate(S):
    J = np.array(r["oracle"]["J"][:15]); dJ = J - J[0]
    pp = np.concatenate([[0], P[i]]); lp = np.array(r["models"]["all"]["pred"][:15])
    rho_p.append(spearmanr(pp, dJ).statistic if np.ptp(dJ) > 0 else np.nan)
    rho_l.append(spearmanr(lp, dJ).statistic if np.ptp(dJ) > 0 and np.ptp(lp) > 0 else np.nan)
    imp_p.append(-dJ[np.argmin(pp)]); imp_l.append(-dJ[np.argmin(lp)]); imp_o.append(-dJ.min())
print("fixed-15 panel: within-slot rho paired-CV %.3f vs learned 'all' %.3f; improvement vs accept-all: paired-CV pick %+.1f, learned pick %+.1f, oracle best %+.1f" % (
    np.nanmean(rho_p), np.nanmean(rho_l), np.mean(imp_p), np.mean(imp_l), np.mean(imp_o)))
# conservative pick: deviate only if predicted gain > 20
imp_c = []
for i, r in enumerate(S):
    J = np.array(r["oracle"]["J"][:15]); dJ = J - J[0]; pp = np.concatenate([[0], P[i]])
    b = np.argmin(pp); imp_c.append(-dJ[b] if pp[b] < -20 else 0.0)
ic = np.array(imp_c); print("  paired-CV pick with margin 20: mean %+.1f, deviate %d/360, per-episode cluster SE %.1f" % (
    ic.mean(), (ic != 0).sum(), np.std([ic[ep == e].mean() for e in u]) / np.sqrt(len(u))))
