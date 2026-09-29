"""Diag F: is the oracle single-xApp effect STRUCTURED (determined by the upcoming request stream) or noise-like?
Features from the accept-all ref trace of the same seed: per xApp, # requests and net signed requested delta in the
treatment window [t, t+D) (FUTURE info, not observable at decision time) -- an upper-bound probe."""
import json
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from cdd_oran.decision import collect as CO, plans as P
from cdd_oran.decision.trace import Trace

S = [json.loads(l) for l in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl")]
S = [r for r in S if r.get("type") == "slot"]
names = [c["src"][0] for c in S[0]["cands"][:15]]
ix = {n: i for i, n in enumerate(names)}
dJ = np.array([np.array(r["oracle"]["J"][:15]) - r["oracle"]["J"][0] for r in S])
def c(a, b): return np.corrcoef(dJ[:, ix[a]], dJ[:, ix[b]])[0, 1]
print("cross-candidate corr (same slot): SLICE rej-lock %.2f rej-half %.2f | TS rej-lock %.2f rej-half %.2f | ES rej-lock %.2f | freeze vs sum singles-reject %.2f" % (
    c("single:SLICE:reject", "single:SLICE:lock"), c("single:SLICE:reject", "single:SLICE:half"),
    c("single:TS:reject", "single:TS:lock"), c("single:TS:reject", "single:TS:half"), c("single:ES:reject", "single:ES:lock"),
    np.corrcoef(dJ[:, ix["freeze"]], sum(dJ[:, ix[f"single:{x}:reject"]] for x in P.XAPPS))[0, 1]))
# within-episode persistence (slot k vs k+1)
for n in ("single:SLICE:reject", "single:TS:reject", "freeze"):
    a, b = [], []
    for i in range(len(S) - 1):
        if S[i]["seed"] == S[i + 1]["seed"] and S[i + 1]["k"] == S[i]["k"] + 1:
            a.append(dJ[i, ix[n]]); b.append(dJ[i + 1, ix[n]])
    print("lag-1 slot autocorr %-20s %.2f (n %d)" % (n, np.corrcoef(a, b)[0, 1], len(a)))
# episode-level (between-episode) share of variance
for n in ("single:SLICE:reject", "single:TS:reject", "freeze"):
    ep = np.array([r["seed"] for r in S]); v = dJ[:, ix[n]]
    means = np.array([v[ep == e].mean() for e in np.unique(ep)])
    print("between-episode var share %-20s %.2f" % (n, means.var() / v.var()))
# future request-stream features
F = []
for r in S:
    st = CO.dev_stratum(r["seed"])
    F.append(None)
cache = {}
X = []
for r in S:
    st = CO.dev_stratum(r["seed"])
    if r["seed"] not in cache:
        tr = Trace.from_npz(f"scratchpad/e6_dev/runs/e6-policy-v3/npz_all/ref_{st['scenario']}_{st['load']}_s{r['seed']}.npz")
        a = tr.arrays; xn = tr.meta["xapps"]
        cache[r["seed"]] = (a["rq_t"], np.array([P.XAPPS.index(xn[i]) for i in a["rq_xapp"]]), a["rq_prop"] - a["rq_cur"],
                            np.array([k[0] for k in tr.meta["knobs"]])[a["rq_knob"]])
    t_, x_, d_, typ = cache[r["seed"]]
    m = (t_ >= r["t"]) & (t_ < r["t"] + 20)
    f = []
    for j in range(4):
        mm = m & (x_ == j)
        f += [mm.sum(), d_[mm].sum(), np.abs(d_[mm]).sum(), (d_[mm] > 0).sum()]
    X.append(f + [r["load"] == "high", r["scenario"] == "surge", r["scenario"] == "mistune"])
X = np.array(X, float)
print("mean # requests in [t,t+20) under accept-all: MRO %.1f TS %.1f ES %.1f SLICE %.1f" % tuple(X[:, [0, 4, 8, 12]].mean(0)))
ep = np.array([r["seed"] for r in S]); u = np.unique(ep); fold = np.array([np.where(u == e)[0][0] % 5 for e in ep])
for n in ("single:SLICE:reject", "single:SLICE:lock", "single:TS:reject", "single:ES:reject", "freeze", "rollback"):
    y = dJ[:, ix[n]]
    for mdl in ("ridge", "hgb"):
        pr = np.zeros_like(y)
        for k in range(5):
            m = (RidgeCV(alphas=np.logspace(-2, 4, 13)) if mdl == "ridge" else
                 HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=15, learning_rate=0.05)).fit(X[fold != k], y[fold != k])
            pr[fold == k] = m.predict(X[fold == k])
        print("  future-request features -> %-20s %-5s CV R2 %.3f" % (n, mdl, 1 - np.var(y - pr) / np.var(y)))
# direct: SLICE:reject effect vs net signed SLICE requested delta
y = dJ[:, ix["single:SLICE:reject"]]
print("corr(SLICE:reject dJ, net SLICE delta) %.2f, (#SLICE req) %.2f, (#up) %.2f" % (
    np.corrcoef(y, X[:, 13])[0, 1], np.corrcoef(y, X[:, 12])[0, 1], np.corrcoef(y, X[:, 15])[0, 1]))
