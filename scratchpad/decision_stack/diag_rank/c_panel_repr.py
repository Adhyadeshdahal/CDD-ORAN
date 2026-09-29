"""Diag C: representation ceilings + xApp additivity on the oracle panels."""
import json
import numpy as np
from scipy.stats import spearmanr
from cdd_oran.decision import collect as CO, effect_model as EM, plans as P

S = [json.loads(l) for l in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl")]
S = [r for r in S if r.get("type") == "slot"]
def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.nan if np.ptp(a) == 0 or np.ptp(b) == 0 else spearmanr(a, b).statistic
# 1. xApp additivity for network-uniform plans: dJ(code) ~ sum_x dJ(single:x:mode_x) (+ rb ignored -> only rb=0 codes)
single = {}
res_add, tot, pairs = [], [], []
for r in S:
    J = np.array(r["oracle"]["J"]); dJ = J - J[0]
    sidx = {c["src"][0]: i for i, c in enumerate(r["cands"])}
    for i, c in enumerate(r["cands"]):
        cs = set(c["codes"])
        if len(cs) != 1 or i == 0: continue
        code = cs.pop(); rp = CO.decode(code)
        if rp["rb"] or sum(m != "accept" for m in rp["mode"].values()) < 2: continue
        try:
            pred = sum(dJ[sidx[f"single:{x}:{m}"]] for x, m in rp["mode"].items() if m != "accept")
        except KeyError:
            continue
        pairs.append((pred, dJ[i]))
pairs = np.array(pairs)
print("xApp additivity, network-uniform multi-xApp plans: n %d, corr %.3f, R2 %.3f, sd(actual) %.1f sd(resid) %.1f" % (
    len(pairs), np.corrcoef(pairs.T)[0, 1], 1 - np.var(pairs[:, 1] - pairs[:, 0]) / np.var(pairs[:, 1]),
    pairs[:, 1].std(), (pairs[:, 1] - pairs[:, 0]).std()))
# 2. representation ceiling: network-summed policy features, fit ON ORACLE LABELS, CV by episode.
X, y, g, ep = [], [], [], []
for si, r in enumerate(S):
    J = np.array(r["oracle"]["J"]); dJ = J - J[0]
    for c, d in zip(r["cands"], dJ):
        X.append(EM.policy_features(np.array(c["codes"])).sum(0)); y.append(d); g.append(si); ep.append(r["seed"])
X, y, g, ep = map(np.asarray, (X, y, g, ep))
strat = np.array([S[i]["scenario"] + S[i]["load"] for i in g])
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
def within_rho(pred):
    return np.nanmean([sp(pred[g == s], y[g == s]) for s in np.unique(g)])
useps = np.unique(ep); fold = {e: i % 5 for i, e in enumerate(useps)}; f = np.array([fold[e] for e in ep])
for nm, Xf in (("context-free main effects (13, network-summed)", X),
               ("+ stratum interactions", np.hstack([X] + [X * (strat == s)[:, None] for s in np.unique(strat)]))):
    pr = np.zeros_like(y)
    for k in range(5):
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xf[f != k], y[f != k]); pr[f == k] = m.predict(Xf[f == k])
    print("oracle-trained ceiling [%s]: CV within-slot rho %.3f" % (nm, within_rho(pr)))
# 3. ceiling with in-slot knowledge of the singles (what a perfect per-xApp CATE would give, additive over xApps)
rh = []
for s in np.unique(g):
    r = S[s]; J = np.array(r["oracle"]["J"]); dJ = J - J[0]
    sidx = {c["src"][0]: i for i, c in enumerate(r["cands"])}
    pred = []
    for c in r["cands"]:
        # region-additive, xApp-additive: each region contributes 1/R of the network single effect
        v = 0.0
        for code in c["codes"]:
            rp = CO.decode(code)
            v += sum(dJ[sidx[f"single:{x}:{m}"]] for x, m in rp["mode"].items() if m != "accept") / len(c["codes"])
            v += rp["rb"] * dJ[sidx["rollback"]] / len(c["codes"])
        pred.append(v)
    rh.append(sp(pred, dJ))
print("ceiling: exact per-slot network singles, additive over xApps AND regions (1/R share): within-slot rho %.3f (excl. fixed cands trivially included)" % np.nanmean(rh))
# same but only on non-fixed candidates (random + search) to avoid tautology
rh2, rh3 = [], []
for s in np.unique(g):
    r = S[s]; J = np.array(r["oracle"]["J"]); dJ = J - J[0]
    sidx = {c["src"][0]: i for i, c in enumerate(r["cands"])}
    pr_, yy, pm = [], [], []
    for c, d, i in zip(r["cands"], dJ, range(len(dJ))):
        if not (c["src"][0].startswith(("random", "search"))): continue
        v = 0.0
        for code in c["codes"]:
            rp = CO.decode(code)
            v += sum(dJ[sidx[f"single:{x}:{m}"]] for x, m in rp["mode"].items() if m != "accept") / len(c["codes"])
            v += rp["rb"] * dJ[sidx["rollback"]] / len(c["codes"])
        pr_.append(v); yy.append(d); pm.append(r["models"]["all"]["pred"][i])
    if len(yy) >= 4:
        rh2.append(sp(pr_, yy)); rh3.append(sp(pm, yy))
print("  on random/search candidates only: additive-oracle rho %.3f vs learned 'all' rho %.3f (slots %d)" % (np.nanmean(rh2), np.nanmean(rh3), len(rh2)))
# 4. region-heterogeneous random plans: are random:region plans predicted by network-uniform effects?
