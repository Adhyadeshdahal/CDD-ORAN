"""Diag A: oracle panel signal structure + model prediction scale."""
import json
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr

rows = [json.loads(l) for l in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl")]
S = [r for r in rows if r.get("type") == "slot"]
print("slots", len(S))
# consistency: realised accept-all vs oracle J[0]
d = [r["oracle"]["J"][0] - r["realized_accept_all"]["J"] for r in S]
print("J0 - realised accept-all: max|.| =", np.max(np.abs(d)))
J0 = np.array([r["oracle"]["J"][0] for r in S])
head = np.array([r["oracle"]["J"][0] - min(r["oracle"]["J"]) for r in S])
sd = np.array([np.std(r["oracle"]["J"]) for r in S])
rng_ = np.array([np.ptp(r["oracle"]["J"]) for r in S])
print("J0 mean %.0f sd-across-slots %.0f" % (J0.mean(), J0.std()))
print("headroom mean %.1f median %.1f p10 %.1f p90 %.1f; frac slots headroom>10: %.2f" % (
    head.mean(), np.median(head), *np.percentile(head, [10, 90]), (head > 10).mean()))
print("within-slot sd of J mean %.1f median %.1f ; range mean %.1f" % (sd.mean(), np.median(sd), rng_.mean()))
print("within-slot sd / J0: %.4f" % np.mean(sd / J0))
# which source is best
best_src = Counter()
src_gain = defaultdict(list)
for r in S:
    J = np.array(r["oracle"]["J"])
    b = int(np.argmin(J))
    best_src[r["cands"][b]["src"][0].split(":")[0] + (":" + r["cands"][b]["src"][0].split(":")[1] if r["cands"][b]["src"][0].startswith("single") else "")] += 1
    for c, j in zip(r["cands"], J):
        for s in set(c["src"]):
            key = s if s.startswith(("single", "freeze", "rollback", "accept")) else s.split(":")[0] + ":" + s.split(":")[1]
            src_gain[key].append(J[0] - j)
print("best source counts:", best_src.most_common())
print("mean improvement over accept-all by candidate source (>0 good):")
for k, v in sorted(src_gain.items(), key=lambda kv: -np.mean(kv[1])):
    v = np.array(v)
    print("  %-22s n=%4d mean %+7.1f sd %6.1f  P(>0) %.2f" % (k, len(v), v.mean(), v.std(), (v > 0).mean()))
# concentration: gap between best and 2nd best; share of candidates within 5 of best
gap, near = [], []
for r in S:
    J = np.sort(np.array(r["oracle"]["J"]))
    gap.append(J[1] - J[0])
    near.append(np.mean(J - J[0] < 5))
print("best-2nd gap mean %.1f median %.1f; share of cands within 5 of best %.2f" % (np.mean(gap), np.median(gap), np.mean(near)))
# oracle best fixed single-candidate policy across ALL slots (a context-free rule)
fixed = defaultdict(list)
for r in S:
    J = np.array(r["oracle"]["J"])
    for c, j in zip(r["cands"], J):
        s0 = c["src"][0]
        if s0.startswith(("single", "freeze", "rollback")):
            fixed[s0].append(J[0] - j)
print("context-free fixed rules, mean gain: ", sorted(((k, round(np.mean(v), 1)) for k, v in fixed.items()), key=lambda x: -x[1])[:5])
# component decomposition of the headroom: best - accept
comp = defaultdict(list)
for r in S:
    J = np.array(r["oracle"]["J"]); b = int(np.argmin(J)); o = r["oracle"]
    for c in ("viol_LL", "viol_eMBB", "viol_BE", "energy_kwh", "rlf", "churn"):
        comp[c].append(o[c][0] - o[c][b])
print("headroom by component (accept - best; J-priced LL x6, energy x1e4):")
for c, v in comp.items():
    m = np.mean(v); pr = m * (6 if c == "viol_LL" else 1e4 if c == "energy_kwh" else 1)
    print("  %-10s raw %+8.3f  priced %+7.1f" % (c, m, pr) if c in ("viol_LL","viol_eMBB","viol_BE","energy_kwh") else "  %-10s raw %+8.3f" % (c, m))
# within-slot sd of each priced component
for c, w in (("viol_LL", 6), ("viol_eMBB", 1), ("viol_BE", 1), ("energy_kwh", 1e4)):
    print("  within-slot sd priced %-10s %.1f" % (c, np.mean([np.std(np.array(r["oracle"][c]) * w) for r in S])))
# model prediction scale vs oracle
for nm in ("none", "topology", "all"):
    ps, rh, rhs, rh_fix, rh_rand = [], [], [], [], []
    for r in S:
        p = np.array(r["models"][nm]["pred"]); J = np.array(r["oracle"]["J"]) - r["oracle"]["J"][0]
        ok = np.isfinite(p); ps.append(np.std(p[ok]))
        unid = np.array(r["models"][nm]["unid"])
        m = ok & ~unid
        if m.sum() >= 4 and np.ptp(p[m]) > 0: rhs.append(spearmanr(p[m], J[m]).statistic)
        rh.append(spearmanr(p[ok], J[ok]).statistic)
    print(nm, "pred within-slot sd mean %.1f median %.1f | oracle sd %.1f | rho all %.3f supported %.3f (n %d)" % (
        np.mean(ps), np.median(ps), sd.mean(), np.nanmean(rh), np.nanmean(rhs), len(rhs)))
