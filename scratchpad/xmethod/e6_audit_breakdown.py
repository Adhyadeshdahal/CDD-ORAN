"""Experiment D breakdown: the v4 GT label NULL is an EQUIVALENCE label (gt_p.classify: GT CI inside +-delta, delta =
5 % of the KPI's max |mean|), not a sharp null, so a GT-NULL hypothesis may carry a real effect smaller than delta.
Rates are reported on nested sets:
  A  every GT-NULL hypothesis (the R-15 definition)
  B  GT-NULL with the GT 95% CI containing 0 (no detectable GT effect: the closest available proxy for a true null)
  C  A minus the hypotheses whose GT CI excludes 0 by a wide margin AND that PMRT rejects in most slices
plus, per hypothesis, the slice rejection counts, PMRT's declared effect sign vs the GT sign, and the placebo logs (an
exact sharp null by construction: accept is applied whatever the logged mode).

  uv run python scratchpad/xmethod/e6_audit_breakdown.py --audit results/e6_audit/e6_audit.json
         --ref-analysis <v4 analysis_v4.json> --out results/e6_audit/e6_audit_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

ARMS = ("loadsp_c", "loadsp")


def wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    ph, d = x / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(ps, slices, hyps, arm, pk, thr, reps=4000):
    hits, cnt = [], []
    for s in slices:
        v = [ps[s][h][arm][pk] for h in hyps if ps[s][h]["status"] == "tested"]
        v = [x for x in v if x is not None and np.isfinite(x)]
        hits.append(sum(x <= thr for x in v))
        cnt.append(len(v))
    hits, cnt = np.array(hits, float), np.array(cnt, float)
    out = {"rate": float(hits.sum() / max(cnt.sum(), 1)), "x": int(hits.sum()), "m": int(cnt.sum()),
           "ci_wilson_hyp": wilson(int(hits.sum()), int(cnt.sum()))}
    if len(slices) > 1:
        rng = np.random.default_rng([7801, 16, int(thr * 1000), len(hyps)])
        bs = [hits[i].sum() / max(cnt[i].sum(), 1) for i in (rng.integers(0, len(hits), len(hits)) for _ in range(reps))]
        out["ci_cluster"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--ref-analysis", dest="ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.audit))
    ref = json.load(open(a.ref))
    cells = {"|".join((c["family"], c["relation"], c["kpi"])): c for c in ref["gt"]["cells"]}
    ps = d["per_slice"]
    kinds = {k: sorted(n for n in ps if n.startswith(p)) for k, p in (("60", "s60_"), ("120", "s120_"))}
    kinds["pooled"] = ["pooled"]
    A = sorted(d["null_hyps"])
    B = [h for h in A if cells[h]["ci"][0] <= 0 <= cells[h]["ci"][1]]
    per = {}
    for h in A:
        c = cells[h]
        row = {"gt_mean": c["mean"], "gt_ci": c["ci"], "delta": c["delta"], "gt_ci_contains_0": h in B}
        for k, sl in kinds.items():
            for arm in ARMS:
                tested = [s for s in sl if ps[s][h]["status"] == "tested"]
                row[f"{k}_{arm}_rej05"] = sum(ps[s][h][arm]["p2"] <= .05 for s in tested)
                row[f"{k}_{arm}_n"] = len(tested)
                row[f"{k}_{arm}_signs"] = sorted({int(ps[s][h][arm]["sign"]) for s in tested if ps[s][h][arm]["p2"] <= .05})
        row["gt_sign"] = int(np.sign(c["mean"]))
        per[h] = row
    C = [h for h in A if not (not per[h]["gt_ci_contains_0"]
                              and per[h]["60_loadsp_c_rej05"] >= per[h]["60_loadsp_c_n"] // 2)]
    out = {"note": __doc__.split("\n\n")[0], "sets": {"A": A, "B": B, "C": C}, "per_hyp": per, "rates": {}}
    for nm, hy in (("A", A), ("B", B), ("C", C)):
        out["rates"][nm] = {k: {arm: {pk: {str(t): rate(ps, sl, hy, arm, pk, t) for t in (.05, .01)}
                                      for pk in ("p2", "p_used")} for arm in ARMS} for k, sl in kinds.items()}
    pl = ps["placebo"]
    hp = [h for h in pl if pl[h]["status"] == "tested"]
    out["placebo"] = {arm: {pk: {str(t): rate(ps, ["placebo"], hp, arm, pk, t) for t in (.05, .01)}
                            for pk in ("p2", "p_used")} for arm in ARMS}
    with open(a.out, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    for nm in ("A", "B", "C"):
        for k in ("60", "120", "pooled"):
            for arm in ARMS:
                r5, r1 = out["rates"][nm][k][arm]["p2"]["0.05"], out["rates"][nm][k][arm]["p2"]["0.01"]
                print(f"{nm} ({len(out['sets'][nm])} hyp) {k:>6} {arm:9s} .05 {r5['rate']:.3f} ({r5['x']}/{r5['m']}) "
                      f"CI {r5.get('ci_cluster', r5['ci_wilson_hyp'])} | .01 {r1['rate']:.3f} ({r1['x']}/{r1['m']}) "
                      f"CI {r1.get('ci_cluster', r1['ci_wilson_hyp'])}")
    for arm in ARMS:
        for pk in ("p2", "p_used"):
            r5, r1 = out["placebo"][arm][pk]["0.05"], out["placebo"][arm][pk]["0.01"]
            print(f"placebo {arm:9s} {pk:6s} .05 {r5['rate']:.3f} ({r5['x']}/{r5['m']}) Wilson {r5['ci_wilson_hyp']} | "
                  f".01 {r1['rate']:.3f} ({r1['x']}/{r1['m']})")
    for h in sorted(set(A) - set(C)):
        r = per[h]
        print(f"excluded from C: {h} GT {r['gt_mean']:.2f} {r['gt_ci']} delta {r['delta']:.2f} sign {r['gt_sign']}; "
              f"s60 rej {r['60_loadsp_c_rej05']}/{r['60_loadsp_c_n']} PMRT signs {r['60_loadsp_c_signs']}")


if __name__ == "__main__":
    main()
