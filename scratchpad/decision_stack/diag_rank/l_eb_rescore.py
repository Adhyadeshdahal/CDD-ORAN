"""Diag L (offline, no simulation): EB shrinkage of the e6-rank1 panels' predicted contrasts toward accept-all.
Re-picks within the SAME recorded panels (search candidates were chosen unshrunk) and reports raw / supported pick
regret + the beat-zero-contrast precondition, before vs after.

  PYTHONPATH=. python scratchpad/decision_stack/diag_rank/l_eb_rescore.py
"""
import json
import pickle

import numpy as np

from cdd_oran.decision import rank_eval as RE

rows = [json.loads(ln) for ln in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl")]
rows = [r for r in rows if r.get("type") == "slot"]
K = {k: v.manifest["n_members"] for k, v in
     pickle.load(open("scratchpad/decision_stack/rank_models.pkl", "rb"))["models"].items()}
out = {}
for name in sorted(K):
    sh = RE.shrink_rows(rows, name, K[name])
    b0, b1 = RE.summarize(rows, by_stratum=False)[name]["pooled"], RE.summarize(sh, by_stratum=False)[name]["pooled"]
    t2 = np.array([r["models"][name]["tau2"] for r in sh])
    s = np.concatenate([np.asarray(r["models"][name]["s"])[1:] for r in sh])
    f = lambda d: [round(d["mean"], 2), round(d["lo"], 2), round(d["hi"], 2)]
    o = {"tau2_zero_slots": int((t2 == 0).sum()), "n_slots": len(sh), "s_median": round(float(np.median(s)), 4),
         "s_p90": round(float(np.quantile(s, 0.9)), 4)}
    for k in ("regret_raw", "regret", "regret_accept_all", "improvement_raw", "improvement", "rho_raw"):
        o[k] = {"before": f(b0[k]), "after": f(b1[k])}
    o["deviate_raw"] = {"before": sum(r["models"][name]["metrics"]["pick_raw"] != 0 for r in rows),
                        "after": sum(r["models"][name]["metrics"]["pick_raw"] != 0 for r in sh)}
    for p in ("raw", "sup"):
        z0, z1 = RE.beats_zero_contrast(rows, name, p), RE.beats_zero_contrast(sh, name, p)
        o[f"beats_zero_{p}"] = {"before": [z0["beats"], z0["harms"]], "after": [z1["beats"], z1["harms"]]}
    out[name] = o
for n, o in out.items():
    print(n, json.dumps(o, separators=(",", ":")))
