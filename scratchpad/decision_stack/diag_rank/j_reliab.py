"""Diag J: reliability of oracle effects across two runs of the same seed (Kaggle numpy 2.0.2 panel vs local numpy
2.4.2 rescoring; same commit b00920b). corr = share of effect variance reproducible under a different
micro-history/tape."""
import glob, json
import numpy as np
rows = [json.loads(l) for p in glob.glob("scratchpad/decision_stack/diag_rank/out/rep_*.jsonl") for l in open(p)]
rows += [  # seed 100020 (printed run)
    {"d_local": [-30.8, 9.0, 15.2], "d_panel": [-28.8, 18.0, 18.2]}, {"d_local": [29.8, -0.3, 13.2], "d_panel": [4.8, -17.3, -14.3]},
    {"d_local": [9.6, -1.9, 15.5], "d_panel": [16.6, -12.8, 16.6]}, {"d_local": [49.7, 157.2, 135.4], "d_panel": [52.7, 160.7, 124.0]},
    {"d_local": [-55.2, 100.9, 61.3], "d_panel": [-77.3, -39.9, -89.1]}, {"d_local": [-8.0, -47.3, -57.7], "d_panel": [97.9, -29.6, -34.1]}]
L = np.array([r["d_local"] for r in rows]); Pn = np.array([r["d_panel"] for r in rows])
print("n slots", len(L))
for j, n in enumerate(("SLICE:reject", "TS:reject", "freeze")):
    a, b = L[:, j], Pn[:, j]
    print("%-13s corr %.2f  sd %.1f  sd(diff)/sqrt2 %.1f  sign agree %.2f" % (n, np.corrcoef(a, b)[0, 1], np.std(np.r_[a, b]),
          np.std(a - b) / np.sqrt(2), np.mean(np.sign(a) == np.sign(b))))
a, b = L.ravel(), Pn.ravel()
print("pooled corr %.2f ; noise sd %.1f vs total sd %.1f -> reliability %.2f" % (np.corrcoef(a, b)[0, 1], np.std(a - b) / np.sqrt(2),
      np.std(np.r_[a, b]), 1 - (np.var(a - b) / 2) / np.var(np.r_[a, b])))
# within-slot ranking agreement of the 3 candidates + accept-all
from scipy.stats import spearmanr
rh = [spearmanr(np.r_[0, l], np.r_[0, p]).statistic for l, p in zip(L, Pn) if np.ptp(l) > 0 and np.ptp(p) > 0]
print("within-slot rank agreement (4 cands) mean rho %.2f ; argmin agree %.2f" % (np.mean(rh),
      np.mean([np.argmin(np.r_[0, l]) == np.argmin(np.r_[0, p]) for l, p in zip(L, Pn)])))
