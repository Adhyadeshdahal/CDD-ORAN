# Morning brief 4 (night of 2026-09-30 to 10-01)

## Status at a glance
| item | state |
|---|---|
| v4 ground truth (Kaggle e6p-disc-v4gt-1 a/b) | a COMPLETE; b RUNNING (checked 23:47) |
| v4 verdict analysis (frozen code) | waiting on GT-b |
| PMRT rename (branch worktree-agent-a25d794648fbfb148) | done, not merged; local equivalence byte-identical; full-data rerun pending |
| option (a) study | freeze prep done on the rename branch; freeze waits for the v4 verdict |
| docs/ARCHITECTURE.md | update agent running (rename branch) |

## Log
- 23:00-23:45: PMRT rename finished (artifact E6P_PMRT_V4.json sha 9991c390...; 16929-leaf small-cache check and
  16245-leaf analyzer dry run both 0 differences). Option (a) protocol: PMRT artifact sha, dev kernel dev-2, K1 fixed.
- ~00:05: ARCHITECTURE.md rewritten on the rename branch (4d96f87). Stale: cdd_oran/envs/README.md, decision/__init__ docstring.
- ~01:40 **v4 VERDICT: PARTIAL** (frozen code). Everything passes except P2 vs granger_by at n 60 (2/10 wins; 4/5 at
  n 120). granger_by is uncalibrated: 13 placebo declarations (PMRT 0), precision .59-.71 vs PMRT .82-.95. PMRT pooled
  indirect F1 .95, chain 4/4 and premise on every 120-slice, sign accuracy 1.00. SHAP/QACM/two-tower F1 0 pooled.
