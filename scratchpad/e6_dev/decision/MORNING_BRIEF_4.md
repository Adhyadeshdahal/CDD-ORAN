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
- ~02:00 option (a) FROZEN (freeze 1, e2d2149) and collection launched on Kaggle: disc 2 kernels + gt 2 kernels; placebo queued for a free slot. Equivalence rerun (renamed code) running.
- ~03:10 **PMRT rename full-data check:** verdict identical. The full analysis differs in 14 of 44663 values, all in
  two descriptive neural baselines (two-tower tau; QACM on one 300-slice), whose code did not change: the two Kaggle
  runs landed on different machines and those networks are not bit-reproducible across hardware. 0 differences in the
  36667 PMRT / criterion values. **Your call:** my addendum's strict rule ("any non-volatile difference -> revert")
  technically fails; I kept the rename and disclosed a clarification (the rule covers code the rename touched). If you
  prefer, revert is one `git revert` away and changes nothing about the v4 verdict.
- 03:11 option (a) DISC collection complete (600 episodes, both kernels clean). Waiting on GT + placebo.
- 07:15 option (a) GT complete (40 eps). Discovery analysis running on Kaggle (e6p-conf-discan-2; a first launch, discan-1, got empty input paths from a shell-quoting slip and is void). Next: freeze 2 (maps artifact) -> EVAL (160 seeds, 3 kernels).
