# Morning brief 5 (night of 2026-10-01 to 10-02)

## Log
- 00:35 2b EVAL: kernels a, b done; c, d, e running. Next: analysis -> verdict -> archive report.
- 00:45 c, e done; shard logs a/b/c/e clean (16 shards, 136 lines each, last seeds 191258/191259, 0 error lines). d running.
- ~00:55 d done; all 20 shards clean (last seed 191259). Analysis launched on Kaggle: e6p-cs-evalan-1 (quoting verified, pin match).
- ~01:00 **2b certified-safe VERDICT: PARTIAL** (160 EVAL seeds, frozen code, artifact verified). CS:PMRT is eligible
  and safe: R +0.256 [0.169, 0.339], energy retention 1.15, every guard ratio below 1 (RLF 0.96, UB90 0.99 <= 1.20 cap,
  so S1 passes). E, D1 (beats the best associational map CS:corr@dev by 0.16, LB90 0.09), D2 (Holm, all 7 rejected) and
  S1 pass; D3 fails: CS:PMRT is bit-identical to never_sleep (delta 0), because the gate deferred exactly the sleep-up
  units and nothing else. This is the outcome the builder projected before EVAL. It fixes the option (a) RLF failure
  (MG:PMRT on these seeds: R +0.201, RLF 1.05). CS:GT +0.221. Associational maps: corr@dev +0.095, qacm -0.01,
  granger / granger_by / two_tower / shap -10.5 to -11.1. STOP RULE applies: no further referee iterations.
  Files: decision/cs_verdict.json, cs_eval.json, cs_evalan_job.log.
- ~01:20 raw EVAL records pulled locally (runs/e6p-cs-eval-1, runs/e6p-conf-eval-1). Archive report builder (subagent) started: reports/2026-10-02-v4-pmrt-confounded-certsafe/.
- ~01:45 Archive report done: reports/2026-10-02-v4-pmrt-confounded-certsafe/ (index.html + make_figs.py; 55 assertion
  blocks recompute everything from raw records and all pass). The independent recomputation corrected three of my claims:
  (1) granger_by precision is .59-.75 across all 18 slices (not .59-.71); (2) the "ptx -> nbr RLF z -2.2" is the
  plain_c arm; PMRT's primary statistic (loadsp_c) gives z -0.70, so the edge was not close to being declared;
  (3) the ungated MG:PMRT referee was ELIGIBLE on the 160 fresh 2b seeds (R +0.201, RLF 1.05, UB 1.16): the option (a)
  1.31x RLF failure did not reproduce. Other findings: the 2b gate does not stop the catastrophic associational
  referees (CS:granger_by etc. eligible at R -10.6 to -11.1, harm from about 18,400 prot_min deferrals that no guard
  measures); the reviewers suggested gating by deferring unresolved actions, while the frozen protocol does the
  opposite; the reviewer CONTEXT file still has the old (wrong) sleep-edge RLF explanation.
- 2026-10-02 USER DECISION: keep PMRT rename (strict-rule reading accepted; settled).
