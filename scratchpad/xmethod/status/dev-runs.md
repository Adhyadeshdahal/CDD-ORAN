# dev-runs status (2026-10-04): ALL DEV RUNS DONE (10 arms, CI, pmrt_nl, cdl); final T1 S = 40; EVAL mode + R-57 + VPS lane DONE

Branch xm/dev-runs from feat/v2 dea83cf; feat/v2 acd3dfa merged in; cdl files only from feat/v2 5a811de. Never pushed.

## Done
- Steps 1-2: driver `cdd_oran/xmethod/campaign.py` + `dev_power.py` (R-12); pilot 1648 / 1648 ok. EVAL mode (R-35,
  Q6, Q7): frozen-protocol guard, integrity stamps, lock install + pin_check, one platform per dataset, T3, 2x cap.
- Steps 3-5 (10 arms): 41 840 / 41 840 ok (a7b6e36, Py 3.12.13). results/dev/full/: REPORT.md, dev_cells.json, t1.json,
  t1_pairs.json, agg.json, merged.jsonl.gz. Validity (824 cells): VALID 268, INVALID 249, INCONCLUSIVE 247, NO_READ
  60. T1: S_power 14, S = 40 (floor), MDG .082. Max 156 CPU-s per dataset.
- CI-test pilots (e89a340) -> Q10 (results/dev/pilot_ci_*; projection results/dev/pilot_ci_projection.json).

- CI DEV run, plan C (Q10; R-48 pdcor, R-49 cmi_knn dropped): specs/dev/ci_c.json = pcorr x7 + rcot2 x3 full grid
  + mscr x3 n <= 1000. Kaggle k1-k5, Colab c1b / c2 / c3, Lightning l1-l3 (stopped 13:01; rest re-run via
  --skip-complete-from). Merge 58 480 / 58 480 ok, 0 missing / errors; 39 duplicate input lines resolved to one
  platform per dataset (0 mixed); method code identical in all records (17 176 dirty = launcher fix only).
  results/dev/ci_c/: REPORT.md (feat/v2 eval_analysis), dev_cells, t1, t1_pairs, agg, merged (+ summary).
  * 1138 cells: VALID 122, INVALID 444, INCONCLUSIVE 572. mscr INVALID in every R2 cell (null raw .93-1.00) -> Q11.
    pcorr / pcorr_hac / pcorr_hac_fb native INVALID in R2 (null raw .21-.46); their eq / eq_min arms .04-.06.
  * T1 (10 arms + 13 CI arms, focal pmrt_eq): 108 pairs, S_power 14, S = 40 unchanged; redone with pmrt_nl_eq
    (R-43) when its run is in. Cost: max 719 CPU-s per dataset (mscr_eq_min n 1000); nothing near the budget.

- FINAL T1 + REPORT (R-58(2); R-56; focal pmrt_nl_eq): results/dev/final/ (REPORT.md, notes.txt, t1, t1_pairs,
  dev_cells) over every DEV record (10 arms + 13 CI + pmrt_nl_eq + cdl; 2086 cells, 106 840 records, 0 screen
  issues): 115 pairs, S_power 14 (pc_eq E5 R2 n 4000; cdl max 11), S = 40 (floor), cap 60 not binding, min power
  at S .999, max MDG .083. pmrt_nl_eq: 0 INVALID cells (VALID 11, INCONCLUSIVE 13).
- cdl DEV run (R-51) DONE: specs/dev/cdl.json full grid 5080 / 5080 ok (n 8000 / 24000 too: arrived in time, so
  specs/dev/cdl_le4000.json was not needed), 0 missing / errors / dup, 0 mixed platforms. results/dev/cdl/ (merged,
  agg). Kaggle k3 / k4 / k6 / k8 / k12 / k13 + r7, Colab c1-c3, Lightning l1; dispatcher unscheduled 11:59 UTC.
  Code: cdl / classic / all cdd_oran except campaign.py identical over 8 commits; every Kaggle code bundle checked
  vs its commit (k13 = 44b0a31's campaign.py, launcher option only). Validity (100 cells): INVALID 45, VALID 30,
  INCONCLUSIVE 13, NO_READ 12. NOTE (orchestrator): cdl INVALID in E1 R2 / E3 R2 at every n >= 1000 (truth-null
  declarations .09-.21; confounded placebo <= .08); R-53's F4 "primary family VALID" does not carry to these DEV
  cells. E2 R2 / E5 R2 VALID at n <= 4000. Reported, not tuned on. CPU-s max 1654 (n 24000) << 7200.
  LIGHTNING l1 (parts 60-63, user GO): 2.77 studio-h, ~0.42 credits used, ~0.10 left (est.; API stale at .5168).
- pmrt_nl_eq T3 cost pilot DONE (checklist B; p1-040214 96d3a57 + p2-040523 900725c; results/dev/pmrt_nl_cost/,
  cited by the xm-harness freeze): per dataset n 8000 E2 471-500, n 24000 E2 1049-1148 CPU-s; E4 R3 n 8000 33-37,
  n 24000 94-103: feasible at every n; R-58(3) fallback not triggered.
- R-43 pmrt_nl_eq DEV run DONE: Kaggle xm-dev-pmrtnl-k1 (e645167), specs/dev/pmrt_nl.json (gbm, T1 cells):
  1440 / 1440 ok, clean; results/dev/pmrt_nl/. Overrun 7.9 h vs ~3.5 h = costlier units (35 / 53 / 143 CPU-s mean
  at n 500 / 1000 / 4000 vs <= 57 est.), not contention.
- R-55 DONE (2a0cb02; proc count fixed 21607d4): records log load {start, end}; EVAL <= 1 process per vCPU.
- R-57 DONE: EVAL_PYTHON = "3.12.14" exact; EVAL preconditions and pin_check(python=...) refuse any other
  patch ("python" mismatch) before data, on every platform (check runs in campaign run); EVAL setup installs
  3.12.14 exactly (uv). DEV unchanged. Tests updated + pin_check test; file 47 + 2 Linux-only pass.
## QUESTIONS
- Q1-Q8 ANSWERED (Q5: flag a cell only with >= 10 measure seeds; Q6 torch version pin; Q7 Py 3.12 EVAL).
- Q9 ANSWERED (pmrt-diag; reserve seeds 3_000_160-189 unused). feat/v2 pmrt_core adds only the nonlinear
  statistics (v2): pmrt_eq / pmrt_r3 DEV rows stand (linear path v1 unchanged). R-43 run DONE (above).
- Q10 ANSWERED (user): plan C (+ R-48, R-49). Q11 ANSWERED (user) = R-54: mscr stays in Study A as a
  reported-INVALID arm; EVAL runs mscr at n <= 1000 only; never in C2b (mscr R2 null raw .93-1.00).
- R-51 ANSWERED (user GO): cdl DONE above. Freeze: needs campaign.py + dev_power.py + every adapter (item 2).
- VPS lane DONE (daded38): campaign vps = vps_run.py (xm/exp-c, CLI v1) push + venv (Py 3.12.14, exact pin
  check) + launch (<= 7 procs, systemd scope); dispatcher lane (init --vps): 1 job of 7 parts at a time, pulled
  each tick, requeue-able, 30 min backoff on a safety refusal. cpu_quota now reads the process's own cgroup
  (the scope's 700 %), not only the root.
- VPS LIVE SMOKE PASSED (b8b33d6; xm-citests gave the slot): queue xm-vps-smoke (init --only vps, new) -> tick launched
  xm-vps-smoke-v1-041259 (specs/dev/vps_smoke.json, 4 DEV units, 2 procs, vps_run 1b33ca1 venv Py 3.12.14, 0 pin mismatches)
  -> exit 0 in < 1 min -> tick pulled + checked (0 incomplete). Records: 4 / 4 ok, clean, host vps AMD EPYC-Rome, cpu_quota 7; nothing left running.
