# dev-runs status (2026-10-05): READY-TO-MERGE (EVAL merged): xm-eval-a 150 960 / 150 960 ok, 0 missing / mixed / N1

Branch xm/dev-runs from feat/v2 dea83cf; feat/v2 acd3dfa merged in; cdl files only from feat/v2 5a811de. Never pushed.

## EVAL (Study A, freeze feat/v2 93856c2 merged = 75a8fff) - DONE, MERGED (no analysis run; eval_analysis = xm-harness)
- Run id xm-eval-a: specs/eval/full.json (eval, 150 960 units, 11 000 datasets) from clean worktree CDD-ORAN-wt/xm-eval
  at 5a95186 (freeze tree + dispatcher EVAL fix + cost table results/dev/eval_cost_table.json sha 9a684592...).
- 96 LPT parts (5.25 h at Kaggle speed); 20 jobs: Kaggle k1-k14 (4 parts / session, <= 5 at once), VPS v1-v6 (7 procs).
  Launched 10-04 15:10 UTC, last pulled 10-05 06:52 UTC; 0 requeues, 0 lost parts. Tick unscheduled (06:52).
- MERGE results/eval/merged.jsonl.gz (+ .summary.json): 150 960 / 150 960 ok; missing 0, errors 0, infeasible 0,
  unexpected 0, duplicates 0, conflicts 0, mixed-platform datasets 0, dirty 0. Sets (1 value each): commit 5a95186,
  protocol c5f7a4fe, spec, spec_file, python 3.12.14, lock, isolation fork+rlimit, budget 7200, cost table, parts 96;
  platforms kaggle + vps. N1: 96 raw shards, 150 960 lines = 150 960 keys, 0 with both infeasible and ok.
- merged.jsonl.gz = 104 705 944 B (just under GitHub's 100 MiB file limit).

## Done
- Steps 1-2: driver `cdd_oran/xmethod/campaign.py` + `dev_power.py` (R-12); pilot 1648 / 1648 ok. EVAL mode (R-35,
  Q6, Q7): frozen-protocol guard, integrity stamps, lock install + pin_check, one platform per dataset, T3, 2x cap.
- Steps 3-5 (10 arms): 41 840 / 41 840 ok (a7b6e36, Py 3.12.13). results/dev/full/: REPORT.md, dev_cells.json, t1.json,
  t1_pairs.json, agg.json, merged.jsonl.gz. Validity (824 cells): VALID 268, INVALID 249, INCONCLUSIVE 247, NO_READ
  60. T1: S_power 14, S = 40 (floor), MDG .082. Max 156 CPU-s per dataset.
- CI-test pilots (e89a340) -> Q10 (results/dev/pilot_ci_*; projection results/dev/pilot_ci_projection.json).

- CI DEV run plan C (Q10, R-48, R-49): specs/dev/ci_c.json 58 480 / 58 480 ok, clean; results/dev/ci_c/ (REPORT,
  cells, t1, agg, merged). 1138 cells: VALID 122, INVALID 444, INCONCLUSIVE 572; mscr INVALID in every R2 (Q11).

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
- pmrt_nl_eq T3 cost pilot DONE (results/dev/pmrt_nl_cost/, in the freeze): max 1148 CPU-s (E2 n 24000); E4 R3 <= 103.
- R-43 pmrt_nl_eq DEV run DONE (xm-dev-pmrtnl-k1, 1440 / 1440 ok; results/dev/pmrt_nl/).
- R-59 DONE (465ae56 -> feat/v2 0542a99): F2-F9 in campaign.py / vps_run.py, 7 tests; audit
  results/audit/campaign_eval_audit.md; re-audit PASS-WITH-NOTES (feat/v2 93856c2). F1 = xm-harness.
- R-55 DONE (2a0cb02; proc count fixed 21607d4): records log load {start, end}; EVAL <= 1 process per vCPU.
- R-57 DONE: EVAL_PYTHON = "3.12.14" exact; pin_check(python=...) refuses another patch on every platform.
## QUESTIONS
- Q1-Q9 ANSWERED (Q5 >= 10 measure seeds per flag; Q6 torch pin; Q7 Py 3.12 EVAL; Q9 reserve seeds unused).
- Q10 ANSWERED (user): plan C (+ R-48, R-49). Q11 ANSWERED (user) = R-54: mscr stays in Study A as a
  reported-INVALID arm; EVAL runs mscr at n <= 1000 only; never in C2b (mscr R2 null raw .93-1.00).
- R-51 ANSWERED (user GO): cdl DONE above. Freeze: needs campaign.py + dev_power.py + every adapter (item 2).
- VPS lane DONE (daded38): campaign vps (push, venv Py 3.12.14, launch in the scope); dispatcher init --vps / --only.
- VPS LIVE SMOKE PASSED (b8b33d6; xm-citests gave the slot): queue xm-vps-smoke (init --only vps, new) -> tick launched
  xm-vps-smoke-v1-041259 (specs/dev/vps_smoke.json, 4 DEV units, 2 procs, vps_run 1b33ca1 venv Py 3.12.14, 0 pin mismatches)
  -> exit 0 in < 1 min -> tick pulled + checked (0 incomplete). Records: 4 / 4 ok, clean, host vps AMD EPYC-Rome, cpu_quota 7; nothing left running.
