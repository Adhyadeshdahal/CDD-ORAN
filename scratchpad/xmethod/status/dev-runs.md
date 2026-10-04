# dev-runs status (2026-10-04): READY-TO-MERGE (R-59): campaign.py + vps_run.py audit fixes F2-F9 done

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
- pmrt_nl_eq T3 cost pilot DONE (results/dev/pmrt_nl_cost/, in the freeze): max 1148 CPU-s (E2 n 24000); E4 R3 <= 103.
- R-43 pmrt_nl_eq DEV run DONE (xm-dev-pmrtnl-k1, 1440 / 1440 ok; results/dev/pmrt_nl/).
- R-59 DONE (feat/v2 d48cc69 merged in; audit results/audit/campaign_eval_audit.md): F2 EVAL 2x-cap breach =
  infeasible with cpu_s, never re-run (OOM / other signals stay error); F3 --skip-complete-from applies the merge's
  accept_rule (mode, protocol sha, EVAL spec sha; dispatcher done-keys keep run_mode + spec sha); F4 integrity
  stamps cost_table_sha256, part, parts; EVAL merge refuses mixed cost table / part count; F5 each part writes
  rc_<i>.txt, session status non-zero if any part failed (gate-safe; vps_run exit_code + status show it); F6 EVAL
  refuses --budget != spec, F7 refuses non-isolated runs; isolation, budget, cap factor stamped, per-record limits;
  F8 vps_run scope CPUQuota = procs x 100 %; F9 merge summary sets python / lock / spec_file sha (+ isolation,
  budget, cost table, parts). 7 new tests (EVAL via run_units, child stand-in); campaign + eval_analysis + exp_b +
  exp_c tests pass (2 Linux-only skips). DEV behaviour unchanged (records gain stamps). F1 = xm-harness.
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
