# exp-b status (2026-10-04): DONE (P0 + P1 merged, tables + report built)

READY: Exp B complete (descriptive). Report: docs/xmethod/EXP_B.md s.9; tables / costs in results/exp_b/.

## Result (2026-10-04)
- Counts: 1328 / 1328 expected units recorded; 1316 ok, 12 infeasible; missing 0, errors 0, duplicates 0,
  conflicts 0, dirty 0, no mixed-platform dataset. Infeasible = the 12 cdl units P0 projected (n >= 9 751, R-13, Q9).
  - P0 `xm-expb-p0b` 144; Kaggle `xm-expb-p1a` parts 0-3: 214 (06:44Z); `xm-expb-p1b` parts 4-7: 314 (07:07-~12:25Z,
    9bbeeb4, Py 3.12.14, pins 75 / 75, counts = v4 in all 19 slices); VPS `xm-expb-p1-v1` parts 8-15: 656 (exp-c,
    219bf0b, Py 3.12.13). Every part = its planned partition.
- Costs (Kaggle ref; VPS via factors_vps_py3.12.13.json, orchestrator ruling vps.md Q4, xm/exp-c aee2fab): 82.0
  core-h; cdl 53.7 (24.0 = infeasible at 7200 s), notears 12.9, pc_eq 5.5, pmrt_nl_eq 4.1, rest <= 2.0.
- Headline (BY, recall 60 / pooled; GT-NULL 60 / pooled): pmrt_nl_eq .39 / .55, .05 / .14; linear adjusted arms
  (pmrt_eq / r3, pcorr, hac_fb, granger_eq) .49-.52 / .69-.72, .05 / .10; frozen v4 primary .56 / .79, .10 / .10.
  Unadjusted corr / granger_native and mscr reject the P_placebo column at .40-.59. Graph learners recall .11-.34.
- Caveats in the report: mscr (sleep only) and cdl rows are partial coverage, their null rates rest on the single
  sleep NULL hypothesis; slice-cluster CIs with 2-5 clusters are narrow / zero-width.
- Table renderer now prints hits / n and dataset coverage; new `exp_b costs` subcommand (tests 3 pass, ruff clean).
- No Kaggle job running or planned (Kaggle kept free for EVAL). No local pollers.

Branch xm/exp-b from feat/v2 5a811de (worktree xm-classic). Never pushed. Plan: docs/xmethod/EXP_B.md.

## Done
- **Data** (read-only, Kaggle kernel outputs only): eval `e6p-disc-v4ev-1-a/-b` (600 eps), placebo
  `e6p-disc-v4plc-1-a` (40); DEV not used. Truth = the stored v4 GT, copied verbatim to results/exp_b/gt_v4.json
  (source sha256 4da798e9...; TRUE 29 / NULL 21 / INDET 10).
- **Code.**
  - e6_bridge.py: v4 loaders -> unit tables -> one Dataset per (slice, family). Action = sgn x logged level, design
    logged categorical_rows with the exact v4 propensity, P_placebo with the same design.
  - exp_b.py: prep / counts / run / merge / table / project / kaggle / colab, on campaign.py (5df74f9, dataset hook,
    R-55).
  - Specs: exp_b (19 arms, 76 datasets, 1328 units), exp_b_pilot (144), exp_b_rest (1184).
  - Tests: 3 pass; ruff clean.
- **Laptop smoke** (old non-v4 cache): 74 / 74 ok after 4 truth-free bridge fixes (EXP_B.md s.2).
- **P0 pilot**, Kaggle `xm-expb-p0b` (23:17-00:58 UTC, 5665 s wall, 4 vCPU; launched at 3 RUNNING). The first try
  `xm-expb-p0` failed at push (sources need owner/slug, fixed in 9f54800).
  - 144 / 144 ok, 0 not-testable / STOP / error.
  - Records: commit 9f54800, clean, Py 3.12.14, pins 75 / 75.
  - Tables rebuilt from the frozen JSONL (sub v4, pi0 .5 / .5); unit counts = the stored v4 analysis for all 19
    slices.
  - Files: results/exp_b/p0/. Tables (20 MB) local in scratchpad/e6_dev/runs/xm-expb-p0b/out/tables.
- **Measured cost** (Kaggle CPU-s, max at n <= 5064): cdl 4036 (linear in steps, ~.78 s / step, cap 16 000), notears
  927, mscr 385, pc_eq 222, pmrt_nl_eq 200, two_tower 41, every other arm <= 5.
- **P1 projection** (results/exp_b/p1_projection.json, power-law fits from P0):
  - ~54 core-h for 1184 units: cdl 31, notears 13, pmrt_nl_eq 3.6, pc 3.3, rest < 3.
  - 12 cdl units projected over the 7200 s budget (up to ~12.6k s): every dataset with >= ~9.1k units, i.e.
    pooled carrier / ptx / prot_min, s300 carrier x2 / prot_min x2, s120 prot_min x5.
  - No other arm is near the budget (notears max ~3.9k s at n 50 731).

## Questions: all ANSWERED
- Q1 ANSWERED (orchestrator, 2026-10-04): campaign.py + dev_power.py copied from xm/dev-runs; re-synced before P1.
- Q2 ANSWERED (orchestrator): tau = leave-one-slice-out conformal; pooled / placebo in-sample, labelled.
- Q3 ANSWERED (orchestrator): DEV-v4 not used.
- Q4 ANSWERED (orchestrator): context kept (equal information with frozen PMRT).
- Q5 ANSWERED (orchestrator): mscr at n <= 1000 (R-54).
- Q6 ANSWERED (orchestrator): granger kept, labelled.
- Q7 ANSWERED (orchestrator): GO P0 only, lower priority, launch at < 4 RUNNING, <= 1 Kaggle session.
- Q8 ANSWERED (orchestrator): GO P1; 1 Kaggle session at a time (5th slot); Colab only if it accepts runtimes and < 3 jobs; no new P1 session once the Study A EVAL launches.
- Q9 ANSWERED (orchestrator): run the over-budget cdl units, measured per R-13.
