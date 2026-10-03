# Overnight 2026-10-02/03: what was decided for you (please review; all reversible before EVAL)

You said: "go through it all, don't await me, I review in the morning". Every decision below is in CONTRACT.md
section 8 (R-numbers). Independent reviews / audits are in scratchpad/xmethod/consult/ and audit/.

## Your two answers before sleep
- R-27 kappa = .25 (pooled), sweep .125 / .5.  DEV launch now + 40 extra seeds for validity cells.

## Decided overnight (orchestrator; for your approval)
1. NOTEARS kept, F3 19/20 disclosed.  two_tower labelled "adaptation".  Budget 2 CPU-h / dataset (nothing near it).
2. Protocol reviewed by 3 independent agents (statistics / adversarial / reproducibility) -> R-29..R-36:
   - R-29 score-only methods: threshold now holds 5 % false alarms (old rule gave 9.5 % in E4).
   - R-30 each method x cell = VALID / INVALID / INCONCLUSIVE, same burden of proof for PMRT and baselines.
   - R-31 claim C2 split: C2a PMRT; C2b every other test given the design covariates. Fallback wording fixed now.
   - R-32/R-38 new design-blind baseline robust to serial dependence: pcorr_hac (fixed-b HAC; chosen on synthetic
     data only). Without it a referee could call C1 a strawman.
   - R-33 minimal-design arm (only the focal setpoint added) so high-dimensional covariates don't penalise kNN/kernel tests.
   - R-34/R-39 seeds: E4 300 (rule: >= 90 % chance a correct PMRT passes C3); others from power calc, 40-100.
   - R-35 everything pinned (packages, Python 3.12, torch version), EVAL guard tied to the frozen protocol.
   - R-36 wording "validity does not depend on the outcome model"; estimand check PASSED (no null edge has a real effect).
3. R-37 the placebo-confounder column is never used as a conditioner (any method).
4. R-40 MSCR's "equal-information" arm is not truly equal information -> excluded from C2b, labelled.
   pdCor is a dependence test, not a CI test -> labelled, and not counted for C1.
5. R-41 CMI-kNN runs on GPU (bit-identical to the CPU reference); 2 h GPU wall budget per dataset.

## Findings worth knowing
- Design-blind methods ARE invalid in R2 (smoke): native pairwise Granger 57/160 null edges declared; eq arm 0.
- RCoT's own null is liberal (even the authors' R code: .075-.11); with the full design set E2 R2 null .16.
- SHAP-DAG / two-tower flag many null edges (score-only, no p-value).
- CMI-kNN and pcorr in the equal-information arm hold level in the smoke runs.

## Status at time of writing
- Merged into feat/v2: all method adapters (classic, CI tests, pcorr_hac), audits, protocol v3 (docs/xmethod/PROTOCOL_A.md,
  NOT frozen), eval_analysis.py, EVAL spec.
- Running on Kaggle: DEV campaign (10 arms) + CI-test DEV pilot. Next: CI DEV run, power calc -> fill TBDs -> freeze -> EVAL.

## 09:40 update: DEV full run done; IMPORTANT finding
- pmrt_eq is slightly liberal in E2 R2 at n 1000 (truth-null rate .070 at kappa .125, .084 at .5; .061 at .25),
  fine at n 500 / 4000 and on the placebo. A design-based test should be exact here (estimand check: zero effect),
  so this looks like an implementation flaw (suspect: covariates built from the action's own past dither held fixed
  while the dither is redrawn). Worker pmrt-diag is diagnosing on reserve DEV seeds; any fix must be justified by
  theory and disclosed in PROTOCOL_A s.0. Freeze waits for this.
- CMI-kNN GPU cost: ~40 min per dataset at n 4000, ~5 h at n 8000 (over budget). Full grid at n 4000 may need far
  more GPU-hours than Kaggle's ~30/week: decision pending the pilot projection.
- R-44 Lightning up to 3.77 credits; Colab-first retry policy (user).

## Night 2026-10-03 -> 04 (user asleep; user approved: freeze + EVAL launch tonight if green; cloud only, no laptop shards, no Lightning)
- Evening: CI DEV run done (58 480 / 58 480 ok; results/dev/ci_c/REPORT.md in wt xm-pmrt). R-54 (user): mscr kept as
  reported-INVALID arm, EVAL n <= 1000. rcot2_eq INVALID in most R2 cells (.067-.104) -> C2b fallback wording (R-31).
- Running: cdl DEV (dispatcher, Kaggle + Colab), pmrt_nl_eq T1 DEV (Kaggle xm-dev-pmrtnl-k1).
- Workers tonight: aud1 = finish cdl + pmrt_nl DEV, power calc; xm-harness = freeze package (T-register, EVAL spec,
  launch plan, checklist) on xm/freeze-prep; cdl = Experiment B plan/driver on xm/exp-b (no launch until slots free);
  xm-citests = Experiment C runtime-table script on xm/exp-c.
- Decisions overnight are logged below with the subagent panel's votes.
- 00:20 R-55 (Exp C / T3 timing; xm-citests Q1-Q4): panel of 3 subagents (generalist, skeptical referee, pragmatist)
  unanimous: trimmed paired calibration under controlled load before EVAL; host = platform + CPU model; paper runtime
  table from a dedicated uncontended Kaggle timing run (campaign CPU-s supplementary); pool rows, median/mean/min/max.
  Also: EVAL runs 1 process per vCPU (ci_c ran ~3.5x oversubscribed) and logs CPU model / load per unit.
- 00:45 R-56 (xm-harness freeze-prep Q1-Q3; panel: methodologist, skeptical referee, pragmatist):
  Q1 3/3 keep C2b rule + name rcot2_eq as failing eq arm + unfiltered eq-arm table. Q2 2/3 (referee dissent: set_D
  false, "padding") keep mscr_native in set D + C1 sensitivity with/without mscr. Q3 2/3 (methodologist dissent: keep
  100, cap unlikely to bind) T6 cap 60, S = clip(S_power, 40, 60). REVIEW: Q2 and Q3 were split votes.
  Also answered Q4 myself: calibration anchors at n 500/1000/4000, 3 repeats.
- 01:10 Exp B (cdl worker, xm/exp-b): plan + code ready (19 arms on frozen v4 E6 episodes, read from Kaggle outputs;
  GT as stored: TRUE 29 / NULL 21 / INDET 10; ~71 core-h, cdl ~55). Answered defaults: LOSO conformal tau, full
  context in eq + native, mscr n <= 1000, granger kept (labelled). GO for the pilot only, lowest priority (launch when
  < 4 of my Kaggle sessions run); P1 waits (EVAL first). Copied campaign.py / dev_power.py into its worktree.
- 02:20 Merged into feat/v2: e087160 freeze package (protocol v5, T-register filled except T1 S; final EVAL spec;
  launch plan S 40 ~1 240 CPU-h ~56 h wall on 4 Kaggle + 3 Colab; checklist), aa6d02e EVAL report generator (DEV
  rehearsal: C1 SUPPORTED 5/6 (4/5 without mscr), C2a SUPPORTED, C2b SUPPORTED, C3 NOT SUPPORTED = INCONCLUSIVE cells
  at DEV seed counts, 0 INVALID for pmrt; EVAL uses 300 E4 seeds). eval_report.py frozen with eval_analysis (T7).
