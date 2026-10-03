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
