READY-TO-MERGE
# audit-classic2 status (2026-10-02)

Branch xm/audit-classic2 from feat/v2 dea83cf. This was a report-only audit: no method code was changed. It ran on local CPU only,
with no cloud jobs. Seeds used: DEV 3_000_000-004 only.

## Running
- nothing

## QUESTIONS
- none

## HAND-BACK

Deliverables:
- `scratchpad/xmethod/audit/classic_v2.md`: verdict per method and the exact fixes.
- `scratchpad/xmethod/audit/classic2/`, scripts and their outputs:
  - `arms_check.py` / `.json`: covers checks 2, 4 and 5;
  - `not_testable.py` / `.json`: check 3;
  - `degenerate.py` / `.json`;
  - `smoke.py` / `.json` / `smoke_runs.jsonl` and `smoke_breakdown.py` / `.json`: check 6;
  - `tune_arm.py` / `.txt`.

Verdicts:
- two_tower, shap_dag, pc and granger: OK.
  - pc has a LOW disclosure about the R4 node set.
  - granger has a LOW optional fix: a silent NaN for a collinear source, at kappa 0 only.
- `_classic_common`: FIX NEEDED. `ClassicBase.tune()` always tunes the default (eq) arm; the native arm cannot be tuned through it.

Key numbers:
- Not testable at kappa .25, n 1000, seed 3_000_000: 0 in every world, regime, method and arm. The smallest residual share is .052
  against the 1e-10 threshold.
- granger eq equals the statsmodels OLS test given Z_eq to relative 1e-10.
- BY declarations: 0 mismatches. Signs: 0 mismatches over 293 candidates.
- Smoke: granger eq raw level .053 on nulls and BY declares no null. shap_dag and two_tower declare the E1 R2 indirect lagged nulls
  (P0->K2, P1->K3) in 5 of 5 seeds; this is design-blind behaviour, disclosed.

Tests: `uv run --group baselines pytest tests/test_xmethod_classic.py` gives 42 passed. Without the group, 5 tests fail on the shap
import (finding D).

Dependencies added: none. Deviations from the brief:
- granger was also run on E1 R2 and E2 R2, labelled descriptive, alongside its study world E3 R2.
- E4 used lam 1.0 for the check-3 counts.
