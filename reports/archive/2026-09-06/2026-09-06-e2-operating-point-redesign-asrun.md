# E2 operating-point redesign — AS-RUN freeze report

**Date:** 2026-09-06
**Protocol (pre-registration):** `docs/benchmark/E2_OPERATING_POINT_REDESIGN.md`
**protocol_commit:** `8ed31bf61b261f625a5bc0d889acd9571c04969c` (repo HEAD at run time)
**Branch:** feat/v2
**Verdict:** **ACCEPT** — all §5 acceptance criteria (A–D) and all three degeneracy guards pass.

Anti-p-hacking: the pre-registered change was applied verbatim (`id_ranges[6] -> (1.5, 4.0)`,
`KPI_MEAN_STD[5]` re-derived at n=1e6). `tau_E2 = 0.10`, the P0 grid, the bank rules, and the
degeneracy definitions were **not** touched. THERMAL: single-core, all BLAS/OMP thread envs = "1",
no parallelism.

## Step 1 — Standardization constant re-derivation (deterministic, n=1e6)

Provenance procedure (`cdd_oran/envs/statistics.py:120-139`):
`get_env_ii_mean_std(param_ranges=ID, seed=0, num_samples=1_000_000)` over `compute_kpis_ii`.
The sampler draws each param coordinate independently sized `num_samples`; KPI[0..4] depend only on
P0–P5 (drawn before P6), KPI5 depends on P0/P6/P7.

**Validation (mandatory) — reproduce the CURRENT frozen KPI_MEAN_STD (e2.py:41-48):** all six
entries reproduced to ~6 sig figs (max |Δmean|=4.86e-07, max |Δstd|=3.88e-07):

| idx | derived (old ID) | frozen literal | match |
|-----|------------------|----------------|-------|
| 0 | (21.482606, 27.694787) | (21.482606, 27.694787) | OK |
| 1 | (26.776967, 34.651755) | (26.776967, 34.651755) | OK |
| 2 | (40.931657, 44.767996) | (40.931657, 44.767996) | OK |
| 3 | (14.975340, 32.369335) | (14.975340, 32.369335) | OK |
| 4 | (18.755966, 40.460422) | (18.755966, 40.460422) | OK |
| 5 | (-10.140367, 12.583744) | (-10.140367, 12.583744) | OK |

Provenance procedure + seed CONFIRMED. `VALIDATION_REPRODUCED = True`.

**Re-derivation with the NEW range `id_ranges[6]=(1.5, 4.0)` (all other ranges identical):**
entries [0..4] came out **bit-identical** (P6 is K5-exclusive, drawn after P0–P5), only [5] changed.

- **NEW `KPI_MEAN_STD[5]` (full precision):** `(-0.9567299654371699, 4.771898201319839)`
- Frozen into e2.py at 6-decimal format (matching the existing literals): **`(-0.956730, 4.771898)`**
- (Pre-reg exploration estimate was `(-0.961945, 4.788258)` — close, superseded by this deterministic
  freeze value as §4 requires.)

## Step 2 — Env change applied (`cdd_oran/envs/v2/e2.py`)

1. `id_ranges[6]`: `(-60.0, 65.0)` -> `(1.5, 4.0)` (line ~63).
2. `KPI_MEAN_STD[5]`: `(-10.140367, 12.583744)` -> `(-0.956730, 4.771898)` (line ~47); [0..4] untouched.
3. Provenance comment added at the KPI_MEAN_STD change citing protocol_commit 8ed31bf.

Nothing else in e2.py changed. K0–K4 mechanism lines (115-119) and all other constants are untouched.

## Step 3 — Pre-registered gate + degeneracy guards

`scripts/e2_decision_gate.py` (32 K5-positive + 32 K5-negative committed states, env_seed 0..4095 scan,
2-advance lag, panel {K0,K1,K2,K5}):

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| A. mean_positive(gap_norm) | **0.190022** | >= tau_E2 = 0.10 | PASS |
| B. mean_negative(gap_norm) | **0.000000** | <= tol = 0.01 | PASS |
| C. factor-removal positive gap (drop xApp4/K5) | **0.000000** | <= tol = 0.01 | PASS |
| D. feasibility | **32 positive + 32 negative** | 32 + 32 | PASS |

Separation (pos − neg) = 0.190022. Gate `RESULT: PASS`.

(Truth-free prediction in §6 was mean_positive ≈ 0.204; observed 0.190 — in the predicted regime,
comfortably clear of tau. No tuning was done to reach it.)

**Degeneracy guards (per-state, over the 32 positive-bank states; true-K5 threshold −25, satisfy-below):**

| Guard | Count | Verdict |
|-------|-------|---------|
| Reachable-satisfied (some P0 gives true K5 ≤ −25) | **32/32** | PASS (every state) |
| Not-free-at-optimum (true K5 violated at the {K0,K1,K2}-optimal P0) | **30/32 (94%)** | PASS (broad majority) |
| Both regimes coexist (a satisfiable AND a violated P0 both present) | **32/32** | PASS (every state) |
| Genuine cost (satisfying K5 requires moving P0 off the {K0,K1,K2} optimum) | **30/32 (94%)** | PASS (broad majority) |

All guards hold: the trap is a genuine trade-off, not a maximised number or a degenerate/unsatisfiable trap.

## Step 4 — Test state

- `tests/test_e2_decision_gate.py::test_positive_control_meets_preregistered_target`: the
  `@pytest.mark.xfail(strict=True)` decorator was **removed** (kept the assertion and `tau_E2`
  unchanged), since the redesign makes it an expected PASS (strict-xfail would otherwise XPASS-fail).
- `pytest tests/test_e2_decision_gate.py -q`: **9 passed, 0 failed (GREEN).**
- Broader E2 surface `pytest tests/ -k "e2 or E2 or discovery or structure" -q`: **all passed
  (exit 0, ~200 tests, 0 failures).**

**No breakage — nothing to classify as EXPECTED or UNEXPECTED.** These tests are behavioral /
recompute-from-the-live-env (dataset determinism, mechanism consistency, discovery machinery,
gate logic), not frozen-numeric-artifact comparisons, so they track the new env automatically. Per
§7, the previously-run frozen E2 discovery RUN records (RCoT-v1/v2 recovery outputs) are on the OLD
env and are now stale; those are run-output artifacts to be **re-run later**, not test failures in
this suite.

**K0–K4 bit-identity claim — HOLDS (verified two ways):**
1. Structural: `V2Env.reset` (`base.py:91-97`) draws each param from an independent per-coordinate
   RNG keyed on `variable=i`; the range only scales coordinate i. So changing `id_ranges[6]` perturbs
   **only** P6. K0–K4 (`e2.py:115-119`) reference only p[0..5], never p[6]/p[7].
2. Empirical: the n=1e6 re-derivation returned KPI[0..4] mean/std unchanged.
No non-K5 KPI value changed → no red flag.

## Files changed (uncommitted — orchestrator owns commits)

- `cdd_oran/envs/v2/e2.py` — id_ranges[6] tuple, KPI_MEAN_STD[5] tuple + provenance comment.
- `tests/test_e2_decision_gate.py` — removed strict-xfail on the positive-control test.
- `reports/2026-09-06-e2-operating-point-redesign-asrun.md` — this report.

## Decision

**ACCEPT.** All §5 criteria (A, B, C, D) pass and all three degeneracy guards pass. The E2
shared-knob trap is now live (mean_positive gap_norm = 0.190 ≥ 0.10 with mean_negative = 0 and
factor-removal = 0), tau_E2 was not tuned, and K0–K4 remain bit-identical.
