# E6 end-task metric (DRAFT, not frozen)

Status: DRAFT 2026-09-27. Margins marked **[DEV]** are set from the DEV headroom grid (seeds 11–15) before any TEST
run; the frozen version is committed and hashed before TEST seeds are touched. Background: three independent arguments
in `scratchpad/e6_design/metric/` (statistician, operator, reviewer) and `SYNTHESIS.md`.

## 1. Primary endpoint
**SVR** = violated UE-seconds per UE-hour over the scored window, all UEs in the denominator, no slice weights
(`E6Env.score()["svr"]`). A UE-second is violated if:
- **LL:** the maximum head-of-line delay in that second exceeds 100 ms, or the UE is in RLF re-establishment, or out of
  coverage. A plain 50 ms HO interruption is not an outage; the packets it holds back count through their delay.
- **eMBB:** backlogged for ≥ 0.2 s and user-perceived throughput < 2 Mb/s, or in RLF outage / out of coverage.
- **BE:** in RLF outage / out of coverage.

Why plain SVR: it has no free parameter. Slice-averaged SVR and an SLA-contract penalty index rank all 16 xApp subsets
almost identically on DEV (`SYNTHESIS.md`), so weighting is not a lever worth a researcher degree of freedom. They are
reported as secondaries.

## 2. Guardrails (all must pass; non-inferiority, one-sided, paired by seed)
Reference arms: **noarb** (all xApps, accept everything) and **freeze** (reject everything).

| guardrail | rule |
|---|---|
| per-slice violation rate | each slice's rate ≤ 1.10 × noarb **and** ≤ 1.10 × freeze |
| RLF / UE-h | ≤ 1.10 × noarb |
| severe incidents | ≤ noarb + 0.5 per scored hour |
| energy | ≤ noarb kWh + an absolute margin fixed on DEV (constraint, not objective; the noarb–freeze energy gap is only 2–8 %, so a ratio-of-differences rule is noise) |
| goal retention | for each xApp whose own-KPI gain under noarb (vs freeze) is positive, the arm keeps ≥ **[DEV, proposed 0.9]** of that gain |

Own KPIs: MRO = RLF/UE-h, TS = eMBB violation rate, ES = energy kWh, SLICE = LL violation rate.
Retention is measured against noarb, not against each xApp's solo run: under noarb ES already keeps only ~47% of its
solo energy saving (DEV seed 11), so a solo-based bar would fail every arm, noarb included.
Freeze fails goal retention by construction; this is what stops "turn everything off" from winning.

## 3. Comparators (all scored on the same seeds and scorer)
none/freeze, noarb, priority, knob-lock, best static xApp subset (chosen on DEV among subsets passing the guardrails),
and every published method behind a declared mitigation wrapper, with settings in its favour and the same DEV tuning
budget. Each published method is also reported on its own native metric.

## 4. Strong-edge decision rule (split bar, decided 2026-09-27; `scratchpad/e6_dev/decision/DECISION.md`)
Per comparator b: ρ_b = geometric mean over cells of (Σ_seeds violations_ours / Σ_seeds violations_b), paired
stratified seed bootstrap (BCa 95 %).
- **Primary (conflict mitigation):** for EVERY goal-keeping comparator (faithful published methods behind their
  declared wrappers, noarb, priority, knob-lock, best static subset, tuned static): point estimate ρ_b ≤ 0.75,
  one-sided reject ρ_b ≥ 0.80 at α = 0.025, win in ≥ 12/16 cells, all guardrails pass. The one-sided test is
  evidence that ρ_b < 0.80, NOT that ρ_b ≤ 0.75; 0.75 is a descriptive margin. No multiplicity correction is
  needed for an "all nulls rejected" claim.
- **Sanity vs freeze (switch the xApps off):** ρ_freeze ≤ 1.00 (one-sided, same test at margin 1.05). Freeze is
  plotted in every figure; it is not a goal-keeping policy and fails goal retention by construction.
- **Secondary (reported, not required):** ρ_freeze ≤ 0.75 (the stricter reading).

## 5. Secondaries (descriptive)
Slice-averaged SVR; SLA penalty index with a β sweep; per-slice Pareto table; weight-simplex robustness plot; energy,
RLF, HO, ping-pong; severe incidents; oracle-normalised headroom captured.

## 6. Open before freezing
- [DEV] retention fraction and cell factorial (load × mobility × KPM × update → 16 cells).
- Episode length for TEST (DEV uses 10 scored minutes; spec says 30).
- Arbiter action class: O-RAN WG3 conflict mitigation only (accept/reject/modify/defer/lock/rollback), churn
  parity with noarb on the same tape; no free-form own writes.
- Scenarios: base (calm control) + 3GPP SON stress cases (`E6_STRESS_SCENARIOS.md`).
