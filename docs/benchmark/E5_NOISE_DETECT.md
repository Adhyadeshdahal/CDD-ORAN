# E5 with noise: MSCR vs SHAP detection (pre-declaration, LOCKED)

**Status:** LOCKED 2026-09-26 by the commit that adds this line. Drafted 2026-09-25 before any noisy-E5 data;
revised per sol review (`scratchpad/sol_review_e5_noise.md`: three-way outcome, registry and provenance at lock,
wording). The user approved running it as part of the overnight chain (2026-09-25). This is a **synthetic
observation-noise stress test** of the E5 CORE discovery corpus. It is not process noise and not the fully
composed noisy E5.

**Disclosure before lock:** a pre-launch timing check ran one TEST corpus (seed 930000, κ = 0.3) through MSCR
and the baselines' importances, and printed that MSCR declared the harmful edge on it. No rule, threshold or
seed was changed after that; the check should have used a DEV seed.

## Why this study exists
E5 was agreed as the realistic **noisy** composed environment (`plans/015` §3: "Noise ON"). Every E5 result so
far is **noiseless**:
- the frozen structure contract (`GATE_CONTRACT_E5.md`) used the noiseless CORE corpus;
- `scripts/e5_baselines.corpus` calls `_update_kpis` directly, so no noise ever reaches the data;
- `e5.py`'s built-in process noise is unscaled N(0, 1) and keyed by time step, which the corpus never
  advances.

Stage 0 then found that on this noiseless corpus, the SHAP-GBDT proxy at a low threshold (τ ≤ 0.02) matches
MSCR (recall 20/20, FDR 0). A plausible reason is that without noise, irrelevant knobs get essentially zero
SHAP importance, so any small τ is clean.

**Question:** with realistic noise, does MSCR's built-in false-discovery control give it an advantage over a
SHAP threshold that has to be tuned?

## Data
- E5 CORE mechanism, as frozen: subdom 0.2, chain_gamma 0, theta 0; params i.i.d. Uniform(id_ranges);
  n = 24,000 per corpus.
- **Observation noise, added by the corpus generator:** y_k ← y_k + ε, with
  ε ~ N(0, (κ · σ_k)²) i.i.d. per row and per KPI.
  - σ_k is the frozen standardization spread (`E5_KPI_MOMENTS`): K_ben 4.634914, K_harm 24.662764,
    K_mid 57.680719, K_dist 20.192141.
  - **κ ∈ {0.1, 0.3}**, both reported.
  - RNG: `default_rng([seed, 5, int(1000·κ)])`, separate from the parameter draws.
- For scale: ±0.8·σ_K_harm is the harmful edge's in-gate **half-range** (±20 units over the P0 range), not
  an average row effect. It is active in only about 5% of rows, so the global signal is far smaller. The two
  κ levels are synthetic mild and moderate noise, not a calibrated O-RAN noise model. The realized
  per-KPI noise fraction is reported.

## Detectors and threshold selection
- **MSCR**, frozen v2, per-target BY q = 0.05 over the 4 params.
- **SHAP-GBDT proxy** (`e5_baselines.shap_mask` recipe) and **pooled |corr|**, each with a relative
  threshold τ ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}.
- The baseline τ is chosen per κ on DEV seeds 0–19, same rule as Stage 0 F-detect: the highest recall on the
  harmful edge subject to DEV family FDR ≤ 0.05. If no τ qualifies, the result is "no FDR-valid operating
  point". **Tie-break:** among τ with equal DEV recall, pick the lower DEV FDR, then the larger τ. The
  selected τ, every DEV (recall, FDR) and the test curves are reported. DEV seeds 0–19 are the same
  parameter-seed identities used in earlier noiseless E5 work, with fresh noise streams. They are used only
  for τ selection and are not an independent replication.

## Seeds
TEST: E5 **930000–930019** per κ. **At lock**, `SEED_REGISTRY.json` gains this study's E5 DEV and TEST usage
and the noise stream tag (5, 1000·κ). The harness asserts TEST disjointness against ALL prior E5 uses. It
writes the registry sha256, git HEAD, and the code, config and noise-generator hashes before scoring.

## Metrics and decision rule (identical to Stage 0 F-detect)
- **Metrics:** recall on the harmful edge P0→K_harm, and family FDR (per seed, the mean over KPIs of
  V/max(R, 1) over the 4 params; bootstrap over seeds).
- **"SHAP matches MSCR" at a given κ:** the one-sided 95% paired-bootstrap lower bound of
  (recall_SHAP − recall_MSCR) ≥ −0.05, AND the SHAP family-FDR upper bound ≤ 0.05.
- **Three-way outcome, reported separately per κ and per baseline (SHAP, corr):**
  1. **Baseline matches MSCR:** by the rule above, AND MSCR itself meets its target (recall ≥ 0.95, FDR UB
     ≤ 0.05). "Matching" a failing MSCR (e.g. both at zero recall) is not outcome 1.
  2. **MSCR meets its target while the baseline fails the predeclared operating-point criterion:** MSCR
     recall ≥ 0.95 and FDR UB ≤ 0.05, while the baseline does not match. This is a narrow operating-point
     claim, not general superiority.
  3. **Inconclusive, or both fail.** This includes every case where MSCR misses its own target. What the
     baseline did is then reported, as a limit of MSCR at that κ.

  (Clarified 2026-09-25 before lock, after a tiny-size smoke test gave "matches" with both at zero recall.)

  A **recall-advantage** claim additionally requires the one-sided 95% paired-bootstrap LOWER bound of
  (recall_MSCR − recall_baseline) > **+0.10**. Without it, only outcome 2's narrow wording is allowed. There
  is no pooled or global "noise shows MSCR is better" claim across κ and detectors. Each of the four
  (κ, baseline) cells stands on its own. At N = 20 these are screening rules, not formal equivalence or
  FDR guarantees.
- Also reported: full τ-curves, and MSCR's recall and FDR at each κ.

## Honest scope
This is one mechanism (E5 CORE) with synthetic Gaussian observation noise. It is not the fully composed E5
(chain + confound), and not O-RAN-grounded noise. A κ at which MSCR itself loses recall is reported as a limit
of MSCR, not hidden. The frozen noiseless E5 structure result is unchanged; this study adds the noisy condition
the original design required.

## Cost
Measured before lock: about 15 s per corpus at n = 24,000 (MSCR on the GPU plus SHAP) × 40 corpora × 2 κ,
which is about 20–30 min. Device and threads are resolved at run time and recorded in the provenance.
