# Stage 0: three falsification checks before building E6 or the guard (pre-declaration DRAFT)

**Status:** DRAFT 2026-09-25, written before any Stage 0 data; revised per sol review (scratchpad/sol_review_stage0.md: F-dither redesigned, seed registry, L1 specified, paired F-detect rule). User approved the plan ("LGTM, proceed"); this
document goes to sol, then it is locked by commit before running. Design context:
`scratchpad/design_resolver/SYNTHESIS.md` (Revision v2) and `scratchpad/sol_review_synthesis_resolver.md`.

Three checks. Each tests a different claim, has its own pass mark fixed here, and has its own consequence.
None of them validates MSCR by passing; each can only weaken a specific claim by failing.

## Shared rules
- **Fresh seeds, machine-checked.** `docs/benchmark/SEED_REGISTRY.json` lists every seed range already used,
  per world (corpus seeds, bank/env seeds, DEV probes, frozen studies). The harness loads it, asserts that
  every TEST seed is disjoint from it, and writes the registry's sha256 into the results before scoring. DEV
  seeds may overlap earlier exploratory work; TEST seeds may not.
- Dev seeds are used ONLY to pick baseline thresholds and regularization strengths (never MSCR's, which is
  frozen).
- Uncertainty: one-sided 95% percentile bootstrap over seeds (the seed is the unit), 10,000 resamples, RNG
  seed 0. Paired comparisons resample seeds jointly across arms.
- **Family FDR** (used identically on DEV and TEST): for each (seed, KPI), the family is that KPI's
  parameter candidates; FDP = V/max(R, 1). Seed-level FDR = mean FDP over the KPIs, and the bootstrap runs
  over seed-level values. This is an empirical rate on these designs, not a proof of FDR control.
- MSCR = `cdd_oran.discovery.discover_mscr`, frozen config, unchanged.
- A check's consequence applies only to the claim it names. All three are **screening falsifiers** with
  modest sample sizes, not formal equivalence proofs.

## F-detect: does another detector already find the gated harmful edge?

**Claim at risk:** MSCR's detection novelty ("others miss the gated edge").

- **Worlds:** E2 (gated edge P0→K5, n = 4000) and E5 CORE (gated subdominant edge P0→K_harm, n = 24000),
  using the same generators as their frozen studies.
- **Detectors:**
  - MSCR (frozen, BY q = 0.05);
  - SHAP-GBDT proxy (`scripts/e5_baselines.shap_mask` recipe; for E2 the same recipe on E2 rows), relative
    threshold τ;
  - pooled |corr|, relative threshold τ;
  - RCoT-v2 on E2, from its frozen protocol.
  - SHAP interaction values are NOT included: tree-explainer interaction support for HistGBR is not
    guaranteed, and a conditional baseline cannot be frozen.
- **Threshold choice for baselines (no tuning on test seeds):** on DEV seeds (E2 0–19, E5 0–19), sweep
  τ ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}. Pick the τ with the highest gated-edge recall subject to
  DEV family FDR ≤ 0.05, and freeze it per detector and world. If no τ meets that bar, the detector has "no
  FDR-valid operating point"; its highest-recall τ is scored as descriptive only and cannot trigger the
  falsifier.
- **Test seeds:** E2 722000–722019, E5 922000–922019 (N = 20 each).
- **Metric:** gated-edge recall (fraction of seeds declaring the harmful edge) and family FDR.
- **Falsifier (per world, paired):** for some baseline at its frozen τ, the one-sided 95% paired-bootstrap
  LOWER bound of (recall_baseline − recall_MSCR) is ≥ −0.05, AND its family-FDR upper bound is ≤ 0.05.
- **Consequence:** MSCR's detection novelty is dropped for that world, and it is reported as "matched by
  <detector>". It does not by itself stop the decision claims.
- **Descriptive (not gated):** full τ-curves (recall vs FDR) for every baseline on the test seeds.

## F-graph: is the graph's decision value just "fewer inputs"?

**Claim at risk:** that discovered structure (not generic input pruning) is what makes the learned model good
enough to decide on.

- **World:** E2, n = 4000. Decision scoring as in D1: the frozen 32-state bank, H = 1 brute-force pick on
  the learned model, realized on the true env.
- **Common learner:** every arm uses the D1 predictor class and recipe (`learned_world_model.py`: 3 MLPs
  2×64 SiLU, standardized inputs/target, Adam lr 3e-3 with cosine decay, 1500 full-batch epochs, member seeds
  0–2). Every arm sees the identical corpus rows.
- **Arms:**
  - `mscr`: inputs = MSCR graph on that corpus (discovery rerun per corpus);
  - `l1`: all 8 standardized inputs, loss = MSE (standardized target) + λ · Σ_j ‖W1[:, j]‖₂. The sum runs
    over input j's full column of first-layer weights (all 64 hidden units), a group lasso that can zero an
    input entirely.
    - λ is chosen per KPI from {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} on DEV seeds E2 0–4 (five, for compute; a DEV-only choice) as follows: fixed
      80/20 row split (first 80% train, last 20% validation) within each DEV corpus; fit on train; choose the
      λ with the lowest mean validation MSE across DEV seeds.
    - λ is then frozen and the model refitted on the full test corpus. It is selected by prediction error
      only, never by decision loss, and the selected λ per KPI is reported;
  - `shap`: inputs = SHAP-GBDT proxy graph at τ = 0.10 (descriptive);
  - `all`: all 8 inputs, no penalty (descriptive);
  - `true`: true parents (ceiling, descriptive).
- **Test seeds:** E2 723000–723009 (N = 10), paired across arms.
- **Metric:** mean decision loss per seed. Paired difference d = loss(l1) − loss(mscr), per seed.
- **Falsifier:** the one-sided 95% paired-bootstrap UPPER bound of mean(d) is ≤ the margin
  max(0.20, 0.25 × mean loss(mscr)). Read it as: the bound rules out L1 being meaningfully worse than MSCR.
  A pass is NOT "equivalent"; both margin components, the paired seed values and the full interval are
  reported.
- **Consequence:** the graph-value claim for decisions is stopped. The guard/decision-maker claim does not
  proceed on "the graph helps" (it may still proceed on conflict-information grounds in E6, stated as such).
- **No seed top-ups.** If the interval straddles the margin, it is reported as inconclusive; seeds are not
  added selectively.

## F-dither: does MSCR still find the gated edge on bounded experiments?

**Claims at risk:** (a) that a bounded operating regime even *visits* the gate (a data-regime property), and
(b) that MSCR finds the gated edge when it is visited (a method property). They are reported separately.

- **Data regime (piecewise setpoint + local dither):**
  - **Nominal setting:** the midpoint of each ID range.
  - **Operating envelope:** each knob's setpoint stays within ±25% of its range around the nominal.
  - **Setpoints:** 20 per corpus, drawn uniformly inside the envelope, each held for n/20 consecutive rows.
  - **Dither:** each knob independently gets U(−δ, +δ) × its range on top of the setpoint, clipped to the
    ID range. Clipping counts are logged.
  - Logged per row: block ID, setpoint, commanded value and clipped value.
  - Otherwise identical to `generate_rows` (lagged KPIs, prime advances).
- **Levels:** δ ∈ {0.05, 0.10, 0.20}. **Primary: δ = 0.10**, n = 4000; n = 12000 is secondary.
- **Test seeds:** 724000–724019 (N = 20) per δ.
- **Gate diagnostic (truth used only to measure the data regime):** a row is gate-active if its true K5 < −5
  (inside the harm bump). Report the gate-active row fraction and the number of blocks with ≥ 1 gate-active
  row, per corpus.
- **Metrics:** P0→K5 recall; total param-edge recall; family FDR, reported **descriptively only**. MSCR's
  permutation null does not respect block or time structure, so its false-alarm rate is not a valid gate on
  this design. A block-aware test would be a new method (deferred).
- **Falsifier (method, primary δ = 0.10, n = 4000):** among corpora with ≥ 200 gate-active rows, P0→K5 recall
  < 0.80 (an operational screen; its bootstrap interval is reported).
- **Data-regime finding (not a falsifier of MSCR):** if under 25% of corpora reach 200 gate-active rows, the
  bounded regime rarely visits the gate. The deployability claim then requires twin-sourced structure or a
  wider envelope, whatever MSCR's conditional recall.
- **Consequence:** a method failure means MSCR does not work on bounded dither even when the gate is visited,
  so deployability needs twin-sourced structure. The randomized-design results stand either way.

## What is NOT decided here
The E6 design, the guard's objective and calibration method, and any venue. Stage 0 results feed the E6
pre-declaration.

## Cost (estimates, to be measured)
- F-detect: MSCR plus the baselines on 40 test and 40 DEV corpora. The E5 n = 24000 MSCR runs dominate, at
  about 30 s each → about 1 h.
- F-graph: 5 arms × 10 seeds × about 2.5 min → about 2 h, plus the λ selection on DEV.
- F-dither: 3 δ × 20 seeds × MSCR (about 20 s) plus n = 12000 → about 40 min.

All run detached, sequentially, with resumable records.
