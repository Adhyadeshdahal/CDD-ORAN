# E5 with observation noise: MSCR vs SHAP / |corr| detection — RESULT (2026-09-26)

Pre-declaration: `docs/benchmark/E5_NOISE_DETECT.md`, locked at `10d78f0`. Harness
`scripts/e5_noise_detect.py`. Provenance was written before any record (git head `10d78f0`, no uncommitted
changes in the listed files; hashes, registry sha256 and runtime in `runs/e5-noise/provenance.json`). The run
resolved to MSCR on CUDA with 16 CPU workers, numpy 2.4.2, torch 2.10.0+cu128. Raw records
`runs/e5-noise/records.jsonl`, scores `runs/e5-noise/score.json` (gitignored). 80 corpora (2 κ × 20 DEV +
20 TEST), about 15–20 s each, completed in one pass without error.

**Disclosure (also in the lock doc):** before the lock, a timing check ran the (κ = 0.3, TEST seed 930000)
corpus through MSCR and computed the SHAP and |corr| importances. It printed MSCR's declaration of the harmful
edge (declared) and the timings; the baseline importance values were computed but not printed. No rule,
threshold or seed changed afterwards. The full N = 20 analysis below follows the protocol but is **partially
peeked** at κ = 0.3. A sensitivity analysis excluding that case (N = 19 at κ = 0.3) is reported below; it
changes nothing.

This is a **synthetic observation-noise stress test** of the E5 CORE discovery corpus: one mechanism,
Gaussian noise added to the KPIs. It is not process noise, not the fully composed E5, and not O-RAN-grounded
noise. At N = 20 these are screening rules, not formal equivalence or FDR guarantees.

## Outcome (pre-declared three-way rule, per κ and baseline)

| κ | Baseline | Outcome |
|---|---|---|
| 0.1 | SHAP-GBDT | **1: baseline matches MSCR** |
| 0.1 | pooled \|corr\| | 2: MSCR meets its target, baseline has no FDR-valid operating point (narrow) |
| 0.3 | SHAP-GBDT | **1: baseline matches MSCR** |
| 0.3 | pooled \|corr\| | 2: MSCR meets its target, baseline has no FDR-valid operating point (narrow) |

There is **no recall-advantage claim** in any cell. The paired LB95 of (recall_MSCR − recall_SHAP) is 0.00,
and none can be computed for corr. Per the pre-declaration, there is no pooled or global claim.

## Numbers (TEST seeds 930000–930019)

Realized noise fraction (noise sd / σ_k), averaged over test corpora, was about 0.100 and 0.300 on every
KPI, as designed.

**MSCR (frozen v2, BY q = 0.05):**
- κ = 0.1: harmful-edge recall **1.00**, family FDR mean 0.000, UB **0.000**. Meets target.
- κ = 0.3: recall **1.00**, FDR mean 0.006, UB **0.019**. Meets target.

**SHAP-GBDT proxy.** DEV selected **τ = 0.02** at both κ (recall 1.00 at DEV FDR 0; the tie-break picks the
largest such τ).
- On test at τ = 0.02: recall 1.00, FDR UB 0.000; paired LB95 (SHAP − MSCR) = 0.00, so it matches.
- Test curve, (recall, FDR) by τ:

| τ | 0.005 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| κ = 0.1 | 1.00, 0.000 | 1.00, 0.000 | 1.00, 0.000 | 0.00, 0.000 | 0.00, 0.000 | 0.00, 0.000 |
| κ = 0.3 | 1.00, **0.578** | 1.00, 0.000 | 1.00, 0.000 | 0.00, 0.000 | 0.00, 0.000 | 0.00, 0.000 |

**Pooled |corr|.** No τ reached DEV family FDR ≤ 0.05 at either κ (the best DEV FDR was about 0.18–0.20, at
τ = 0.20). Test FDR ranged 0.19–0.49 at κ = 0.1 (recall 1.00) and 0.18–0.49 at κ = 0.3 (recall 0.95–1.00).

**Sensitivity, excluding the peeked (κ = 0.3, seed 930000) case, N = 19:** MSCR recall 1.00, FDR UB 0.020
(meets target). SHAP at the same DEV-selected τ = 0.02: recall 1.00, paired LB95 0.00, FDR UB 0.000, so
outcome 1 still holds. |corr| still has no FDR-valid operating point (outcome 2). No classification changes.

## What this means
- **The hypothesis behind this study was not supported.** Stage 0 suggested that SHAP matched MSCR on E5 only
  because the corpus was noiseless. With observation noise of 10% and 30% of each KPI's spread, SHAP at its
  DEV-chosen threshold still matches MSCR on the gated harmful edge. **MSCR's detection novelty on E5
  remains dropped**, now also under this noise model.
- **Descriptive (not a claim):** noise narrows SHAP's usable threshold band. At κ = 0.3, τ = 0.005 gives FDR
  0.58, and τ ≥ 0.05 misses the edge entirely. So for SHAP at κ = 0.3, on the tested grid, only
  τ ∈ {0.01, 0.02} works, while MSCR needed no tuning. Here, however, 20 DEV seeds were enough to find that band. Whether the band stays
  findable under harder conditions (process noise, confounding, other gates) is untested.
- Against pooled |corr|, MSCR met its target while |corr| had no FDR-valid operating point. This is the narrow
  outcome-2 wording only; |corr| is a weak baseline.
- On E5, what remains distinctive about MSCR is **BY-adjusted permutation testing that needs no
  per-environment threshold**, with good empirical FDR here. That is a procedural property, not an FDR
  guarantee under noise and not a detection advantage demonstrated here.
  E2 (Stage 0) is still the only world where MSCR was the only detector with FDR control.

## Honest scope
One mechanism (E5 CORE: subdom 0.2, chain_gamma 0, theta 0), randomized parameter rows (n = 24,000),
independent homoskedastic Gaussian KPI observation noise at two synthetic levels. All claims are about this
generator, the fixed SHAP-GBDT proxy recipe and the six-point threshold grid. A future harness should also
reject duplicate or unexpected record keys (sol, non-blocking; the 80 keys here were unique and complete). The DEV seeds 0–19 reuse the parameter seeds of earlier
noiseless E5 work with fresh noise streams (DEV only). The frozen noiseless E5 structure result is unchanged.
