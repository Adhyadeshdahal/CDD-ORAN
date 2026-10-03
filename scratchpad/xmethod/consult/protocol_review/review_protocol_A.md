# Independent referee report: PROTOCOL_A (draft) + eval_analysis.py

Scope: docs/xmethod/PROTOCOL_A.md, scratchpad/xmethod/{CONTRACT.md, PROTOCOL_NOTES.md, status/protocol.md,
eval_analysis.py}, plus spot checks of cdd_oran/xmethod/score.py (placebo_tau) and worlds/generate.py (RNG keys).
Read-only; nothing in the repo edited.

Overall: a careful pre-registration (frozen hashes, disjoint seed blocks, truth used only for scoring, FINAL/
PROVISIONAL guard, intersection-union over C1-C3). The decision rules are not rigged in PMRT's favour in any gross
way. But four things would draw a "major revision" from a referee: (i) the tau rule's implied false-positive level
is not .05 and differs by world; (ii) the validity flag rewards low-information cells (absence of evidence);
(iii) the chance-flag count F_max assumes independent cells at p0 = .025, which does not hold; (iv) C2 tests only
PMRT, so "using the design restores validity" is not actually tested as a general statement.

## 1. C1-C3 decision rules (section 10)

Rigged / impossible? Mixed stringency, not one-directional:
- Per-cell flag (R-20, "ABOVE iff bootstrap lower bound > .05") is LENIENT for any arm: a cell with few null
  candidates passes even at true level .10. E4 has 1 placebo (+1 P_placebo_conf in R3) per seed; at S = 40 a placebo
  flag needs about 7/40 hits (P(flag | true .10) is about .2). C3 rests on exactly these cells, so C3 is weakly tested.
- Pooled condition (b) "pooled lower bound <= .05" is STRICT: pooled over ~25 cells x S seeds x ~20 nulls the SE is
  about .002, so a true .055 fails. Right for PMRT ("valid by construction" should be exact), but if the same rule is
  applied to asymptotic eq arms (see C2b) it fails them for practically irrelevant excess.
- Recommendation: keep R-20 for gating power, add an equivalence-style practical criterion pre-registered now:
  Bradley's liberal band (rate <= 1.5 alpha = .075; upper CI bound <= .075) for "practically valid", and a cell
  whose upper CI bound exceeds .10 is "inconclusive" (not "ok") and is excluded from power comparisons.

F_max = 95 % quantile of Binomial(N, .025): right idea, wrong calibration.
- p0 is not .025. The flag is an OR over up to 4 rates (null_raw, plac_raw, null_decl, plac_decl). BY declaration
  rates never fire under the null, but null_raw OR plac_raw is roughly .03-.045 per cell, and the percentile
  cluster bootstrap with 40 clusters under-covers (more false flags). Bonferroni-safe p0 = .05.
- Cells are not independent. RNG streams are keyed by (seed, world, regime, label), not by n (generate.py): R1 data
  at n 500 is a prefix of n 1000 etc., and kappa cells share draws. Chance flags cluster across n within a
  (world, regime); the count's 95 % quantile is larger than Binomial gives.
- Fix (pick one, pre-register): (a) p0 = .05 and count at the (world, regime[, lambda]) level, a unit flagged if
  any of its n-cells is ABOVE after Holm over its 5 n; or (b) calibrate the flag-count null by simulation that
  reproduces the nested-prefix structure (uniform p-values on nested data). (a) is simpler and conservative.

">= half of R2 cells ABOVE" for C1: acceptable as a magnitude rule (it demands "broadly invalid", not "invalid
somewhere"), but add a statistical leg: R2 #ABOVE > F_max(N_R2) AND pooled R2 rate lower bound > .05 (the mirror of
C2). Also: the design-blind failure mechanism here is a 20-block effective sample size (setpoint clusters), which
WORSENS with n; small-n cells may legitimately pass, so "half" is defensible. Two code issues: (1) `verdicts()`
silently skips a D arm with no counting R2 cells (`if not r2: continue`), shrinking the denominator; fix D's
membership at the freeze, a missing arm counts as "not a failure". (2) Arms invalid in R1 (rcot2, liberal authors'
null .065-.084) stay in D's denominator but can never be failures; either pre-register "D = native p arms valid in
R1" with the others reported as "invalid regardless of design", or keep as is and state it is conservative.

Should C2 also be judged on the other eq arms? Yes. As written, C2 is "PMRT is valid in R1/R2", which is C3's
by-construction property, and the general sentence "using the design restores validity" goes untested. Split:
C2a = pmrt_eq (current rule); C2b = for each eq arm whose native partner is a C1 failure, the same (a)/(b) rule
with the practical band; SUPPORTED if >= half pass. If C2b fails while C2a passes, the honest claim becomes
"design-based inference restores validity; adding design covariates to model-based tests does not suffice",
which is a stronger and more interesting finding. Pre-register that wording now so it is not a fork later.

## 2. Drafter's questions

- Q1 (exclude E4 from T1): AGREE; per-seed recall is 0/1 there. But C3 lives in E4 R3, so add a validity-driven
  seed rule for E4 R3/R4: S_E4 large enough that a true placebo rate of .10 is flagged with power .8 (about 150-200
  seeds; E4 x pmrt is cheap).
- Q2 (re-run vs reuse tune records): use a FRESH tune block inside the EVAL range (e.g. 3_190_000-019), not the DEV
  tune seeds; same cost as re-running and removes all DEV contact from tau. Reuse of DEV records is the weakest.
- Q3 (EVAL mode owner): dev-runs (owner of campaign.py) implements it with a test, protocol worker reviews; it must
  be merged before the freeze and campaign.py's sha256 recorded in the freeze commit too.
- Q4 (S floor 40 / cap 100): CONFIRM for E1-E3/E5, but S should also satisfy a validity MDE target (per-cell
  truth-null CI half-width <= .02 after deff); report the per-cell MDE for placebo-only cells.
- Q5 (rules): see section 1. "PARTIAL when only n 500 fails" is acceptable only because it is pre-registered and
  reported as not SUPPORTED; keep it symmetric in spirit (no analogous carve-out is applied anywhere else).
  "Any infeasible unit makes the cell infeasible": keep (no seed-level selection), but see M5.

## 3. Flaws a referee could reject on

- R1 tau rule level (score-only arms). tau = 2nd-largest pooled placebo score => out-of-sample P(placebo score >
  tau) = 2/(M+1) with M = #placebo scores = 20 x #KPIs. E4 (1 KPI): 2/21 = .095; E1 (4 KPIs): 2/81 = .025. Score
  arms are then judged against .05: structurally flagged in E4, over-conservative in E2. Fix: conformal rule,
  declare iff score > the ceil(.95 (M+1))-th smallest placebo score (out-of-sample level <= .05 by
  exchangeability), or judge tau arms against their implied level. Applies to the secondary "p arms scored with tau".
- R2 estimand. Truth = direct SCM edges; PMRT tests a sharp null of no effect of the focal (lag-1) random part;
  CI tests test conditional independence. They coincide only if no truth-null candidate has a nonzero lag-1 total
  effect (E3 cascades, E5 "chain"). Add a pre-freeze check: for each truth-null candidate, intervene on the focal
  random part in the SCM and confirm the outcome is unchanged. Otherwise "invalid" may be correct detection.
- R3 forking via missing cells. counts() drops infeasible / incomplete / few_seeds cells from C2/C3 silently.
  For the focal arm any non-counting cell must cap the verdict (NOT EVALUABLE if > 10 % of cells missing).
- R4 cross-platform datasets. PROTOCOL_NOTES: datasets byte-identical per platform only. If arms of one dataset run
  on different platforms, the hash check makes the run PROVISIONAL forever. Pre-register: all arms of a dataset run
  in one shard on one platform, or data generated once and shipped.
- Not fatal but state: validity defined as "not significantly above" (R-20) is absence of evidence; the practical
  band fixes this. Not-testable candidates leave the validity denominator: require a minimum testable share
  (e.g. >= 50 %) for an "ok" flag. DEV tuning: only tau and no-clip (R-14, chosen from DEV null rates on E2 R2,
  disclosed) are DEV-informed; acceptable with disclosure since EVAL seeds are fresh. Multiplicity across C1-C3 is
  handled by the intersection-union rule; V3 paired-t per cell is many tests: label descriptive or use a
  simultaneous (e.g. max-t over cells) band. Pooled clusters (world, seed): the same seed drives all worlds' RNG,
  so cluster on seed alone (conservative).

## 4. Must-fix vs nice-to-have

Must-fix before freeze
- M1 Fix the tau level (conformal order statistic) or judge tau arms against their implied level (R1).
- M2 Recalibrate F_max: p0 = .05 and count at (world, regime, lambda) level with Holm over n, or simulate (sec. 1).
- M3 Add the practical-validity band (Bradley .075) and "inconclusive" when upper CI > .10; gate power on it.
- M4 Split C2 into C2a (pmrt_eq) and C2b (other eq arms), pre-register the fallback claim wording.
- M5 Missing / infeasible / incomplete focal cells cap C2/C3; D membership fixed at freeze; no silent skips.
- M6 Lag-1 total-effect check of every truth-null candidate (R2).
- M7 E4 R3/R4 seed count from a validity power target (C3 otherwise near-untestable at S = 40).
- M8 One platform per dataset (or shipped data); campaign.py EVAL mode hash in the freeze commit.
Nice-to-have
- N1 C1 statistical leg (pooled R2 lower bound > .05) beside the half rule; D restricted to R1-valid arms.
- N2 Fresh EVAL tune block instead of DEV tune seeds (Q2).
- N3 Cluster on seed (not (world, seed)) for pooled rates; BCa or studentized bootstrap instead of percentile.
- N4 Minimum testable share for the flag; report per-cell MDE for validity.
- N5 A block-aware model-based baseline (cluster-robust or block-permutation pcorr) in the native arm, to pre-empt
  "the native tests only fail because they treat 20 setpoint blocks as n i.i.d. rows".
- N6 V3 simultaneous band or explicit "descriptive, no inference" label.
