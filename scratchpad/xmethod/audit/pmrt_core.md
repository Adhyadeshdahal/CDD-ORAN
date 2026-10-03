# F7 audit (aud2): pmrt_core

Auditor aud2, 2026-10-02, branch xm/audit-aud2 (from feat/v2 b7456c2). Read-only review; I did not write pmrt_core.
Disclosure: in an earlier role in this study (xm-classic) I wrote the competitor adapters pc / notears / shap_dag /
two_tower / corr / granger. Fairness findings below that concern them should be weighed with that in mind.

**Verdict: PASS-WITH-NOTES.** The code implements what PMRT_CORE.md says. The redraw laws match the harness designs
exactly. The weights are predictable. R-2 / R-4 / R-6 / R-9 / R-10 hold, and the adapter cannot read Truth. The
null rate replicates on fresh DEV seeds. One medium item needs an orchestrator decision before EVAL: the design-information
asymmetry versus the R-3 conditioning sets (finding 1). The missing null evidence for the logged (R3) redraw under
live confounding is now filled: the rate is nominal (finding 2, "R3 check").

Reference used: `cdd_oran/decision/pmrt.py` (module docstring: validity argument, "what is NOT allowed", arms,
test), `cdd_oran/decision/fdr_layer.py` (declare, weighted_step_up, _vectors), `docs/benchmark/METHOD_NAMES.md`,
`docs/xmethod/PMRT_CORE.md`, `cdd_oran/xmethod/worlds/generate.py` + `e4_logged.py` (the true assignment laws),
`cdd_oran/xmethod/score.py`, CONTRACT.md R-1..R-16, xm-pmrt hand-back, and `scratchpad/xmethod/results/*`. There is no
external paper: PMRT is this project's method, so the E6 code is the reference.

## Findings (most severe first)

1. **MEDIUM, fairness: PMRT alone uses the design decomposition, and the R-3 conditioning sets exclude it.**
   `pmrt_core.py:255-257` adds the setpoint (`fixed_part`) of every dither action, including the focal one, as
   covariates. `:258-269` adds two lagged rows of every action. `:273-280` adds concurrent other actions. R-3 gives
   the CI tests other actions + lagged KPIs (+ context) only, and no classic adapter reads `Design` (grep:
   `fixed_part` / `random_part` / `propensity` appear only in pmrt_core and a test generator). In R2, the 20
   setpoint blocks are what invalidate the i.i.d. p-values of corr / granger (placebo declared 2-4x, FIDELITY_CLASSIC
   integration). PMRT's validity edge in R2 therefore comes partly from information that `api.Design` exposes to
   every method but the protocol lets only PMRT use. Redrawing only the dither is legitimate and is the method's
   point. Conditioning on the setpoint is not unique to it. Fix (orchestrator): either (a) add a fairness arm in
   which the CI / association methods also condition on the R2 setpoints (or block indicators) and lagged actions,
   or (b) state in the paper that the competitors run without the design decomposition, and why.
2. **CLOSED (was MEDIUM), evidence gap: no null calibration of the logged-design redraw under
   live confounding.** F4 has no logged scenario (R1, R2, R1t only: `pmrt_fidelity.py:13-16`). In the harness E4
   R3 diagnostics (`results/f4_noclip/diag_E4_R3_*.json`), "real" = 40 p of P_placebo -> K0 only, and P_placebo is
   i.i.d. "swap" replaces the KPI series with the next seed's, which breaks Z -> K0. P_placebo_conf (logged,
   Z-dependent, no effect, Z drives K0) is the one column that tests the R3 CRT, and it was not summarised. The
   integration notes only "0 placebo_conf declarations" at one seed per cell. Closed by this audit's check
   ("R3 check" below): .051 pooled, every cell's interval covers .05. Fix: add the table to PMRT_CORE.md's F4
   evidence.
3. **LOW: one configuration choice used DEV truth.** R-14 (no clip) was chosen from truth-null rejection rates on E2
   R2 DEV (PMRT_CORE.md "Why no clip"). CONTRACT section 5 says "Truth is never used for tuning". The choice is
   user-approved, motivated by validity, and costs power (recall .17 vs .31 at n 1000), so it does not inflate
   PMRT's power. It does give PMRT a validity calibration on these worlds that the competitors (authors' defaults,
   R-13) do not get. Fix: one disclosure sentence ("the clip was switched off after a DEV null-calibration check on
   the benchmark worlds").
4. **LOW, naming / claims: "PMRT core" is PMRT's `plain` arm.** The kernel degenerates to 1 (no matched filter),
   there is no clip, and declarations use plain BY instead of `wby1s`. METHOD_NAMES.md defines PMRT by the matched
   filter, the clip and prior-weighted directional BY. PMRT_CORE.md's kept / reduced / dropped table is accurate.
   Fix: in the paper, never carry E6 `loadsp_c` power or validity claims over to the core, or the reverse. Name it
   "PMRT core (plain arm, design-based CRT)" at first use.
5. **LOW, F2 scope.** The bit-exact equivalence (`results/pmrt_equiv_B9999.json`: 180 hyp, p bit-exact, rel dz
   3e-15) runs pmrt's own weights (`W_override`), pmrt's RNG and fixed B. It validates the CRT engine and the
   categorical assignment, not the core's adjustment. The adjustment agrees only loosely: corr z .91, median |dz|
   .55, max 4.0, 17 vs 24 hypotheses at p <= .05. PMRT_CORE.md says so. Fix: none; keep the wording "engine
   bit-exact" and do not shorten it to "bit-exact with PMRT".
6. **LOW: the harness null diagnostics ran B 999 fixed, not the production R-9 rule.** `pmrt_null_diag.py:1`. Not
   material at .05 / .01. My replication with the production config on disjoint seeds agrees (below). n 500 was not
   checked on harness worlds. It is in the EVAL grid, and it is where the asymptotic (fixed-W) argument is weakest.
   F4 synthetic n 500 was nominal. Fix: one E2 R2 n 500 null check before freezing.
7. **NOTE, harness (not pmrt_core): `Dataset.meta["roles"]` carries structure hints** ("conduit", "non-descendant
   control", "distractor", "gate axis"; `generate.py:107-114, 247-248`). No adapter reads it. pmrt_core reads only
   `meta["primary_candidates"]` (`:493-494`), and the classic adapters read only the candidate lists. Fix: move
   `roles` out of the Dataset (into Truth or the scorer) so a future adapter cannot read it.
8. **NOTE, cosmetic:** `pmrt_core.py:184`, a long run of spaces inside the clipped-normal draw expression (lost line
   break). Harmless.

## What I verified
- **What is redrawn / held fixed** (`run` `:445-471`, `crt` `:351-394`): only the focal action's random part, at
  every row, from its design law. Other columns, the setpoints and W stay fixed. One draw set is shared by all KPIs of
  that action. This matches pmrt.py's test ("the family's modes re-drawn ... every other family fixed").
- **Redraw law = true assignment law, per regime:**
  - R1: uniform on the ID range, or uniform on the 101-point grid for E4 = `_draw_iid`.
  - R2: U(+-.1 * range) dither, with `random_part` exactly the realised dither. The generator asserts no clipping
    and column == fixed + random.
  - R3: the per-row table `policy_probs(Z, lam)` uses the same clip + round-half-up bins as `grid_index`; P0 and
    P_placebo_conf are drawn from independent streams (env eta tape vs `_rng(..., "placebo_conf")`).
  - R4 placebo: clipped-normal mean and variance checked analytically against `:171-185`.
  - Kind `none` gives not applicable, never a permutation.
- **Predictability:** the ridge, GCV lambda, standardisation and burn-in use rows < the block start only
  (`:337-341`). `bad_t` and `_finite_block` use whole-series information only on null-invariant series (Y, covariates),
  which pmrt.py's argument allows. Lagged actions require a true previous step (`:263`).
- **R-8:** concurrent columns = the other designed columns (`:273-280`). In the harness their draws are independent
  of the focal draw given the context: separate RNG purposes for R1 / R2, and the R3 streams above. The generator
  asserts that the committed params equal the design draws, so the env does not couple actions.
- **R-2 / R-6 / R-10:** BY over `meta["primary_candidates"]` (incl. P_placebo). Untested hypotheses keep NaN p, count
  in m and are never rejected (`fdr_layer._vectors`, `weighted_step_up`). P_placebo_conf gets its own BY. tune()
  sets tau = the 2nd-largest P_placebo |z| over the DEV datasets and reads no truth.
- **R-4 / R-9:** sign = sign(S). B 9999 with Besag-Clifford h = 20, p = h / k, else (1 + c) / (B + 1). Two-sided
  exceedance |z_b| >= |z| - tol. RNG `default_rng([seed, 7801, ai, split])`.
- **No Truth access:** the adapter sees only `api.Dataset`, and `score.py` counts a not-applicable candidate as not
  declared (R4 P0 and KPI -> KPI lower PMRT's recall; no advantage).
- **Tests:** `uv run pytest -q tests/test_xmethod_pmrt.py`: 17 passed.
- **Independent null replication** (`audit/aud2_null_rate.py`, production config, DEV seeds 3_000_100-3_000_199,
  disjoint from xm-pmrt's 3_000_000-039; truth-null primary edges incl. placebo, seed-cluster bootstrap):
  E2 R2 n 1000: .054 [.044, .065] at .05 (206 / 3800), .008 [.005, .012] at .01; 303 CPU-s.
  E3 R2, E5 R2, E1 R2 and E2 R1 are in "R3 check and further replications": all nominal.

## R3 check and further replications (Kaggle aud2-pmrt-null-1)
Setup: 4 vCPU, numpy 2.4.2 / scipy 1.18.1 (pin match), 1 thread per process, production config, git dcf92ad.
Scripts: `audit/aud2_r3_conf_null.py` and `audit/aud2_null_rate.py`. Outputs: `audit/results/`.

E4 R3, P_placebo_conf -> K0 (logged Z-dependent design, no effect, Z drives K0). One p per dataset, so the
datasets are independent and the Wilson interval is valid. DEV seeds 3_000_000-3_000_199:

| n | lam | rate .05 [Wilson] | rate .01 | z mean / sd | P_placebo .05 | P0 -> K0 declared, sign -1 |
|---|---|---|---|---|---|---|
| 1000 | .5 | .075 [.046, .120] (15/200) | .025 (5/200) | .11 / 1.02 | 9/200 | 200/200 |
| 1000 | 1.0 | .045 [.024, .083] | .005 | .06 / .97 | 9/200 | 200/200 |
| 1000 | 1.5 | .035 [.017, .070] | .005 | .05 / .95 | 12/200 | 200/200 |
| 4000 | 1.5 | .050 [.027, .090] | .010 | .03 / 1.06 | 10/200 | 200/200 |

Pooled: 41/800 = .051 at .05 and 9/800 = .011 at .01. Every .05 interval covers .05. The lam .5 cell is the
highest. Its .01 rate of 5/200 (Poisson tail ~.05) is unremarkable among 8 rate checks. The confounding strength
(lam) shows no trend, so finding 2's gap is closed: the logged redraw is calibrated under live confounding. The
diagnostic BY family has m = 1, so declarations equal the .05 counts.

Harness null rates on fresh DEV seeds 3_000_100-3_000_199 (truth-null primary edges incl. placebo, n 1000,
seed-cluster bootstrap):

| cell | .05 | .01 |
|---|---|---|
| E2 R2 (local) | .054 [.044, .065] (206/3800) | .008 [.005, .012] |
| E3 R2 | .050 [.039, .061] (104/2100) | .009 [.005, .013] |
| E5 R2 | .049 [.039, .061] (69/1400) | .008 [.004, .012] |
| E1 R2 | .052 [.040, .064] (83/1600) | .011 [.006, .016] |
| E2 R1 | .051 [.042, .059] (192/3800) | .011 [.007, .015] |

These replicate xm-pmrt's numbers on disjoint seeds. Cost per dataset at n 1000 is 0.7-3 CPU-s; E4 R3 is 0.9-1.4
CPU-s at n 1000 and 3.8 CPU-s at n 4000. n 500 is still unchecked on harness worlds (finding 6).
