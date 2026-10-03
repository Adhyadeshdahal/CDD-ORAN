READY-TO-MERGE

# hac (branch xm/hac from feat/v2 7f84b04, 2026-10-03): rulings R-32, R-33, R-36, R-37

1. **R-32 `pcorr_hac`** (`cdd_oran/xmethod/methods/pcorr_hac.py`, registered in `classic.py`): OLS t-test of the
   source in Y ~ 1 + Z + source, rows in time order (`info_order`), Newey-West HAC SE, Bartlett kernel, Andrews (1991)
   AR(1) plug-in bandwidth L = floor(1.1447 (alpha(1) n)^(1/3)) on the tested coefficient's score x~_t u_t (FWL),
   factor n/(n-k), t(n-k) (statsmodels `use_correction=True, use_t=True`). Z = `_classic_common.cond_set` (native =
   R-3 set, concurrent='all'). BY per family (R-2/R-6), sign = sign(coef) = pcorr-given-Z sign (R-4), not testable in
   `score.py` format (exact fit of 1+Z+source, or source collinear), no RNG. CPU ~.2 s (E1/E5) / .7 s (E2) at n 1000.
   - F2 PASS: coef / SE / p == statsmodels OLS HAC on the same design and L (rel 1e-10 / 1e-8), synthetic + 4 worlds.
   - F4 (2000 reps/cell, through the adapter; FIDELITY_CLASSIC "pcorr_hac F4"): at .05, iid .049 / .054 / .051
     (n 500/1000/4000), plain .046-.055; AR(1) rho .5: HAC .070 / .062 / .055 vs plain .13 / .12 / .12; rho .8: HAC
     .104 / .084 / .060 vs plain .36 / .34 / .35. **PARTIAL**: HAC holds level at n 4000 and removes most of the
     inflation, but over-rejects at n <= 1000 under strong persistence (known Bartlett-HAC finite-sample bias).
     Reported, not tuned (Q3).
2. **R-33 eq_min**: `covariates.design_covariates(..., setpoints="all"|"focal")` + wrapper `eq_min_covariates(data,
   focal)` = R-3 set + `sp:<focal>` (default unchanged; helper == legacy pmrt test still passes). `cond_set` arm
   "eq_min"; pcorr_hac arms native (default) / eq_min (any other arm raises). |Z| of P0 native / eq_min / eq (R2):
   E1 8/9/23, E2 14/15/41, E3 9/10/24, E4 2/3/8, E5 8/9/23; R1/R3/R4 eq_min = native (no setpoint).
3. **R-36 estimand check** (`scratchpad/xmethod/estimand_check.py`, `results/estimand_check.json`): **PASS, all 18
   cells** (E1-E5 x R1/R2, E4 R3/R4 x lambda {0,.5,1,1.5}). Method: do(A_t = a') on a deep copy of the env, then
   `advance()`; tape coordinate-keyed => byte-identical exogenous draws (unit-level CRN contrast). (A) 3000 random
   states over the whole ID range, (B) every row of the regime's own trajectories, DEV seeds 3_000_190-199, n 1000,
   kappa .25, 2 values per action = 20 000 contrasts per action and cell. All 140 truth-null primary (real action ->
   KPI) pairs have max |Delta| == 0.0 exactly in A and B; secondary lagged-KPI nulls too. Placebos are never applied
   to the SCM (zero by construction). Sensitivity: every true edge moves K_{t+1} (none zero), and a planted
   P0 -> K2 leak in E1 is flagged FAIL (negative control). Note: E5 P0 -> K_harm (K1) is nonzero at only 5 % (R1) /
   12 % (R2) of units (the gate), a recall ceiling, not an estimand error. 842 CPU-s total (local).
4. **R-37** (added task): `_classic_common.cond_set` drops every `P_placebo_conf` column (`concurrent:`, `@t-L`,
   `:missing`; `is_diagnostic_column` / `drop_diagnostic`) from every other source's set, all arms (granger eq,
   pcorr_hac, pcorr_given_Z signs follow). PC: primary graph without the conf node; conf candidates read from a second
   PC run with every node (`notes['diagnostic_graph']`). Tests: scrambling `P_placebo_conf` leaves every other
   candidate identical (pc eq/native, granger eq, pcorr_hac native/eq_min; E4 R3 and R4), and the same check fails
   with the rule disabled. FIDELITY_CLASSIC Arms updated (supersedes the audit-classic2 C disclosure), plus the
   pcorr_hac F2 / F6 / F4 entries.

Tests: `uv run --group baselines pytest tests/test_xmethod_{covariates,classic,harness,pmrt,noise,pcorr_hac}.py` ->
283 passed. ruff clean on touched files. No new dependencies (statsmodels 0.15.0 already in uv.lock).
Deviations: estimand run 14 CPU-min locally (slightly over the ~10 CPU-min guide; rows cut 4000 -> 1000 per seed
for that reason, enough for exact-zero checks). Branch is on 7f84b04; feat/v2 is now e1d12f1 (citests merge + R-37
text), no file overlap; citests tests not run here.

## Questions

- Q1 R-37 says "any method": notears, shap_dag and two_tower still use every action column, incl. `P_placebo_conf`,
  as a joint variable / feature (E4 R3/R4), so the diagnostic can still shift their other scores. Align them the PC
  way (fit without it, score conf candidates from a second fit)? Not done (outside the brief's scope).
- Q2 eq_min for a lagged-KPI source (secondary family) = its native set (it has no focal setpoint). OK?
- Q3 pcorr_hac small-n over-rejection under strong persistence (rho .8: .104 / .084 at n 500 / 1000). Keep as
  specified (Andrews, no prewhitening) and report, or add a pre-registered variant (Andrews-Monahan prewhitening or
  fixed-b critical values)? Same question for how R2 (20 setpoint blocks) behaves: not measured here (R-21 DEV check).
- Q4 `ClassicBase.tune` still uses the R-2 2nd-largest placebo tau; R-29 (conformal tau) is not implemented on this
  branch (shared by every classic adapter; pcorr_hac's primary rule is BY, tau is secondary only).
