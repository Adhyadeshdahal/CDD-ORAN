READY-TO-MERGE

# hac2 (branch xm/hac2 from feat/v2 40d20a2, 2026-10-03): ruling R-38

1. **Fixed-b variant of pcorr_hac** (`config['inference'] = 'fixed_b'`, default stays 't'; version 1.1). Same
   statistic, same Andrews L; reference = Kiefer & Vogelsang (2005, Econometric Theory 21, 1130-1164) fixed-b limit of
   the Bartlett HAC t at b = M / n, M = L + 1 (statsmodels weights 1 - j / (L + 1) = KV's k(j / M)).
   `methods/_fixedb.py`: t*(b) = W(1) / sqrt(Q(b)), Q(b) = (2 / b)[int B^2 - int_0^(1-b) B(r + b) B(r)] (B bridge,
   independent of W(1)) => p = E[2 Phi(-|t| sqrt(Q))] over 20 000 simulated bridges (T 1000 as KV; fixed seed
   (7803, 3801), data-independent; ~80 MB cache per process); b < .01 interpolates to the normal p at b = 0.
   - F3 PASS: at KV Table I Bartlett cv(b) (coefficients verified from the paper PDF), simulated p = .199-.201 /
     .097-.100 / .048-.051 / .019-.021 for nominal .20 / .10 / .05 / .02, b in {.02, .05, .1, .2, .5, 1}.
   - F4 (Kaggle `hac2-f4b-1`, 2000 reps/cell, same datasets as R-32; the t column reproduces the local R-32 run):
     X -> Y at .05, n 500 / 1000 / 4000: iid t .049/.054/.051, fixed-b .047/.053/.051; rho .5 t .070/.062/.055,
     fixed-b .063/.059/.053; rho .8 t .104/.084/.060, fixed-b .091/.080/.056 (plain pcorr .12-.13 / .34-.36).
   - **Selection (R-38, synthetic only): max |rate - .05| = .0405 fixed-b vs .0535 t => set-D member = pcorr_hac
     `inference='fixed_b'`**; t reported as secondary (`results/pcorr_hac_f4b.json`, `selected_set_D`).
   - R-30 verdicts: fixed-b INVALID at rho .5 n 500 and rho .8 n 500 / 1000, VALID elsewhere; t INVALID at
     rho .5 and .8 for n 500 / 1000. **Correction**: my R-32 F4 text (hac.md, FIDELITY) called t at rho .5 n 500
     "INCONCLUSIVE" and n 1000 "VALID"; both are INVALID (lower CI .060 / .052 > .05). FIDELITY F4 rewritten.
     Placebo (iid) marginally INVALID in 2 of 9 cells for both HAC variants (lower CI .050-.052), read as chance.
2. **R-37 extended** to notears, shap_dag, two_tower: primary fit without the `P_placebo_conf` variable / feature /
   input; its candidates from a second fit with every one (`notes['diagnostic_fit']`). Scramble-invariance test now
   covers 8 (method, arm) pairs x E4 R3 / R4 (18 tests); fails without the rule (checked for notears).
3. **R-29 conformal tau**: `score.placebo_tau(results, max_declarations=None, alpha=.05)` default = ceil((M+1)(1-
   alpha))-th smallest of the M finite placebo scores (largest if index > M; M = 0 -> +inf); `max_declarations=k`
   keeps the old R-2 rule for comparison. `ClassicBase.tune` -> `_classic_common.placebo_tau` -> this function,
   `tau_rule = 'placebo_conformal_0.05'`. Tests: harness (M in {1, 3, 18, 19, 20, 39, 40, 200}, float edge
   (M+1)(1-alpha) integer, NaN / non-placebo ignored), classic (tune on 10 SYN datasets, M = 30 -> max).
   **eval_analysis.py (feat/v2) uses `score.placebo_tau` directly (line ~309) => it now applies R-29 with no change.**
   FIDELITY Declaration line updated.

4. **Q2 follow-up (pmrt_core)**: `PmrtCore.tune` now sets `tau = score.placebo_tau(dev runs)` (R-29 conformal) and
   `tau_rule = 'placebo_conformal_0.05'` (was the 2nd-largest score). `test_tune_placebo_threshold` checks tau ==
   `score.placebo_tau` == the largest of the M = 8 DEV placebo scores (M < 19) and 0 placebo declarations;
   `tests/test_xmethod_pmrt.py` 17 passed.

Tests (local, one process): `uv run --group baselines pytest tests/test_xmethod_{classic,harness,eval_analysis,
pcorr_hac}.py` -> all passed (2 skipped, pre-existing in eval_analysis). ruff clean. No new dependencies.

## Questions

- Q1 ANSWERED (orchestrator): keep default 't'; the spec sets `{'inference': 'fixed_b'}` for the set-D arm (key name
  passed to the protocol worker).
- Q2 ANSWERED: pmrt_core switched here (item 4); `_citests_common.tau_from_scores` is switched by xm-citests.
- Q3 ANSWERED: the protocol worker adds `tau_is_pos_inf` to eval_analysis.
