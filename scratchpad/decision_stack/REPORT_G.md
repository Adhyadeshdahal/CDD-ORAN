# Worker G report (policy-effect WM + conformal gate) — orchestrator summary, 2026-09-28
Files: cdd_oran/decision/effect_model.py, gate.py, tests/test_decision_effect_model.py (10 tests).
- Estimand Δ_r(π|x) = E[C_H(π) − C_H(accept-all)|x] per cost component (viol LL/eMBB/BE, RLF, kWh, churn); labels
  horizon_outcomes(trace, t_e, H) from region_labels; H re-choosable without recollection.
- PolicyEffectModel: K members on episode bootstrap; member predicts m_k(x,π) − m_k(x,accept-all) (paired, accept-all
  ≡ 0); ridge on context basis + policy features + policy×modifiers + neighbour-policy terms (or "hgb"); stabilised
  clipped IPW from logged pol_prop (all weights 1 with today's context-free propensities); AIPW audit tool.
- ContextSelector none/all/weights (MSCR/SHAP/topology (R,R) weights) = graph-as-soft-prior hook; topology_weights.
- EffectWM (privileged=False): EffectScore(mean,std,members,ood,stratum); ood if deviating in out-of-box/stale context.
- ConformalGate: selection-aware split conformal on the SELECTED plan's (pred − realized) improvement; pooled or
  episode_max; per-stratum q with fallback; abstains (q=inf) with too few records; oracle_record via TrueSimWM.
- Validated synthetic only: confounded planted effect recovered (bias < .5 vs −4.8; naive +1.1); spill recovered;
  gate coverage ≥ 1−α−.03 when calibrated on selected plan (under-covers if on random candidates).
- NOT validated: E6 ranking/regret (step-4 gate), real-data coverage, graph selector value, guardrail bounds.
- OPEN: (1) H=90 > D=20 → label window includes later random draws: estimand = "π then logging policy" (add future
  randomized policy features as covariates, or lower ε after a perturbation); (2) data: 260 rows/episode; residual SD
  eMBB 26.5 UE-s → ~6 episodes to detect a 5 UE-s mean effect, ~280 for the full context basis → restrict modifiers
  or collect far more; (3) calibration needs ≥30 oracle records per stratum (2 TrueSim rollouts each).
- Parity: arbiter must forward every second's obs to EffectWM.observe; collector _context → module-level function;
  churn_left mismatch under epoch_churn_cap.
