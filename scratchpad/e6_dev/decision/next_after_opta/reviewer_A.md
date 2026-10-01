# Reviewer A (statistician, skeptical of post-hoc rescues)

## 1. Recommendation
Write the NOT ELIGIBLE result up now as the primary confirmatory outcome (option 1). Allow exactly ONE disclosed follow-up, and make it 2b in an equivalence form. Report it as a separate, labelled study, and stop after it whatever it shows. Reject 2a and 2d.

## 2. Strongest argument for it
The failure is not a surprise. On DEV, GT2_own (the GT map with its nbr/far edges removed) was already ineligible at RLF 1.26, so "a map missing spillover-on-guard edges gets ineligible" was a known mechanism before the eval ran. The flaw is in the referee's logic, not in PMRT's sensitivity: MapGateV2 reads a non-significant edge as a zero edge. A sleep->nbr RLF z of 0.2 to 1.3 means "underpowered", not "safe". 2b fixes this with a general rule: allow an action only if harm on every guard KPI is shown to be bounded (a TOST/CI upper bound below a tolerance). That is principled, it applies the same way to every map, and it does not need knowledge of which edges were missed.

## 3. Strongest argument against the main alternative (2a)
2a is the classic forking path. We know the GT effects (+.114, +.119, -.061), we know the three missed edges, and we would be designing a matched filter until those three z-scores cross threshold. Then we would test on the same simulator, where the same spillover geometry generates the data. Any gain is partly fitted to the test. On top of that, MapGateV2 was itself designed after K-A on DEV seeds, and this would be a third design iteration chasing a third failure. A reader cannot tell 2a apart from tuning to the answer. 2d (more data) has the same problem: n would be picked after seeing z near 1. A rough calculation (z about 1 needs 6 to 9x the data to reach about 2.5) also makes it expensive.

## 4. Concrete next steps (pre-register before any new run)
- Freeze the rule from the discovery output alone: for each candidate action and each guard KPI k in {svr, nonprot, ll, rlf} and each rel in {own, nbr, far}, require UB90(effect) <= tau_k, otherwise defer. tau_k is derived mechanically from the existing 1.10 guard (the per-edge effect that would push pooled ratio to 1.10 at the baseline action rate). It is not chosen by looking at DEV R.
- Leave unchanged: the 1.10 guard, retention .90, R/R*, E, D1 (re-selected max), D2 (Holm), the bootstrap seed and the N_BOOT. Do not loosen anything.
- Apply 2b to every arm (PMRT, GT, granger, granger_by, corr, two_tower, SHAP). Add a pre-specified D3: R*(PMRT) - R*(never_sleep) with LB90 > 0. My prediction: 2b defers most sleep requests, so the honest competitor becomes never_sleep (+.221), not the associational maps (which fail on wrong/placebo edges, and 2b does not fix those).
- PMRT needs a test-inversion CI per edge. Implement it, unit-test it and freeze it before any data run. DEV may be used only for code smoke tests, not for tuning tau or the deferral logic.
- Use a fresh seed block registered in SEED_REGISTRY, a new protocol doc and a hash-frozen artifact. Write down the predicted outcome and the stop rule in advance: if 2b is NOT ELIGIBLE or fails D3, there is no further rescue and the paper reports both studies.
- In the paper, list every eval look (K-A, step-2, option a, 2b) and state that 2b was designed after the option (a) failure.

## 5. Honest claims available now
- Discovery from confounded incumbent logs: PMRT declared 22 edges, 18 true with the correct sign, 0 wrong sign, 0 placebo. granger_by declared 47 (6 wrong, 29 placebo), granger 32 (6 wrong, 19 placebo), corr 10 (2, 8), SHAP 5 (0, 2). The v4 randomized-log study is PARTIAL: everything passes except P2 vs granger_by at n 60.
- In a closed loop, associational maps are harmful (R -9.8 to -11.5), and PMRT's map is the only learned map with a large positive R (+.47; SLA viol .86x).
- The pre-registered verdict is NOT ELIGIBLE: RLF 1.31x [1.10, 1.56]. Missing weak, rare-event neighbour spillover edges on a guard KPI makes a referee unsafe. This is a substantive finding about causal-map referees.
- Not claimable: superiority to baselines under the safety constraint, superiority to never_sleep, or "better than GT". R above GT comes from taking the harmful sleep actions.
