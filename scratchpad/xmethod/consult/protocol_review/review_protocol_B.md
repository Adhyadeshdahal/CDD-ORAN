# Adversarial review B: PROTOCOL_A (DRAFT) before freeze

Reviewer stance: a journal referee (telecom / causal ML) looking for unfairness to baselines and overclaiming.
Read: PROTOCOL_A.md, CONTRACT.md R-1..R-28, PROTOCOL_NOTES.md, status/protocol.md, audit/{pmrt_core,two_tower,
granger,pc}.md, fix-classic2 status, generate.py truth docstring. Read-only; nothing in the repo was edited.

## Verdict
The protocol is careful about integrity: freeze hashes, disjoint seeds, FINAL/PROVISIONAL gating, deviations log.
Its weak point is the claim logic. As written, C1 can be passed by known strawmen (i.i.d.-null tests on serially
dependent data). C2 and C3 are tested only on PMRT, with an "absence of evidence" criterion. "Using the design
restores validity" is never given a verdict for any design-using test other than PMRT. Fix S1-S6 before freezing.

## 1. Fairness to baselines
1a. Equal information (s.5, R-17). Good that it is the primary arm. Gaps:
 - Native-only status for shap_dag, notears and two_tower is a choice, not a necessity. XGBoost (SHAP-DAG) takes
   extra features: add setpoints and lagged actions as features and read SHAP of the focal column only. NOTEARS can
   take setpoint and lag nodes as roots (forbid incoming edges, as PC tier-0 does). two_tower can take them as extra
   inputs. Leaving them native means s.9 A1/A2 compares design-blind methods with pmrt_eq and calls that the primary
   table. Either add eq variants or put them in a separate "design-blind by construction" block of V1/V2 and keep
   them out of A2/V3.
 - "Equal information" is not "equal benefit". Z_eq (R-3 + setpoints of ALL designed actions + 2 lags of ALL
   actions) is high-dimensional: about 3x the action count plus lagged KPIs. PMRT uses Z only in a ridge adjustment,
   so extra Z costs it at most variance. For cmi_knn, rcot2 and pdcor, a 20-40-dim Z destroys power, and with local
   shuffles it can also cost validity. Report |Z_eq| per world in s.5. Pre-register a secondary minimal
   design-sufficient arm, Z_min = R-3 + sp:focal (+ focal lags), which is the information needed to make the dither
   exchangeable. Without it, a referee will read a poor eq-arm recall for kNN/kernel tests as an artefact.
1b. Authors' nulls (R-13). rcot2's LPB4 null is liberal in R1 (F4 .084/.065). Under R-20 it is then "inv" almost
   everywhere, and under s.10 C1 it cannot count as a design-blind failure (R1 ABOVE). It is effectively removed from
   every comparison. The CI tests are already put on a common B 9999 BC resampling resolution (R-9). Add rcot2 with
   a permutation null as a secondary arm (or promote B2's tau scoring of rcot2 in its table), and say in s.4 that the
   authors' null is kept for faithfulness. Same note for pc: Fisher-z under the pinv fallback in E1/E3 (audit pc #2).
1c. Tau from placebo (s.6, T2). Tau = 2nd-largest of about 20 x |K| placebo scores, so each test runs near a 1-2 %
   per-test error rate, while p arms run BY-FDR at q .05. Consequences:
 - V3 paired recall differences across different error-control regimes (BY-FDR vs placebo-quantile) are not a like
   comparison. Restrict V3 to pairs under the SAME rule. For tau arms, compare against pmrt_eq scored with tau
   (B2 already computes it; move that pair into the primary).
 - The tau validity flag on the placebo is near-guaranteed by construction (tuned on the same column's
   distribution), so tau arms pass on the placebo almost automatically. Only the truth-null rate is informative.
   State that in s.8, and do not count tau arms as "valid" on placebo evidence alone.
 - The 2nd order statistic of about 100 tuning values is noisy (pc R1 null FPR .121 from 30 placebo scores, audit
   pc #1). Consider 40 tune seeds, or report tau's bootstrap spread per cell.
 - Tau per (n, kappa, regime) cell is generous to the baselines: empirical placebo calibration is itself a way to
   use the design (the placebo follows the R2 design, R-10 amended). This bears on C1/C2 (section 3).
1d. Budget (s.7, R-13). 7200 CPU-s single-thread with RLIMIT_CPU:
 - GPU paths (cmi_knn torch neighbour search, two_tower, XGBoost) are not counted by RLIMIT_CPU. Declare whether EVAL
   runs CPU-only, or else how GPU time is budgeted. Otherwise the budget is unequal in either direction.
 - Q5 "any infeasible unit makes the cell infeasible" removes a near-limit method's whole cell because of one
   straggler seed, and with it the method's C1/C2 counts and power. Use a rule that T3 decides feasibility from DEV;
   in EVAL, a cell counts if at least 90 % of its units finish, with the count reported. Or grant a fixed 2x
   straggler allowance to every arm.
1e. Missing standard design-blind competitor. Neither arm has a dependence-aware but design-blind test: a pcorr /
   Granger with a HAC (Newey-West) or block-bootstrap null, or a block permutation over setpoint blocks. The R2
   placebo failures of corr/granger/permutation tests come from autocorrelated X (20 setpoint blocks) times
   autocorrelated residuals (spurious-regression variance inflation). That is textbook. A referee will say C1 shows
   that i.i.d. nulls fail on serial data, not that ignoring the design fails. Add at least pcorr_native_HAC (and
   Granger-VARX with HAC) to set D. This is the single most likely rejection point.

## 2. C1-C3 rules (s.10): trivial passes and stacking
 - C1: SUPPORTED needs ">= half of D" failures, and D = corr, granger_native + the native CI tests admitted by T4. If
   T4 admits few or none, D is {corr, granger}, both already seen to declare the placebo 2-4x in R2 (PROTOCOL_NOTES;
   integration). C1 then passes on two known strawmen, the second of which is E3 only. Require |D| >= 4, including
   at least one dependence-aware test (1e); otherwise the verdict is "not assessable". Report C1 per arm, not only
   as a majority.
 - C1 direction was known from DEV/integration before the freeze. s.0 must say so explicitly (R2 corr/granger
   placebo 2-4x; T5 DEV validity of all arms).
 - C1 favours detection at large n: 8000/24000 cells give tight CIs, so any small inflation flags ABOVE. Report the
   per-n pattern; that is fine, but do not call it "invalid at all n".
 - C2/C3 asymmetry: a baseline fails if the lower bound is > .05 (evidence of excess), while PMRT passes if the pooled
   lower bound is <= .05. Fewer seeds or wider CIs make PMRT's pass easier. Replace (b) with an equivalence-style
   bound: pooled upper bound <= .065 (pre-registered margin; DEV pooled E2 R2 .054 [.044, .065] fits). Keep (a).
 - C3 per-cell (a) is nearly vacuous at S = 40: E4 R3 has one P_placebo p and one P_placebo_conf p per dataset, so
   per-cell Wilson half-width is about .07, and ABOVE needs about 5/40. E4 is cheap for every arm (pmrt 1-4 CPU-s):
   set S_E4 = 200 for R3/R4 (cf. the aud2 R3 check, 200 seeds per cell).
 - PARTIAL "only n = 500 fails" is a pre-registered escape hatch, and n 500 is unchecked on harness worlds (aud2 #6).
   Run the E2 R2 n 500 DEV null check before freezing, then keep or drop the clause on that evidence. Record which.
 - Stacking: C1 is tested on native arms, C2 on pmrt_eq, C3 on pmrt_eq. No component compares design-using
   non-PMRT tests, and "supported" needs all three, so the PMRT-specific parts look like the whole claim.

## 3. Is "using the design restores validity" tested on non-PMRT tests?
No. s.10 C2 gives a verdict only for pmrt_eq. The eq CI tests (pcorr_eq, mscr_eq, pdcor_eq, rcot2_eq, cmi_knn_eq,
pc eq, granger eq) are "reported next to it (no verdict)". Given the s.1 claim text, this is the main overclaim
risk. With sp:focal in Z, pcorr_eq's residual X is the i.i.d. dither, so pcorr_eq may well be valid in R2 and more
powerful than PMRT. Pre-register C2' (general): over the eq arms, the share valid in R2 where native is ABOVE, with
a SUPPORTED threshold (e.g. >= half). Then C3 becomes what is PMRT-specific: validity by construction in all cells
plus R3. Also add the eq CI arms to E4 R3 (V6): with ctx in Z, a linear eq test may also hold level there. If it
does, "stays valid under logged confounding" is not PMRT-specific and the paper must not imply it is. Optional
design-using competitor: a propensity / IPW (Horvitz-Thompson) test in R3.

## 4. Drafter's Q1-Q5
 - Q1 (exclude E4 from T1): AGREE. Per-seed recall is 0/1, so a paired power calculation is meaningless there. But
   size E4 separately (S_E4 = 200, see section 2) for validity and sign accuracy.
 - Q2 (re-run tune seeds at the freeze commit): RE-RUN. s.11 FINAL requires code.commit == freeze commit for every
   record. Reuse would need an exception path, and byte-identity of method + world + score code is hard to audit.
   Cost is +50 % of measure, acceptable.
 - Q3 (who adds the EVAL mode): the campaign.py owner (dev-runs) implements it before the freeze, an auditor other
   than its author reviews it, and it is in the freeze commit. Test: refuses EVAL seeds on a sha mismatch, refuses
   DEV/EVAL seed mixing, and records the protocol sha in every record.
 - Q4 (S 40-100): floor 40 OK for p arms on many null candidates; E4 at 200 (above). Keep cap 100 but pre-state the
   MDG report (already in T1). Also budget CI-test cost from DEV before fixing S (unknown now).
 - Q5 (rules): CHANGE: C1 minimum |D| and per-arm reporting; C2/C3 equivalence bound; C2' for eq arms; the
   infeasible-cell rule as in 1d. F_max = Binomial(N, .025) 95 % quantile is fine, but cells are correlated (same
   seeds across n, shared seed clusters), so state that it is approximate.

## 5. Must change before freezing (by section)
 S1 s.4/s.5: add a dependence-aware design-blind arm (HAC or block-bootstrap pcorr; Granger-VARX + HAC) to set D.
 S2 s.10: C1 needs |D| >= 4 incl. S1, reported per arm; add C2' for the eq arms; C2/C3 (b) as upper bound <= .065.
 S3 s.7/T1: S_E4 = 200 for E4 R3/R4; run the n 500 E2 R2 null check, then fix the PARTIAL clause.
 S4 s.5: eq variants (or an explicit separate block) for shap_dag / notears / two_tower; report |Z_eq|; Z_min arm.
 S5 s.6/s.9: V3 only between arms under the same declaration rule; pmrt-tau vs tau arms in the primary; tau-arm
   validity judged on truth-null, not placebo.
 S6 s.1 wording: "needs no outcome model" -> "validity does not depend on the outcome model" (pmrt uses a ridge W);
   scope "RIC-style" to the one R2 design tested (20 blocks, +/-.10 dither), or add a block-count/dither sweep;
   C1 is "design-blind tests with i.i.d./exchangeable nulls" unless S1 also fails.
 S7 s.7: state CPU-only EVAL or a GPU budget rule; replace "any infeasible unit" with the 90 % / T3 rule.
 S8 s.0: list everything seen pre-freeze that bears on the verdicts: corr/granger R2 placebo 2-4x, T5 DEV validity,
   R-14 tuned on DEV truth-null rates (aud2 #3), the PMRT-favourable R2 design chosen in Stage 0.
 S9 s.4: rcot2 permutation-null secondary (or tau) so it is not silently excluded; two_tower stays "adaptation" and
   moves out of "published O-RAN" (row-share score fixed, F3 is a regression check only, audit #3).
