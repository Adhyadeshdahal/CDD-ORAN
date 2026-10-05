# Paper-facing decisions after Study A EVAL (2026-10-05) - PROVISIONAL, for user review

Source: Study A EVAL REPORT (feat/v2 6f04f02; FINAL, claim SUPPORTED); 3-reviewer naming panel (editor, reviewer,
skeptical referee). User: "apply directly, but we review and discuss in the morning". Backups: TERMINOLOGY.md.bak,
WRITING_AGENT_PROMPT.md.bak (same folder). Every new entry is tagged [PROVISIONAL 2026-10-05].

## D1 Names (code -> paper)
- "PMRT" alone = the frozen E6-P test (E6P_PMRT_V4: predictable ridge adjustment + loadsp_c matched filter + wby1s).
- pmrt_nl_eq -> "PMRT-GBM"; pmrt_eq -> "PMRT-Lin"; pmrt_r3 -> supplement only ("PMRT-Lin, R3 covariates").
  Defined at first use as: the PMRT randomization construction (redrawing the logged design) with a gradient-boosted
  / linear statistic, plain BY at q .05, no matched filter, no weighted-BY layer.
- Mandatory sentence: "Study 4 tests the randomization construction PMRT shares (redrawing the logged design); it
  does not evaluate the matched-filter statistic or the weighted-BY layer of the E6-P test."
- Families (define once): design-based tests (p-value from redrawing the logged design: PMRT variants);
  design-adjusted tests (standard CI test given the logged design variables as covariates: partial correlation,
  Granger, RCoT); design-blind methods (ignore the logging design: correlation, PC, NOTEARS, SHAP-DAG, two-tower,
  CDL, MSCR, plain partial correlation / Granger / RCoT).
- Never in prose: pmrt_nl_eq, eq, native, eq_min, R-numbers, "Study A", "Exp B".

## D2 Structure
- Studies 1-3 (E6-P) unchanged = main result.
- "Study 4: Validity across controlled diagnostic environments (E1-E5)": short main section (~1 page), one verdict
  table (C1, C2a, C2b, C3); full grids, power (V2), kappa sweep, R4 rows -> Appendix A. E1-E5 = controlled diagnostic
  environments, each isolating one mechanism; none models a deployed RAN.
- "Bridging analysis: the Study 4 methods on E6-P logs" (Exp B): first sentence "post hoc, descriptive, no verdict";
  design-adjusted methods behave alike; correlation and plain Granger reject 0.40-0.59 of the placebo column; frozen
  PMRT row = reference, not competitor; slice CIs from 2-10 clusters (not precision). Tables/cost -> Appendix B.
- Report PMRT-GBM's lower E6 recall openly: "Validity comes from the design; power depends on the statistic. The
  Study 4 primary statistic was chosen on E1-E5 development data, not on E6-P."

## D3 Rescoped claim (exact; matches REPORT s.1 / PROTOCOL_A s.10)
"Tests that ignore the randomization design are invalid on the R2 (setpoint + dither) design tested; using the design
restores validity (design-based inference and design-covariate adjustment), although design-adjusted RCoT is invalid
in R2 as already in R1 (not restored by design covariates). PMRT-GBM stays valid under the logged confounded policy
(R3) by construction; so does design-adjusted partial correlation, so the property is not specific to PMRT."
Required disclosures: PMRT-Lin NOT SUPPORTED under R3 (3 INVALID cells vs F_max 2; pooled rates valid); PMRT-GBM 1
INVALID cell in R1+R2 (within F_max 3); MSCR = reported invalid arm; power descriptive only; EVAL records under
amendment A-1 (dispatcher-only change, commit 5a95186); C1 5/6 design-blind arms fail (4/5 without MSCR).

## D4 Final method set (Study 4)
PMRT-GBM (primary), PMRT-Lin, PMRT-Lin R3 (supplement); design-adjusted partial correlation, Granger, RCoT;
design-blind correlation, PC, NOTEARS, SHAP-DAG, two-tower, CDL (R1/R2, n <= 4000), MSCR (n <= 1000), plain partial
correlation (+ HAC variants), plain Granger, plain RCoT. Dropped before EVAL: pdcor, cmi_knn (R-48/R-49), MSCR eq_min.

## D5 CDL wording [DECIDE: TERMINOLOGY s.10 says avoid "CDL" outside the conference version]
"With a placebo-calibrated threshold, our port of the conference CDL method exceeded the nominal truth-null rate in
9 of 30 cells (E1, E2, E3; R1 and R2, n <= 4000). At its original fixed threshold it rarely declared any edge at
n <= 1000. It was not evaluated under logged confounded policies." Practitioner rule: do not use learned-discovery
(CDL) edge declarations as causal evidence on designed/dithered logs unless the same pipeline declares nothing on a
placebo column drawn from the logged design. Never: "CDL is invalid", "CDL fails on O-RAN data".

## D6 Do-NOT-use wording
"Study 4 validates the E6 PMRT / the matched filter"; "PMRT validity does not depend on the outcome model" or "admits
any statistic" as tested results; "PMRT is uniquely valid under confounding"; "more powerful", "outperforms";
"valid in RIC deployments" / "RIC-style" beyond the R2 design; "pre-registered" without commit hashes; MSCR as valid;
"exact".

## D7 Housekeeping
TERMINOLOGY s.2: "E4 (scaffolded only)" and "E1-E5 inclusion open decision" are out of date -> update to Study 4.
