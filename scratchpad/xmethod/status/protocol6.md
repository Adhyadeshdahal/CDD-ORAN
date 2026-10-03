READY-TO-MERGE
# protocol6 status (2026-10-03): R-42 part 3 (R4 NA for PMRT, like-for-like headline V0, pmrt_nl_eq placeholder)

Branch xm/protocol6 from feat/v2 93a4c9e. Local commit, not pushed. No EVAL seed claimed or generated;
SEED_REGISTRY untouched; no dependency added. The spec sha changes.

## HAND-BACK
- (a) R4: `power_not_applicable(arm, regime)` = PMRT arm and regime in `NO_DESIGN_REGIMES` ("R4"). A test ties it to
  the generator: R4 real actions are kind "none" (R1-R3 are not). Such cells keep recall / sign = None, never 0, and
  have no per-seed recall, so they drop out of V3. V2, V6 and V0 print "NA". Placebo rates and validity are still
  read (P_placebo is i.i.d. in R4). FDP is kept (it counts only placebo declarations there).
- (b) V0 `V0_like_for_like`, the headline, printed first in EVAL_TABLES.md:
  - per cell (kappa .25) and arm: p arms twice, `raw_p` (p <= .05 per edge, no multiplicity) and `tau` (conformal
    tau, the existing secondary_tau); tau arms once (`tau`);
  - each row: mean recall, truth-null / placebo / confounded-placebo declaration rates with CI and validity;
  - raw_p validity uses all three rates; a tau row's placebo is its tuning column, reported only;
  - recall is shown for every counted cell next to its rates (Q16); untuned tau scorings are shown as "unt".
- (c) EVAL spec:
  - new arm `pmrt_nl_eq` (pmrt_core, covariates eq, config `statistic: "TBD-R-42"`, analysis primary);
  - top-level `focal: "pmrt_nl_eq"`; `pmrt_eq` is now secondary, labelled "linear statistic (pmrt-core-v1);
    secondary PMRT arm (R-42)";
  - pmrt_nl_eq added to the E4 R3 / R4 TBD_E4 block (C3 reader); `tbd.pmrt_nl_eq` explains the placeholder;
  - 30 arms. Units: S 40 = 257 360 / 16 200 datasets (56 480 tune); S 100 = 396 680 / 19 500 (the old 29-arm
    numbers reproduce with the same count).
- eval_analysis (`xm-eval-analysis/3`): `focal_arm(spec)` (spec `focal`, else pmrt_eq) drives C2a, C3, V3, T1
  (`t1 --focal`) and the arm order. V4 `pmrt_secondary` reports the C2a / C3 rules for the other PMRT arms
  (pmrt_eq, pmrt_r3), no effect, never in C3's "so do <arms>". DEV specs (no `focal`) behave as before.
- (d) PROTOCOL_A (DRAFT v3, R-1..R-42; 307 lines):
  - s.0 (h): the DEV power read, quoting R-42's numbers only, plus the R-42 choices;
  - methods row and s.5 arms (pmrt_nl_eq primary, pmrt_eq secondary);
  - s.6: raw p scoring; s.7: C3 readers and unit counts; s.8: the R4 NA rule;
  - s.9: HEADLINE V0; s.10: C2a / C3 for the PMRT arm, others reported; s.12: V0;
  - T1: pairs use pmrt_nl_eq (`--focal`, Q14); T9 wording; new T10: "statistic chosen by the R-42 rule" (rule copied
    from CONTRACT R-42), VALUE TBD.
- Tests `tests/test_xmethod_eval_analysis.py`: 32 pass + 1 skip (the campaign.expand test needs xm/dev-runs). Only
  this file was run. ruff clean. New tests: `test_r4_pmrt_power_not_applicable_r42`,
  `test_like_for_like_headline_r42`, `test_focal_pmrt_arm_from_spec_r42`, `test_eval_spec_pmrt_nl_placeholder_r42`,
  `test_r4_real_actions_have_no_design`.

## Findings
- F13 V0 rates are declaration rates (a not-applicable candidate counts as not declared, section 8). For raw_p they
  can therefore differ from the V1 raw rates, which drop not-applicable candidates. This only matters where an arm
  has not-applicable candidates.
- F14 The config key `statistic` is a placeholder. pmrt-nl decides the real key and value; T10 writes them, and
  until then the EVAL spec does not run (as before: seeds TBD).

## Coordination
- pmrt-nl: expose the winner as a pmrt_core config option. The protocol assumes the linear statistic stays the
  default (pmrt_eq unchanged).
- dev-runs: campaign ignores `focal`. EVAL records must stamp the new spec sha.

## QUESTIONS (all ANSWERED; second commit: protocol T1 / T10 / s.5 + spec tbd note, no code change)
- Q14 ANSWERED: T1 pairs use pmrt_nl_eq (`t1 --focal pmrt_nl_eq`). dev-runs runs the T10 winner on the T1 cells.
- Q15 ANSWERED: no candidate passes -> pmrt_nl_eq removed, pmrt_eq focal / primary. Written in T10, s.5, spec tbd.
- Q16 ANSWERED: V0 shows every recall with flags; V2 / V3 keep the not-INVALID gate (implemented as is).
