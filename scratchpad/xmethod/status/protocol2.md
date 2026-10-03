READY-TO-MERGE
# protocol2 status (2026-10-03): PROTOCOL_A revision for R-29..R-38 + review must-fixes

Branch xm/protocol2 (feat/v2 7f84b04, fast-forwarded to 40d20a2 for R-37/R-38/pcorr_hac). Local commit, not pushed.
No EVAL seed claimed or generated; SEED_REGISTRY untouched; no dependency added. status/protocol.md left as is.

## HAND-BACK
- `docs/xmethod/PROTOCOL_A.md` (249 lines, DRAFT v2): s.0 disclosure list (R-36: pilot C1 direction, DEV validity,
  R-14, Stage-0 R2 design favours PMRT, every post-review choice, estimand check PASS); R-36 wording; pcorr_hac +
  fixed-b rows (R-32/R-38), eq_min (R-33), R-37; conformal tau (R-29); three-way validity (R-30); simulated F_max
  table; C1 per arm, |D| >= 3, HAC-scoped wording; C2a / C2b + fallback wording (R-31); C3 for pmrt_eq and every
  eq arm; caps (R-35); same-rule V3; T1-T8 mechanical (C item 7: files + sha, arm set, host, pilot rule, T4 sources,
  rcot2 ruling, S cap T6, freeze contents T7, HAC choice T8); amendments in a separate file so the frozen sha holds.
- `scratchpad/xmethod/eval_analysis.py` (`xm-eval-analysis/2`): `fmax_simulate` (Gaussian copula of exact-.05
  tests: shared seeds, Brownian in n, estimated lambda / candidate correlation, the analysis' own bootstrap
  replicated exactly; tested); three-way validity; verdicts C1 (per arm, set_D flag) / C2a / C2b / C3 (+ eq arms) with
  wordings and caps; V3 same rule (+ not-INVALID read); integrity (C item 4): expected keys + role screening,
  duplicates, candidate completeness, dirty must be False, run_mode stamps, protocol frozen + sha, pkgs vs pkgs_lock,
  one platform per dataset, amendment commits + persistent errors (s.11 / s.14), T3 cells shown; `t1` on DEV
  records with this module's definitions (one definition, C item 2h; several DEV specs; MDG at S).
- `scratchpad/xmethod/fmax_calib.py` + `results/fmax_sim/{dependence,fmax_table}.json`: dependence on DEV seeds
  3_000_100-199 (pcorr stand-in; candidate |r| <= .12, E4 lambda .69-.95; Brownian check .68/.51/.34 vs .71/.50/.35).
- `specs/eval/full.json`: 29 arms (+ pcorr_hac, pcorr_hac_fb, 7 eq_min), analysis block, native_partner, set_D
  (T8: pcorr_hac_fb `inference: fixed_b` true, pcorr_hac false), E4 R3/R4 at TBD_E4 (S_E4 300) for C3 readers only, protocol_sha256 / pkgs_lock TBD, dependence sha set.
- R-39 applied: power gate = not INVALID (V2, V3; VALID-only = `V3_paired_valid_only`); S_E4 only for pmrt_eq,
  pmrt_r3, eq arms; T9 S_E4 rule computed (`fmax_calib.py se4` -> `results/fmax_sim/s_e4_rule.json`): P(nominal
  pmrt_eq passes C3) .82 / .93 / .95 / .93 at 200 / 300 / 400 / 600 -> S_E4 = 300 (EVAL block 3_100_000-3_100_299).
- Tests `tests/test_xmethod_eval_analysis.py`: 20 pass + 1 skip (needs campaign.py; passes with xm/dev-runs
  40648f7 copied in, removed). Full xmethod suite once (before R-39): test_xmethod_citests.py errors at collection
  (tigramite not installed in this env; not my file); not re-run (memory). ruff clean. Pilot plumbing output regenerated
  (`results/eval_analysis_pilot/`, 1648/1648 records screened in; meaningless verdicts, stated).

## Findings
- F5 Simulation (fmax_table.json): chance INVALID per cell at an exact level .02-.026 (not .05); F_max 3 (S 40-80) /
  4 (S 90-100) on the 50-cell C2 set; an exactly nominal pmrt_eq passes C2a w.p. .91-.94, C3 (E4 R3, S 200) only .82.
- F6 At S 40 an exactly nominal cell is VALID w.p. .25-.45 (E4 R3 at S 200: .21) -> R-39 not-INVALID power gate.
- F7 Spec size after R-39: S 40 -> 241 160 units / 16 200 datasets (54 480 tune); S 100 -> 377 180 / 19 500.
- F8 Tau arms read validity on truth-null (+ conf) declarations only (P_placebo is their tuning column): E4 R1/R2
  tau cells have no read ("NO_READ"), so C3 for pc_eq uses P_placebo_conf only.

## Coordination
- calib: eval_analysis calls the shared `score.placebo_tau(results, alpha=.05)` (xm/hac2 signature) when it has
  `alpha`; until xm/hac2 is merged, `_conformal_tau_fallback` (same rule, M = 0 -> +inf; test checks delegation).
  tau = +inf -> nothing declared, reported as `tau_is_pos_inf` (cell counted); untuned = no tune records only.
- dev-runs: records must stamp `run_mode.spec_sha256` (only protocol_sha256 today); guard on the current file sha
  is kept by never editing PROTOCOL_A after freeze (amendments file).
- citests: `*_eq_min` arms need config arm "eq_min" in the citests adapters (ARMS = eq, native today).
- hac2: pcorr_hac_fb config `{'inference': 'fixed_b'}`; T8 filled from calib's F4 (fixed-b .0405 vs t .0535).

## QUESTIONS (all ANSWERED; none open)
Q1-Q5 (status/protocol.md) by R-29..R-36: Q1 yes + S_E4; Q2 re-run DEV tune seeds at the freeze commit; Q3 dev-runs
builds EVAL mode; Q4 40 / 100, cap revisited after CI pilot (T6); Q5 -> R-30/31/35. Q6-Q10 by R-39 (below).
- Q6 power gate -> ANSWERED R-39: not INVALID, VALID-only as sensitivity (implemented).
- Q7 n-500 PARTIAL clause dropped -> ANSWERED R-39: OK.
- Q8 S_E4 for C3 readers only -> ANSWERED R-39: yes (implemented in the spec).
- Q9 90 % coverage rule, C1 pooled-R2 leg, INVALID precedence -> ANSWERED R-39: OK.
- Q10 S_E4 -> ANSWERED R-39: smallest of {200, 300, 400, 600} with P(C3 pass) >= .90 -> T9 = 300 (computed).
