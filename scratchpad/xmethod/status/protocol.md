READY-TO-MERGE
# protocol status (2026-10-03)

Branch xm/protocol from feat/v2 b7666d3. Committed locally, never pushed. No EVAL seed claimed, generated or read;
SEED_REGISTRY untouched. No dependency added.

## HAND-BACK
Deliverables
- `docs/xmethod/PROTOCOL_A.md` (222 lines, DRAFT, "FROZEN: no"): question / rescoped claim C1-C3; worlds as
  diagnostic environments with the mechanism of each; regimes; kappa .25 + sweep {.125, .5} (R2, n 1000), kappa 0
  not in EVAL; methods table (version, family, authors' null, arms, fidelity incl. NOTEARS F3 19/20, two_tower =
  adaptation, rcot2 liberal null); arms R-17/18/19/25/28; declaration rules; metrics; grid / budget / seeds;
  primary vs secondary analyses; tables V1-V11 + figures A-E (not drawn); support rules; integrity; deviations;
  TBD register T1-T5, each with its fill rule.
- `scratchpad/xmethod/eval_analysis.py` (`xm-eval-analysis/1`): merged campaign records -> `eval_tables.json` +
  `EVAL_TABLES.md` (V1-V11, verdicts C1-C3, FINAL / PROVISIONAL label, exit 2 unless FINAL). Self-contained (only
  feat/v2 imports: api, runner keys, score, worlds.truth_for); `t1` subcommand applies rule T1 to a DEV aggregate
  (needs dev_power.py from xm/dev-runs). Deterministic: two runs byte-identical; record order does not matter (test).
- `scratchpad/xmethod/specs/eval/full.json`: draft campaign spec, 20 arms (10 classic/pmrt + 10 CI-test eq/native),
  tune = DEV tune seeds 3_000_000-019 (re-run at the freeze commit), measure seeds "TBD" / "TBD_HALF" (sweep). At
  S = 40 it expands to 105 760 units / 5 800 datasets (counted only).
- Tests `tests/test_xmethod_eval_analysis.py`: 12 tests (BY == statsmodels fdr_by; synthetic study: flags, C1-C3,
  tau from tune records only, untuned / infeasible cells not counted, CLI guards, determinism; pilot regression).
  Result: 10 passed + 2 skipped on feat/v2; 12/12 with campaign.py + dev_power.py from xm/dev-runs present
  (copied in temporarily, removed). All tests/test_xmethod_*.py pass. ruff clean.
- Pilot check: dev-runs pilot (1648 records) run end to end; on 4 cells the raw / declared counts, recall and FDP
  equal campaign.aggregate's agg.json (test). Plumbing output: `results/eval_analysis_pilot/` (1 measurement seed,
  so flags / verdicts are meaningless there; stated at the top of the file). BY re-check: 648/648 p-arm records agree.

Findings
- F1 pmrt_core counts not-applicable candidates (R4, P0) in its BY m (`notes['by_family_m']` 2 vs 1 tested):
  conservative, disclosed in PROTOCOL_A section 6; the re-check uses the adapter's m.
- F2 PROTOCOL_NOTES' candidate counts (E1 4/16 ...) count P_placebo among the nulls; E4 therefore has NO real-action
  null candidate (1 true P0->K0, 1 null = P_placebo->K0). C3 validity is read on P_placebo + P_placebo_conf only.
- F3 campaign.aggregate's `null_decl` / `plac_decl` keep not-testable candidates in the denominator (score's n_null),
  while its raw rates and the dev-runs note leave them out. eval_analysis leaves them out of both (PROTOCOL_A s. 8).
  Identical when nothing is not testable (kappa .25: none for citests; classic eq had some at kappa 0).
- F4 campaign `_seed_list` accepts DEV seeds only, so the EVAL spec cannot run as is (needs an EVAL mode; Q3).

## QUESTIONS
- Q1 T1 excludes E4 from the seed-count max: one true edge makes per-seed recall 0/1 (pilot: 177 seeds from E4
  alone). E4 is read for validity / sign, not recall. OK?
- Q2 Tune seeds re-run at the freeze commit inside the EVAL spec (about +50 % of the measure cost at S = 40), rather
  than reusing DEV tune records. Alternative: reuse when method / world / score code is byte-identical. Which?
- Q3 Who adds the campaign EVAL mode (seeds >= 3_100_000 only when spec["protocol_sha256"] == sha of the frozen
  PROTOCOL_A)? dev-runs owns campaign.py.
- Q4 S floor 40 / cap 100 (compute: classic arms ~1.5 CPU-h per seed over the full grid; CI tests unknown until
  their DEV run). Confirm or set other bounds.
- Q5 For the user's review: C1-C3 decision rules (">= half of R2 cells ABOVE", F_max = Binomial(N, .025) 95 %
  quantile, PARTIAL when only n 500 fails), and "any infeasible unit makes the cell infeasible".
