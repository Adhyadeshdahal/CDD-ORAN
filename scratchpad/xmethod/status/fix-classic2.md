READY-TO-MERGE

# fix-classic2 (branch xm/fix-classic2 from feat/v2 5c32e7b, 2026-10-02)

Implements audit `scratchpad/xmethod/audit/classic_v2.md` "Fixes (exact)" 1-4. Fix 5 = orchestrator (already on feat/v2).

1. **A (MEDIUM), option 1.** `ClassicBase.tune(dev, truth_free=True, config=None)`: `base = default_config();
   base.update(config or {})` (`_classic_common.py`). `scripts/xm_classic_integration.py`: new `--arm {eq,native}`
   (default = method's default arm); passed as `tune(dev, config={"arm": arm})` and also put in the spec's `default`
   config, so the run uses the arm the tau was tuned on.
   Test `test_tune_respects_arm`: pc on SYN R2 (n 300, seeds 3_000_020-022): eq tau .879 != native tau .864; default ==
   eq; native tau == placebo_tau of native runs; notears (native only) gives the same tau for either requested arm.
2. **B (LOW).** granger, both arms: status "degenerate" -> listed as not testable, reason `COLLINEAR = "source collinear
   with Z"`; `not_testable_notes(edges, collinear=None)` puts them in `not_testable_edges` (+ `n_not_testable`) and in
   `notes['not_testable_collinear']`; `notes['not_testable_reason']` = the reasons present joined by "; " (see Q1). Test `test_granger_collinear_source_is_not_testable[eq|native]` (lag Y2 := lag Y1): every NaN edge is
   listed.
3. **C (LOW, doc).** FIDELITY_CLASSIC Arms, pc bullet: in R4 the eq node set is wider than Z_eq (non-designed concurrent
   actions incl. P_placebo_conf); R1-R3 equal. Also one line in the Not-testable bullet for fix 2.
4. **D (LOW, dev).** Documented, no default-groups change (keeps plain `uv sync` light): FIDELITY_CLASSIC header and
   the test module docstring say `uv sync --group baselines` / `uv run --group baselines pytest ...`.

5. **Q1 answer, option (a).** pc and granger (shared `not_testable_notes`) put the reason string in
   `notes['not_testable_reason']` and the edges only in `notes['not_testable_edges']` (`score.py` format); no
   `notes['not_testable']` key from the classic adapters any more (citests' dict form under `not_testable` untouched).
   Docstrings (`_classic_common`, pc, granger) and FIDELITY_CLASSIC Not-testable bullet updated. Test
   `test_score_accepts_not_testable_notes[granger|pc]`: `score()` on the SYN repro (noise 0, R2) returns the listed
   edges, every placebo candidate not testable, none of them declared, 0 overridden.

Tests: `uv sync --group baselines`; `uv run --group baselines pytest tests/test_xmethod_{classic,covariates,harness,noise,pmrt}.py`
-> 238 passed (216 s). ruff clean on the touched files.

## Questions

- Q1 ANSWERED (orchestrator: option (a), by fix-classic2 on this branch; done, item 5). (found while testing, existing on feat/v2, NOT fixed: out of scope). `score.not_testable_edges` rejects a str
  `notes['not_testable']` ("must be a list or dict of edges"), but the classic adapters (pc, granger) set it to the
  reason string. So `score()` raises ValueError on any pc / granger result that has a not-testable edge (repro: granger
  on `SYN.make(300, 3_000_012, b=1.0, noise=0.0, regime="R2")`). Not hit at kappa .25 (0 not-testable), but any
  kappa-0 run crashes in scoring. Fix options: (a) adapters rename the reason key (e.g. `not_testable_reason`) and keep
  only `not_testable_edges`; (b) score skips a str `not_testable`. Which one, and by whom?
