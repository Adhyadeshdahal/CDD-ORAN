READY-TO-MERGE
# protocol3 status (2026-10-03): ruling R-40 (mscr_eq out of C2b; pdcor a dependence test)

Branch xm/protocol3 from feat/v2 e2a75b2. Local commit, not pushed. No EVAL seed claimed or generated;
SEED_REGISTRY untouched; no dependency added. status/protocol.md and protocol2.md left as is.

## HAND-BACK
- `specs/eval/full.json`: `mscr_eq` gets `"c2b": false` and `"label": "single-conditioner max statistic; cannot
  condition on the joint design set"`. `pdcor_eq` / `pdcor_native` / `pdcor_eq_min` get `"label": "dependence test
  (pdCor = 0 is not conditional independence); not a CI test"`. Arms, blocks and unit count are unchanged.
- `scratchpad/xmethod/eval_analysis.py`:
  - C2b leaves out every eq arm whose `c2b` is not `true`. Its same-rule R1 + R2 result is reported in
    `V4.C2b.excluded[arm]` with its label and has no effect on the verdict or wording.
  - `arm_label()` reads the spec label. `V4.arm_labels` holds all labels. The markdown prints a label legend and
    tags the C1, C2b and C3 lines.
  - C3 for every eq arm is unchanged (mscr_eq still reported there, tagged).
- `docs/xmethod/PROTOCOL_A.md` (DRAFT v3, R-1..R-40; 255 lines):
  - s.0: adds (f), the audit smoke, and R-40 to the list of post-review choices.
  - Methods table: the mscr row notes the single-conditioner eq arm (not in C2b). pdcor's family is "dependence
    test, not a CI test".
  - C1: D reads "native CI / dependence tests admitted by T4".
  - C2b: "every eq p arm except mscr_eq", plus the R-40 sentence and the labels.
  - T4: the audit's verdicts are mapped (see F9).
- Tests `tests/test_xmethod_eval_analysis.py`: 23 pass + 1 skip (the campaign.expand test, which needs
  xm/dev-runs). New tests: `test_c2b_exclusion_and_labels_r40` (synthetic, incl. the markdown) and
  `test_eval_spec_r40` (EVAL spec flags). Only this file was run (memory note). ruff clean.
- Pilot plumbing output (`results/eval_analysis_pilot/`) is not regenerated. The DEV pilot spec has no R-40 fields,
  so only two empty keys (`C2b.excluded`, `arm_labels`) would be added.

## Q11 applied (second commit)
- Spec: `pdcor_native` `"set_D": false` (out of D); `mscr_eq_min` gets the mscr_eq single-conditioner label.
  D in the EVAL spec = cmi_knn_native, corr, granger_native, mscr_native, pcorr_hac_fb, pcorr_native, rcot2_native.
- eval_analysis: native p arms outside D (pdcor_native, pcorr_hac) get the C1 legs in `V4.C1.descriptive`, printed
  "(not in D, descriptive)", no effect. pdcor_eq drops out of C2b (its partner is not in D).
- PROTOCOL_A (259 lines): C1 text (D = `set_D: true`; pdcor_native out, reason, descriptive report); C2b notes
  mscr_eq_min label and pdcor_eq; T4 keeps mscr admissible (faithful port, R-40); s.0 and pdcor table row updated.
- Tests: 24 pass + 1 skip (new: out-of-D arm descriptive + its eq partner out of C2b; spec D set). ruff clean.

## Findings
- F9 T4 vocabulary: the audit table's OK / DOC FIX / FIX NEEDED is mapped in T4 (OK, or a doc / protocol FIX whose
  fix is merged); mscr admissible via R-40 (Q11).
- F10 pdcor_eq is no longer in C2b, a side effect of its partner leaving D; it stays in V1 / V3 / C3 reports, labelled.

## Coordination
- citests2: the audit fixes 1-6 (incl. the pdcor docstring and FIDELITY_CITESTS F6) are theirs; nothing here
  depends on them.
- dev-runs: campaign ignores the new arm keys (`c2b`, `label`), as it does `analysis` / `set_D`. The spec sha changes,
  so EVAL records must stamp the new spec sha.

## QUESTIONS (all ANSWERED; none open)
- Q11 mscr under T4, pdcor_native in D -> ANSWERED: mscr admissible (R-40); pdcor_native out of D (descriptive);
  mscr_eq_min labelled (implemented).
