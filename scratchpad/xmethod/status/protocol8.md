READY-TO-MERGE
# protocol8 status (2026-10-03): R-51 (Q17 / Q18 answers into PROTOCOL_A)

Branch xm/protocol8 from feat/v2 95663ec. One local commit, not pushed. PROTOCOL_A only (+ this file): spec,
eval_analysis and tests unchanged, so the spec sha is unchanged. No EVAL seed, no compute, no dependency.

## HAND-BACK
- Header: branch xm/protocol8, feat/v2 95663ec, rulings R-1..R-51.
- T1 (Q17, option 1): dev-runs runs cdl on the full DEV grid before the freeze; that run is in the T1 input, and
  T4 (iii) applies to cdl with no waiver.
- T3 (Q18): DEV costs from any platform count, in Kaggle reference seconds:
  - a unit run on another host type (Lightning, Colab) is converted as cost x f;
  - f = sum of Kaggle cost / sum of host cost over the same calibration units, per (arm, host type);
  - calibration units: DEV seeds 3_000_000-002 in the arm's 3 costliest (world, regime) at its largest measured n
    (the existing pilot units);
  - the units, both costs and f go into the freeze commit;
  - the missing-cost pilot may run on any host with its f;
  - GPU arms: the same rule on wall-s, referenced to a Kaggle T4.
- Tests: `tests/test_xmethod_eval_analysis.py` re-run (it reads PROTOCOL_A): 34 pass + 1 skip (campaign).

## Findings
- F18 Choices I made where R-51 gives no detail (confirm or overrule):
  - f is per (arm, host type), since methods scale differently across hosts;
  - f is a ratio of summed costs, not a median of per-unit ratios;
  - the calibration units are the existing T3 pilot units.

## QUESTIONS
- none open (Q17 / Q18 answered by R-51).
