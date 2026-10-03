# Brief: xm-pmrt (PMRT core for the E-series)
Read first: scratchpad/xmethod/CONTRACT.md, cdd_oran/xmethod/api.py, docs/benchmark/METHOD_NAMES.md,
cdd_oran/decision/pmrt.py, cdd_oran/decision/fdr_layer.py, cdd_oran/decision/crt_units_v2.py (design_regressor),
docs/ARCHITECTURE.md sec 3.
Deliver cdd_oran/xmethod/methods/pmrt_core.py implementing api.Method:
- the design-based conditional randomization test of PMRT: assignment score from the KNOWN design (Design.kind iid:
  centred action; dither: centred random part only; logged: propensity-centred; none: not scorable -> report
  "not applicable", never fall back to row permutation); predictable adjustment where the world has lagged state
  (ridge on pre-state, fitted without the outcome-time data); CRT p-values by redrawing ONLY the random part from its
  known distribution (B = 9999, RNG tag 7801); declarations via fdr_layer (BY at q .05; weighted / one-sided variants
  only with priors from independent data, else plain BY).
- The E6 matched filter (cells x time bins) has no analogue here: document exactly which PMRT components are kept,
  reduced or dropped. This text goes into the paper ("PMRT core").
- Fidelity: F4 level on null data (R1 and R2, incl. serial dependence in R2) and power on planted edges; plus an
  equivalence check: on an E6 cache sample (use scratchpad/e6_dev/disc_bench.py loaders), your core with the filter
  disabled must match pmrt.py's corresponding plain arm (state tolerance), or explain the difference.
- Until xm-harness lands, test on your own synthetic api.Dataset objects; then the orchestrator tells you to rebase.
Tests: tests/test_xmethod_pmrt.py.
