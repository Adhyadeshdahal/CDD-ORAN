# Status: xm-covariates  (updated 2026-10-02) -- READY-TO-MERGE
Branch xm/covariates from feat/v2 a25b0a4 (feat/v2 is now f1ef806: CONTRACT / notes only, no overlap with my files;
I did not merge it). Old branch xm/xm-harness clean at 398f4a3. Commit: 80f9cae (+ this status).

## HAND-BACK
Deliverables
- `cdd_oran/xmethod/covariates.py`: `design_covariates(data, include_setpoints=True, include_lagged_actions=True,
  *, lags=2, focal=None, concurrent="designed", include_kpi_lag=True, include_context=True)` -> (names, matrix
  [n, c] in ROW order, row_mask [n]); `concurrent_indices`, `info_order`. Default = pmrt_core's base set Z_eq
  (lagged KPIs, context, setpoints of every dither design incl. focal and placebo, every action at t-1 / t-2);
  `focal=` appends the concurrent columns (pmrt rule "designed", or "all" = literal R-3). Both flags False = R-3 set.
  Names: `lag_kpi:<K>`, `ctx:<name>` (meta["context_names"], else c<j>), `sp:<P>`, `<P>@t-1`, `<P>@t-2`,
  `gap@t-<L>`, `<name>:missing`, `concurrent:<P>`. Encoding = pmrt's (all-NaN col dropped; partial NaN 0-filled +
  indicator; lags 0-filled when not a true previous step; gap indicator only for gaps after the first rows).
  row_mask = every lag is a true previous step (first 2 rows and post-gap rows False).
- `pmrt_core.py`: builds Xc via the helper (old `covariates`, `concurrent_columns`, `info_order` kept as thin
  wrappers); new config `covariates="eq"` (default) | `"r3"` (R-19: no setpoints, no lagged actions; concurrent
  actions still per `concurrent_actions`); notes["covariates"] = {set, base names}. Version string unchanged.
- F7 finding 7: `meta["roles"]` removed from every Dataset; roles live in harness-private `worlds.roles_for(world)`;
  generator "xm-worlds/3" (arrays unchanged, meta changed -> dataset_sha256 differs from /2 records).
- R-23 (`score.py`): candidates listed in notes["not_testable_edges"] (also notes["not_testable"], R-22 text; list
  of "src->tgt" / pairs, or dict keyed by them) count as NOT declared whatever their flag; every block reports
  n_not_testable_true / n_not_testable_null; record adds placebo_not_testable, n_not_testable_overridden (listed
  but flagged declared = adapter bug), not_testable_edges; malformed listing -> ValueError. Merge summary adds
  n_not_testable_true_total / _null_total per cell.
- `docs/xmethod/PMRT_CORE.md` (2 sentences: shared builder, r3 option).

Bit-identity proof (production config B 9999, Besag-Clifford h 20): `scratchpad/xmethod/cov_equiv.py` runs the
pre-refactor module (git a25b0a4) and the new one on 16 datasets (E1 R1, E2 R2, E3 R2, E4 R1/R2/R3/R4 lam 1.5, E5 R2;
n 1000; each also PERTURBED: shuffled rows, time gaps, 2% NaN lagged KPIs): every edge (score, p, sign, declared),
notes z / p_plus / p_minus / adjust / not_applicable and the covariate matrix identical (repr-exact) in 16/16
(`scratchpad/xmethod/results/cov_equiv.jsonl`, "ALL IDENTICAL: True", rerun on the final code).
Tests: `tests/test_xmethod_covariates.py` (vendored legacy builder: helper == legacy on all 12 cells x 3 variants x
lags 0-3; names / blocks / mask / gaps; R-3 flags; concurrent rules; pmrt run == run fed the legacy builder;
r3 option; no roles in meta, no method / covariates source mentions "roles"), R-23 tests in
`tests/test_xmethod_harness.py`. All xmethod tests: 192 passed (covariates, pmrt, harness, classic); ruff clean.
No dependencies added.

## QUESTIONS (non-blocking; defaults implemented, easy to change)
Q1 Default includes lagged KPIs + context ("EXACTLY the covariates pmrt_core uses"). So an equal-information adapter
   should take its whole conditioning set from the helper (`focal=ai`), not add its own R-3 columns on top
   (duplicates). OK?
Q2 Names you did not specify: lagged KPIs `lag_kpi:<K>`, concurrent `concurrent:<P>`, indicators `<name>:missing`.
   Rename (e.g. `<K>@t`, `<P>@t`) before xm-citests depends on them?
Q3 Concurrent rule for the eq arm of CI tests: pmrt uses "designed" (drops kind-"none" columns: R4 P0 and
   P_placebo_conf); literal R-3 = "all". Differs only in R4. Which should citests pass?
Q4 pmrt_core's own notes["not_applicable"] (R4 actions, lagged-KPI sources: no design) are NOT counted as
   not-testable by the scorer (they were already not declared; R-22 is about exact-fit cells). Confirm.
