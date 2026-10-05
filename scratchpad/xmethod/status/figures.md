# figures status (2026-10-05): READY-TO-MERGE: Study 4 / bridging figures F1-F3 built (cddfig), tests pass

Data read-only from the main checkout D:/academia/major-project/CDD-ORAN at feat/v2 696c2da (via `cddfig.paths.cdd`).
Zero compute: no run, no new artifact; F2 streams the two merged.jsonl.gz files (~4 s). No .tex edit, no journal commit.

## New files (figures/ is not a git repo; journal PDFs untracked)
- figures/src/cddfig/figs/_study4.py: paper names, method slots, read-only loaders (recall csv, bridge tables, placebo p).
- figures/src/cddfig/figs/fig_study4_recall.py (F1), fig_study4_calibration.py (F2), fig_study4_validity.py (F3).
- xApp-Journal-Discover-Telecommunications/figures/fig_study4_{recall,calibration,validity}.pdf (full width 5.15 in).
- figures/preview/fig_study4_{recall,calibration,validity}.png (300 dpi review copies).
- Checks: `uv run cddfig build fig_study4_recall fig_study4_calibration fig_study4_validity`; `uv run pytest -q` 12 passed
  (the project's pattern: every registered figure must build at its width; no extra tests added).

## Captions (one line each; add the TERMINOLOGY s.6.1 mandatory sentence wherever PMRT-GBM / PMRT-Lin appear)
- F1 fig_study4_recall: Recall vs n in R2 (kappa 0.25) for PMRT-GBM, PMRT-Lin, design-adjusted partial correlation and
  design-adjusted Granger in E1-E5 (a-e; 40 seeds per cell; bands 95 % cluster-bootstrap CIs over seeds; cross = INVALID
  cell, recall not shown) and in the E6-P bridging analysis by slice length (f; post hoc, descriptive; BY declarations;
  whiskers cluster-bootstrap CIs over 10 / 5 / 2 slices, none for the pooled 600 episodes; dashed = frozen PMRT reference).
- F2 fig_study4_calibration: Empirical CDF of placebo-edge p-values on log-log axes (dashed diagonal = uniform; t = 0.05
  marked): (a) Study 4 R2, E1-E5 and all n pooled, measure datasets (4000 p-values per method; 1000 for the two Granger
  tests, planned in E3 only); (b) E6-P bridging analysis, the ten 60-episode slices (600 per method). Design-blind
  methods (gray, dashed): correlation, plain partial correlation, plain Granger.
- F3 fig_study4_validity: Pooled truth-null rejection rate (raw p <= 0.05) per method and regime in Study 4 (kappa 0.25)
  with 95 % cluster-bootstrap CIs, nominal 0.05 dashed; PMRT arms R1 + R2 pooled (C2a) and R3 confounded truth-null (C3);
  design-adjusted tests R1 / R2 (+ R3 where C3 has a read); design-blind methods R1 / R2 (C1). Source report.json V4_verdicts.

## Data caveats
- Bridge slice CIs come from 2-10 clusters (10 / 5 / 2 slices for 60 / 120 / 300 episodes); narrow or zero-width,
  not a precision statement (EXP_B s.9). The pooled 600-episode point has no CI.
- Bridge: design-adjusted partial correlation and design-adjusted Granger have identical recall at every slice
  (EXP_B s.9 "behave alike"); F1 f dodges them on x so both markers show.
- F1 plots INCONCLUSIVE cells as recall (only INVALID is excluded): one INVALID cell, PMRT-GBM E1 R2 n 24000.
  Design-adjusted Granger is planned only in E3 R2.
- F2 uses raw placebo p (not BY declarations): EXP_B s.9's 0.59 / 0.40 (correlation / plain Granger at 60 eps) are
  BY declaration rates, not read off this figure. Cross-check: Study 4 share at 0.05 = report.json C1 plac_raw
  (correlation 3006 / 4000, plain partial correlation 1402 / 4000). On E6-P, plain partial correlation is calibrated
  (0.037 at 0.05) and sits under the adjusted curves; ECDF floor 1 / 600 in (b).
- F3: the verdict table has no per-regime split for the PMRT arms (only R1 + R2 pooled), so they show one R1 + R2
  marker; design-adjusted Granger has no R3 read (C3 NOT EVALUABLE); omitted: pcorr_hac (C1 descriptive), mscr_eq /
  pc_eq (not design-adjusted family members, TERMINOLOGY s.6.5).

## Decisions needed (DESIGN.md s.12 / TERMINOLOGY s.11)
- Method slots used here: 0 PMRT-GBM, 1 PMRT-Lin, 2 design-adjusted partial correlation, 3 design-adjusted Granger;
  design-blind methods as neutral gray dashed context (not baselines). The frozen PMRT is an ink reference line only.
  TERMINOLOGY s.11 proposes slot 0 = PMRT (E6-P figures): confirm the Study 4 mapping.
- Regime entity slots: R1 1, R2 2, R3 3, R1 + R2 pooled 4 (F3).
- Label "Plain partial correlation, HAC (fixed-b)" for pcorr_hac_fb: TERMINOLOGY lists the HAC variants without a
  paper name.
