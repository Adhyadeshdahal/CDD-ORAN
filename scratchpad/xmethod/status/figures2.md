# figures2 status (2026-10-05): READY-TO-MERGE: Study 4 hybrid layout: F3 by environment, verdict grid, E4 example, E1 page (STOP for review)

Data read-only from the main checkout D:/academia/major-project/CDD-ORAN at feat/v2 823b3b8 (results/eval/report.json,
csv/validity_cells.csv, csv/recall_vs_n.csv; graph from cdd_oran/envs/v2/e1.py + xmethod/worlds/generate.py, copied).
Zero compute. No .tex manuscript edit, no journal commit. Only E1 of the per-environment pages is built (as asked).

## Files (figures/ is not a git repo; journal outputs untracked)
- figures/src/cddfig/figs/_study4.py (edited): PAPER names for every Study 4 arm, C1 set D, cells(), world_null_rate().
- figures/src/cddfig/figs/fig_study4_validity.py (rewritten): F3 faceted E1-E5 + pooled.
- figures/src/cddfig/figs/tab_study4_verdicts.py (new): verdict grid; `uv run python -m cddfig.figs.tab_study4_verdicts`.
- figures/src/cddfig/figs/fig_study4_e4_example.py (new): E4 worked example.
- figures/src/cddfig/figs/_env_page.py (new, Appendix A template) + fig_study4_env_e1.py (new, E1 page figure).
- figures/src/cddfig/figs/tab_study4_env.py (new): per-environment cell table; `uv run python -m cddfig.figs.tab_study4_env E1`.
- Journal figures/: fig_study4_validity.pdf (updated), fig_study4_e4_example.pdf, fig_study4_env_e1.pdf (new).
- Journal tables/ (new dir): tab_study4_verdicts.{tex,csv}, tab_study4_env_e1.{tex,csv} (booktabs tabular only, to
  `\input` inside a table float; both compile at 31 pc with no overfull box: verdicts at \small, E1 at \footnotesize).
- figures/preview/*.png for the three figures. Checks: `uv run cddfig build ...`; `uv run pytest -q` 14 passed.

## Captions (one line each)
- fig_study4_validity: Raw null-edge rejection rate (p <= 0.05) per method and regime in each environment E1-E5
  (mean of the environment's five n cells; no CI) and pooled with 95 % cluster-bootstrap CIs (verdict table); dashed =
  nominal 0.05; E4 R1 / R2 = placebo edge, R3 = confounded null edge (E4 only); open markers = below 0.01.
- tab_study4_verdicts: Claims C1-C3 by environment: INVALID / VALID / counted cells (rest INCONCLUSIVE), kappa 0.25, all
  n; C1 = design-blind set D in R2, C2a = PMRT-GBM R1 + R2, C2b = design-adjusted partial correlation + Granger R1 + R2,
  C3 = PMRT-GBM in E4 R3 (all lambda; R3 exists in E4 only); All = report totals and verdicts.
- fig_study4_e4_example: E4 under R1, R2, R3 (lambda 1): null-edge rejection rate (top; placebo edge in R1 / R2,
  confounded null edge in R3) and recall (bottom; cross = INVALID cell, no recall) vs n for PMRT-GBM, design-adjusted
  partial correlation and two design-blind methods (gray); 40 seeds per cell, 95 % cluster-bootstrap CIs.
- fig_study4_env_e1 (+ tab_study4_env_e1): E1 page: (a) ground-truth graph (actions P0-P3 and placebo Pl, KPIs K0-K3;
  every edge acts one step later; K0 -> K2, K1 -> K3 are lagged KPI edges), (b) R1 / R2 logging designs (schematic),
  (c, d) truth-null rate and (e, f) recall vs n; table: every named method x regime x n, rate / recall with validity mark.

## Data caveats
- C1 does NOT hold uniformly by environment: in E4 only 2 of 22 design-blind R2 cells are INVALID (7 VALID), because
  E4 R1 / R2 have no truth-null edge and are judged on one placebo edge (40 seeds per cell, rate steps of 0.025). The
  E4 design-blind failure shows in R3 (correlation rejects the confounded null in every cell, rate 1.0). F3's E4 facet
  shows the same. C1 is SUPPORTED on the pooled rule (95 / 115 INVALID); say "every environment but E4 (placebo only)".
- Verdict-grid environment counts are re-derived from validity_cells.csv; their sums are asserted equal to the report's
  V4 counts (C1 95/115, C2a 1 INVALID 18 VALID / 50, C2b 0 / 21 / 60, C3 0 / 6 / 20) before writing. Granger is planned
  in E3 only (C2b E3 = 20 cells).
- F3 per-environment rates have no CI (the report has no per-environment pooled CI); the pooled column is the report's.
- E4 example: top row mixes edge kinds by panel (labeled); placebo rates are coarse (one edge per dataset).
- E1 page: panel b is a schematic (illustrative step heights, not data); the spec values (20 setpoints held n/20 rows,
  +/-25 % envelope, +/-10 % dither) are from generate.py. The E1 table mark is the cell's overall validity (truth-null
  and placebo reads), not the truth-null read alone; threshold-only methods use their declared truth-null rate.
- Page fit: E1 figure (4.5 in) + table (two 15-row blocks) is likely more than one page; trim to taste at review.

## Decisions needed
- Review the E1 page template before E2-E5 (graphs for E2-E5 still to be copied from their env classes).
- Journal `tables/` dir and the `python -m` table generators: BUILD_GUIDE s.2 has no table convention (add one, or a
  `cddfig tables` command?).
- Method slots as in F1 (0 PMRT-GBM, 1 PMRT-Lin, 2 design-adjusted partial correlation, 3 design-adjusted Granger;
  design-blind gray) and regime entity slots (R1 1, R2 2, R3 3, R1 + R2 4): confirm for DESIGN.md s.12.
