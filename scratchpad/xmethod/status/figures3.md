# figures3 status (2026-10-05): READY-TO-MERGE: figure-review fixes at the root (shared layout helpers + layout checks)

Review: feat/v2 ae5e346 scratchpad/xmethod/reviews/figure_review_2026-10-05.md (5 figures, 19 findings). Every finding
is classified by root cause, each cause is fixed once in shared code, and a test class now catches it on every registered
figure. Zero compute. No DESIGN.md, manuscript .tex or journal-repo commit. E2-E5 pages are not built (as asked). Data
sources and numbers are unchanged (same report.json / csv / merged.jsonl.gz reads as figures2).

## Root causes -> shared fixes (figures/src/cddfig/layout.py, new) -> checks (figures/src/cddfig/checks.py, new)
| Root cause | Review findings (fig: item) | Shared fix | Check |
|---|---|---|---|
| Ad-hoc fig.legend placement (upper left / right / lower center) | calib: unbalanced legend; recall: asymmetric legend; env_e1: legend below e/f; validity: legend top-right | `fig_legend`: one top row centred over all panels, auto-wrapped into balanced rows after a layout pass (or a right column); never below / inside | legend placement, on canvas |
| Letter and tag at fixed offsets on one baseline | calib: letter/tag 14 pt crowding; recall: tag crowding | `panel_header` + `finalize`: letter over the outer label column (y label / tick / row labels), >= 14 pt from the tag; tag 7.5 pt at the axes edge | text overlap |
| No INVALID convention: crosses at y = 1.13 with clip_on=False, hand-placed notes | e4: note above panel d (crosses in e, f), crosses squeezed between rows; env_e1: unlabeled hero cross | `recall_axis` (reserved strip inside the axes above the trimmed 0-1 scale) + `invalid_marks` (crosses in the strip) + automatic `× INVALID cell (no recall)` legend entry | text intrusion / overlap |
| Equal values drawn at the same spot, hero on top | e4 f, recall c: baselines hidden under the hero at recall 1 | `dodge`: fixed 3.4 pt display offset per series (line, band, error bars, crosses); staggered markevery phases | coincident series |
| NaN-masked ECDF | calib b: isolated diamond | `ecdf_line`: starts at the first positive p (1/N), connected, global marker grid with per-series phase | lone markers |
| Shared axes left half-hidden (set_xlabel("") only) | e4: bare ticks on a-c; calib b: repeated y ticks | `tidy_grid`: non-bottom equal x axes lose ticks / labels / spine; non-left equal y axes lose tick marks / labels | tick hygiene |
| Artificial clipping floor | env_e1 c/d: clip at 0.011 + "≤0.01" tick | `rate_axis`: symlog, linear below 0.01, 0 drawn at 0; ticks 0 / .01 / .05 / .2 / 1 (E4 example, E1 page, F3) | (code rule) |
| Labels offset far from their anchor | recall f: "Frozen PMRT" 22 pt above its line | `label_line`: anchored, <= 6 pt offset | anchored labels (> 8 pt fails) |
| Too many facets for 5.15 in | validity: 6 facets < 0.7 in, cramped log ticks | redesign (below) | facet width (>= 1.0 in) |
| Headers as bold y-tick labels | validity: category headers in the name margin | `banner`: full-width tinted row, bold label | text overlap |
| Close CIs on one row | validity pooled: whisker collisions | redesign: one row per method x regime at 1 row pitch | coincident series |
| Text pushed out of its panel | env_e1: graph legend / design note against c/d headers | E1 page in two subfigures; design text kept inside its 0-100 box; no `set_in_layout(False)` text | text intrusion, on canvas |

## New checks (tests/test_layout.py; `uv run pytest -q`: 131 passed, was 14)
- Nine checks on every registered figure: text overlap (letters, tags, ticks, labels, legends; renderer extents), text
  intrusion into another panel, legend placement, anchored labels, lone markers (incl. log masking), coincident series,
  tick hygiene, facet width, on canvas. That is 12 figures x 9 checks = 108 tests.
- Nine negative tests: each check must flag a deliberately broken figure.
- Audit before -> after (problems found by the checks): calibration 7 -> 0, e4_example 37 -> 0, env_e1 38 -> 0,
  recall 63 -> 0, validity 69 -> 0. Specimens: entities 4 -> 0, heatmaps 10 -> 0, methods 2 -> 0 (specimens now use
  the same helpers).

## Per-figure changes (all rebuilt: journal figures/*.pdf + preview/*.png)
- fig_study4_validity: restructured into (a) pooled rate with 95 % CI per method x regime (forest; rows labelled
  method then regime, so no legend) and (b) an E1-E5 value grid sharing the rows (rate printed in each cell, harbor_seq
  square-root shade, "–" = no cell; R3 rows E4 only; E4 column = placebo in R1 / R2). Category banners, symlog rate
  axis. Panels 1.64 / 1.43 in wide (was 6 x 0.54 in); the row-label margin takes the rest.
- fig_study4_e4_example: top-row y = symlog rate axis ("Null-edge rate"); INVALID strip and legend entry; dodge; x axis
  only on the bottom row; inner y labels hidden; legend top-centred.
- fig_study4_env_e1 (template `_env_page`): page = two subfigures (graph + design | 2 x 2 data with its own top legend);
  truth-null axis symlog (no 0.011 clip); INVALID strip (the n = 24k PMRT-GBM R2 cell is now explained by the legend);
  design text inside its panel; tidy grid.
- fig_study4_recall: INVALID strip + legend entry (replaces the "INVALID" side notes); dodge; Frozen PMRT label
  anchored at the line's first point; tidy grid (x shown where columns differ: c keeps n, f has episodes).
- fig_study4_calibration: ECDF via `ecdf_line` (no isolated markers; the lines start at each series' smallest p);
  panel b y tick labels removed; legend top-centred in balanced rows.
- specimen_entities / _heatmaps / _methods: moved to the helpers (one shared legend, panel_header, staggered markers).

## Caption notes the figures now need
- State "markers dodged horizontally by a few points for legibility" (recall, E4 example, E1 page).
- Rate axes: "symmetric-log scale, linear below 0.01" (E4 example, E1 page, F3 a).
- F3 (b): "mean of the environment's five n cells (E4 R3: all four lam), no CI; shade = rate on a square-root
  scale; – = no cell".

## DESIGN_PROPOSALS
- D:/academia/major-project/DESIGN_PROPOSALS.md (new): P1-P12 (panel header, multi-panel legend, shared-axis hygiene,
  INVALID strip, dodge, ECDF, symlog rate axis, anchored labels, banners, min panel width, on-canvas, compact value
  grid). All are applied now; DESIGN.md is unchanged pending review.

## Files (figures/ is not a git repo; journal outputs untracked)
- new: figures/src/cddfig/layout.py, figures/src/cddfig/checks.py, figures/tests/test_layout.py,
  DESIGN_PROPOSALS.md.
- rewritten / edited: figs/fig_study4_validity.py, fig_study4_e4_example.py, fig_study4_recall.py,
  fig_study4_calibration.py, _env_page.py, specimen_entities.py, specimen_heatmaps.py, specimen_methods.py.
- Not changed: plots.py, style.py, palette.py, registry.py; the tables (tab_study4_*) and their outputs.
- BUILD_GUIDE s.2.1 package layout does not list layout.py / checks.py yet (not edited: guide owner's call).
