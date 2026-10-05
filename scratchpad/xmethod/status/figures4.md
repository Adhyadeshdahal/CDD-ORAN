READY-TO-MERGE (figures4: user picks E1-B page + G-A graph + V-A2 validity built into the registered cddfig figures)

## What changed (D:/academia/major-project, outside any git repo except the journal; nothing there committed)
- figures/src/cddfig/envgraph.py (new): truth-graph panel G-A. Layered bipartite; the node order with the fewest
  crossings (exhaustive KPI orders up to 7 KPIs + barycentric actions + adjacent swaps; never worse than natural);
  source trunks and target-bundled plain edges; other kinds (gated, Z) keep their own port and arrowhead; lagged
  KPI -> KPI arcs on the right, nested, outermost labelled t−1; R3 policy on Z as dotted curves left of the actions;
  legend inside the canvas.
- Crossings, natural -> drawn: E1 1 -> 0, E2 21 -> 6, E3 0, E4 0, E5 4 -> 0 (Z ordered with the actions in E5).
- figures/src/cddfig/layout.py: panel_header(inset=True) (letter at the canvas edge); method_rows / method_labels
  (V-A2: gap 0.95 between methods, pitch 0.92, bold name on the first regime row); DODGE_NARROW_PT 2.6.
- figs/_env_page.py rewritten to layout E1-B: fixed 1.30 x 3.34 in graph canvas left, 2 x 2 rate / recall vs n
  right; ground truth for E1-E5 copied read-only from cdd_oran/envs/v2/e1-e5.py + worlds/generate.py.
- E4 page: the rate panels show the placebo edge (no truth-null candidate in R1 / R2; lam 1 cells). The graph shows
  Z -> K0 and the R3 policy Z -> P0, Plc. The data panels show R1 / R2 only, as in the template; R3 is in the
  validity figure.
- figs/fig_study4_env_e1..e5.py: e2-e5 are new registered figures. Caption hints use the TERMINOLOGY s.2 one-liners.
- figs/fig_study4_validity.py: V-A2 via layout.method_rows / method_labels (banners, forest + grid unchanged).
- figs/tab_study4_env.py: E4 failed on validity NO_READ (threshold-only methods in E4 R1 / R2 have no validity
  read). It now gets the mark ° and a header note. Tables written for E1-E5 (journal tables/*.tex + csv).
- tests/test_envgraph.py (new, 11 tests): crossing cap per environment (E2 <= 6, others 0), ordering never adds
  crossings, no placebo child, every edge endpoint exists.
- prototypes/ left as is (temporary).

## Build and tests
- uv run cddfig build (all 16 registered: 9 paper PDFs to the journal's figures/ + 16 previews): 13 s.
- uv run pytest -q: 182 passed (131 before + 4 new env pages x 10 checks/builds + 11 graph tests).
- All 9 layout checks pass on every figure. One fix was needed: in the 1.5 in data panels, n 500 / 1000 sit
  ~15 pt apart, so the 3.4 pt dodge put markers within 1.3 pt of the next n. The env pages now use the 2.6 pt step.

## DESIGN.md changes (all tagged [NEW 2026-10-05])
- s.7: Method separation rule for forest / value grids (V-A2), and the narrow-panel dodge step 2.6 pt.
- s.8.1 (new): environment truth-graph panel (G-A): size, columns, crossing minimisation, bundling, ports, arcs,
  edge kinds, node roles, legend.
- s.12: fixed slots. Regimes: R1 slot 1, R2 slot 2, R3 slot 3, (R1 + R2 pooled slot 4). Graph roles: action /
  NCP slot 1, KPI slot 3, placebo slot 4, Z slot 5. Different entity types never share a figure's encoding; R4 is
  never plotted (slot 5 if ever).
- s.13: helper list extended (method_rows / method_labels, cddfig.envgraph).

## Open
- Not committed anywhere: figures/ and DESIGN.md (no git); journal figures/ + tables/ (journal repo, user commits).
