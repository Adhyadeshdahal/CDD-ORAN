# Team figure review, 2026-10-05 (Study 4 / bridging figures, cddfig)

### 1. fig_study4_calibration.png
- Isolated marker artifact (panel b): a lone teal diamond (Design-adjusted Granger) floats disconnected at
  (t ~ 0.011, share ~ 0.0017) because preceding values are <= 0 and masked by NaN.
- Panel letter & subtitle proximity: bold panel letters (a, b) and panel tags ("Study 4, E1-E5, R2", "E6-P bridging,
  60 episodes") sit on the same baseline with only 14 pt separation, crowding the top-left corner.
- Unbalanced legend: fig.legend at outside upper left with ncol=3 leaves asymmetric whitespace above panel b. Per
  DESIGN.md s.6, two-panel figures should use legend_top or balanced headers.
- Redundant tick labels: panel b suppresses its y-axis label but keeps identical y-tick values, repeating numbers.

### 2. fig_study4_e4_example.png
- Misplaced explanation label (panel d): "x INVALID (no recall)" is hardcoded above panel d, where all arms are valid;
  the crosses are in panels e and f.
- Vertical crowding of INVALID markers: crosses at y = 1.13 (clip_on=False) squeezed between top panels' ticks/spines
  and the bottom panels' upper boundary.
- Hidden baselines due to exact overlap (panel f): PMRT-GBM and design-adjusted partial correlation both recall 1.0;
  the hero line (1.9 pt, filled markers) hides the baseline for n in {0.5k, 1k, 4k, 8k}.
- Dangling tick marks: sharex=True + top.set_xlabel("") leaves bare x ticks and spines without labels on panels a-c.

### 3. fig_study4_env_e1.png
- Non-compliant legend placement: method legend at outside lower center below e/f violates DESIGN.md s.6 (legends at
  top row or right column; never underneath).
- Vertical boundary collisions: panel a's graph legend (bbox (0.5, 0.0)) collides with panel c's header and letter;
  panel b's multi-line note ("20 setpoints...") presses against panel d's header.
- Unlabeled hero cross in panel f: at n = 24k PMRT-GBM is INVALID, drawn as a blue x at y = 1.13 with no annotation.
- Arbitrary clipping floor: null rates in c/d hard-clipped at 0.011 with an artificial "<=0.01" tick.

### 4. fig_study4_recall.png
- Floating reference label (panel f): "Frozen PMRT" xytext=(0, 22) sits at y ~ 0.88, far above the dashed curve
  (0.55-0.75).
- Total occlusion of baselines (panel c): in E3, PMRT-GBM, PMRT-Lin and design-adjusted partial correlation all recall
  1.0; the hero covers both baselines across all n.
- Asymmetric legend: 2-column legend at outside upper left above panel a leaves empty space above b and c.
- Panel tag crowding: panel letters (a-f) immediately adjacent to tags (E1, R2...) without clear hierarchy.

### 5. fig_study4_validity.png
- Extreme horizontal compression: 6 facets (E1-E5 + Pooled) in 5.15 in -> < 0.7 in each; cramped log ticks.
- Category headers as y-tick labels: "Design-based tests" etc. rendered as bold y-tick labels in the method-name margin
  rather than distinct banner dividers.
- Top-right legend placement: legend at outside upper right above Pooled, not top-left / right column.
- Error bar collisions in Pooled: close rates (e.g. R1 Granger vs partial correlation) overlap points and whiskers.
