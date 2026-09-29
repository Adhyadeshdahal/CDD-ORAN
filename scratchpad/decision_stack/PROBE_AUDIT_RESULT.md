# e6-probe/2 carryover audit (30 jobs, fit seeds j<5 × 6 strata, local; ANALYSIS.txt) — efficacy NOT inspected
First stage by arm (blocks / effective / actuation / restoration / superseded / abort rate):
- cio 28/25, 0.983, 0.614, 0.386, 0.107      - hys 17/14, 1.000, 0.977, 0.023, 0.176
- ttt 22/18, 1.000, 1.000, 0.000, 0.182      - ll_ratio 25/19, 0.981, 0.212, 0.788, 0.200
- carrier 28/15, 0.976, 0.427, 0.573, 0.464   - sham 54/36, –, –, –, **0.333**
Placebo on real no-probe references: 400 tests, rejection 0.0375 (mscr) / 0.050 (signed) at α 0.05 → calibrated.
Replays (30): pre-flip max |diff| 0.0 (exact); |Δutil| during observation 0.006; lag-1 CRT (15 tests): none declared.
ISSUES: (1) abort rules fire on 33 % of SHAM blocks and 46 % of carrier blocks → thresholds trip on natural KPI
noise, not probe harm → effective blocks lost (carrier 15/28). Needs threshold redesign justified from sham/no-probe
reference variability (a no-efficacy diagnostic), then re-audit. (2) xApps supersede ll_ratio (79 %), carrier (57 %),
CIO (39 %) probes → estimand is the assigned-probe effect incl. xApp reactions (already declared).
