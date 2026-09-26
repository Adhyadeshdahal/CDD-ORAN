# QACM action-collapse check — Env I, disagreement_penalty>=1 regime

Input: `.temp/new_arch/phase1/env1/utilities_penaltyGE1.json` (Env I discovered, penalty>=1; QACM
mean = +0.2965). Ran by orchestrator directly (the cheap pane's provider endpoint was down).

| metric | value |
|---|---|
| panels / QACM values | 210 / 270 |
| QACM util mean | +0.2965 |
| QACM util std | 0.811 |
| QACM util min / max | -1.475 / +1.349 |
| satisfied fraction | 0.570 |
| distinct rounded util values | 104 (modal share 0.222) |
| **distinct QACM actions (`algo_actions`)** | **191 / 210 panels (modal action share 1.9%)** |

**VERDICT: VARIED — not a collapse.** QACM chooses 191 distinct actions across 210 decisions (most
common action only 1.9%), utility varies widely per decision (std 0.81, full [-1.47, +1.35] range), and
57% of decisions are satisfied. The +0.297 is a genuine per-decision policy, NOT a degenerate
constant-action artifact. The flatness across penalties 1-20 is argmin saturation of the discrete action
choice, not collapse to one action.
