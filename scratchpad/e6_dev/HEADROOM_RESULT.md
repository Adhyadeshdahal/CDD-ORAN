# E6 DEV headroom: veto-mask lookahead oracle (Kaggle e6-headroom-a/b, bundle @ccb5481, 60/60 jobs)

Setup: M4 mix (MRO+TS+ES+SLICE), mobility mixed, 120 s warm-up + 600 s scored, DEV seeds 11–15, loads medium/high.
Oracle: every 10 s, copy env (true plant + future tape), roll each of 8 per-xApp veto masks 30 s ahead, keep the one
minimising violations + w_LL·LL violations + λ_E·kWh. AR (veto) class only. Raw: `headroom_all.jsonl`.

| load | arm | SVR | /noarb (min–max) | /freeze | LL/noarb | kWh/noarb | ES retention | SLICE retention |
|---|---|---|---|---|---|---|---|---|
| medium | noarb | 271.9 | 1.000 | 1.102 | 1.000 | 1.000 | 1 | 1 |
| medium | freeze | 245.2 | 0.912 (0.83–1.02) | 1.000 | 1.127 | 1.077 | 0 | 0 |
| medium | oracle λ0 w0 | 263.5 | 0.975 (0.94–1.03) | 1.075 | 1.020 | 0.999 | 1.00 | 1.04 |
| medium | oracle λ1e4 w5 | 264.6 | 0.980 (0.95–1.02) | 1.079 | 0.932 | 0.980 | 1.31 | 0.98 |
| medium | oracle λ3e4 w5 | 286.4 | 1.056 (0.98–1.24) | 1.168 | 0.965 | 0.948 | 1.83 | 1.17 |
| high | noarb | 612.5 | 1.000 | 1.074 | 1.000 | 1.000 | 1 | 1 |
| high | freeze | 571.9 | 0.931 (0.91–0.95) | 1.000 | 1.972 | 1.016 | 0 | 0 |
| high | oracle λ0 w0 | 589.5 | 0.960 (0.94–0.98) | 1.031 | 1.415 | 0.998 | 0.76 | 0.41 |
| high | oracle λ1e4 w5 | 592.2 | 0.966 (0.94–0.98) | 1.037 | 0.897 | 0.988 | −0.21 | 1.14 |

## Findings
1. **Veto-only arbitration has little headroom in E6**, even with perfect knowledge of the future: 2–4 % SVR vs
   noarb, and never below freeze (freeze is 7–9 % better than noarb on SVR but keeps no xApp goal). Same pattern as
   D0 on E2-CL (AR ≈ no headroom; bounded joint nudges ≫).
2. The 30 s greedy lookahead is **not an upper bound**: with a high energy price it is worse than noarb (up to 1.24×
   on one seed). The horizon is short relative to the dynamics (TS/ES cadence 10 s, MRO 30 s, carrier dwell 120 s),
   so myopic vetoes can backfire later.
3. The oracle mostly vetoes SLICE or freezes everything for 10 s; accept-all is picked only 10–20 % of decisions.
4. ES retention is noisy (energy gap noarb–freeze is only 2–8 %), so a ratio-of-differences guardrail is unstable. Use
   an absolute energy margin instead (as the statistician argued).
5. Cost: ~21 min per oracle job on Kaggle CPUs (vs ~4 min local).

## Implication
Under the strong-edge bar (≤ 0.75 × every baseline, freeze included, with goals retained) a veto arbiter cannot win
on E6 as built. Headroom must come from a richer action class — MODIFY (partial steps) + coordinated own writes
(e.g. restoring CIO/carriers, tuning ll_ratio jointly) — and a longer planning horizon. Next diagnostic: a
modify/write oracle with 60–120 s horizon, before any learned arbiter.

## Round 2: modify + own-write oracle and 90 s veto (Kaggle e6-headroom2-{a,b,c}, 18/18, seeds 11–13)
Raw: `headroom2_all.jsonl`. Ratios are per-seed means; writes unbudgeted (count shown).

| load | arm | SVR | /noarb | /freeze (min–max) | LL/noarb | kWh/noarb | RLF/noarb | writes |
|---|---|---|---|---|---|---|---|---|
| medium | noarb | 225.5 | 1.000 | 1.135 | 1.000 | 1.000 | 1.000 | 0 |
| medium | freeze | 198.2 | 0.883 | 1.000 | 1.063 | 1.082 | 0.787 | 0 |
| medium | veto 30 s | 220.9 | 0.977 | 1.111 | 0.947 | 0.980 | 0.954 | 0 |
| medium | veto 90 s | 206.4 | 0.916 | 1.040 | 0.959 | 0.984 | 0.949 | 0 |
| medium | mod λ0 | 188.5 | **0.837** | **0.950** (0.88–0.98) | 1.011 | 1.059 | 0.871 | 461 |
| medium | mod λ1e4 w5 | 199.6 | 0.887 | 1.006 (0.93–1.07) | 0.963 | 1.003 | 0.851 | 296 |
| high | noarb | 546.6 | 1.000 | 1.082 | 1.000 | 1.000 | 1.000 | 0 |
| high | freeze | 505.7 | 0.924 | 1.000 | 1.613 | 1.016 | 0.800 | 0 |
| high | veto 90 s | 522.9 | 0.956 | 1.035 | 0.994 | 0.978 | 0.986 | 0 |
| high | mod λ0 | 499.5 | **0.913** | **0.988** | 1.124 | 1.010 | 0.855 | 545 |
| high | mod λ1e4 w5 | 505.6 | 0.925 | 1.001 | 0.941 | 0.992 | 0.814 | 529 |

Findings:
1. Richer actions + longer horizon help: modify/write oracle reaches 0.84–0.93 × noarb (veto 30 s: 0.96–0.98;
   veto 90 s: 0.92–0.96). Horizon alone is worth ~6 points at medium load.
2. With an energy/LL price it reaches **freeze-level SVR (1.00×) while keeping noarb energy and better LL/RLF** than
   noarb, i.e. it Pareto-dominates both noarb (SLA) and freeze (goals). Without the price it beats freeze by ≤5 %
   but gives up ES's energy saving.
3. It does NOT get near 0.75 × freeze. On E6 as built, the xApps add little positive SLA value (TS ≈ −2 %,
   MRO ≈ 0), so most coordination value is harm avoidance (noarb → freeze), and the ceiling vs freeze is small.
4. The oracle is crude: 14 random candidates, network-wide knob-type macros, 60 s horizon. Winning picks are diffuse
   (top candidate ≤ 7 %). True headroom may be higher with per-cell targeted actions; this is a lower bound.

## Round 3: per-region oracle (Kaggle e6-headroom3-{a,b,c}, 12/12, seeds 11–13; raw headroom3_all.jsonl)
| load | objective | /noarb (min–max) | /freeze (min–max) | kWh/noarb | LL/noarb | RLF/noarb |
|---|---|---|---|---|---|---|
| medium | SLA only | 0.815 (0.77–0.86) | 0.924 (0.87–0.98) | 1.055 | 0.984 | 0.895 |
| medium | energy/LL priced | 0.847 (0.79–0.89) | 0.960 (0.91–1.02) | 1.005 | 0.932 | 0.832 |
| high | SLA only | 0.902 (0.89–0.92) | 0.976 (0.97–0.98) | 1.004 | 1.070 | 0.824 |
| high | energy/LL priced | 0.909 (0.88–0.94) | 0.984 (0.97–0.99) | 0.985 | 0.878 | 0.819 |
Diminishing returns vs round 2 (medium 0.84→0.82 vs noarb). Perfect knowledge never reaches 0.87× freeze.
