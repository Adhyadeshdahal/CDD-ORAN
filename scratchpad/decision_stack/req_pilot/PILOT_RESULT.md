# Request-level pilot result (2026-09-28) — VERDICT: DEAD for E6 (predeclared rule)
Run: Kaggle e6-reqpilot1 (3 kernels, 12 diag episodes, clean bundle b00920b, e6_dirty false); 229 labelled requests.
Predeclared: PROMISING iff at same H best CV sign AUC (nonzero labels) >= 0.70 AND nudge reliability >= 0.70 AND >= 30 nz.
| H | frac exactly 0 | n nonzero | best sign AUC | reliability | pass |
| 10 | ~0.75 | 58 | 0.552 (logistic; CI 0.42-0.70) | 1.00 | no |
| 30 | 0.66 | 79 | 0.617 (HGB; CI 0.50-0.75) | 1.00 | no |
| 90 | 0.55 | 103 | 0.575 (HGB; CI 0.46-0.70) | 1.00 | no |
Magnitude CV R2 <= 0.03 everywhere. Nonzero-vs-zero AUC 0.97 (H10) -> 0.70 (H90): WHETHER a reject matters is predictable
(TS CIO rejects ~96% exactly 0 at H10), its SIGN is not. Median |dJ| nonzero 1-5 units (H90 max 105) vs J ~5400/slot.
Nudge reliability 1.00 is the optimistic 1e-6 m check (predeclared caveat; cross-numpy reliability was 0.56).
Implication: in E6 neither region-policy nor request-level features predict the sign of arbitration effects;
the E6 action space lacks systematic conflict structure. E6-P must show sign-stable conflict effects (screen) before
any learned arbiter is built.
