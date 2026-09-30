# Option (a) plan: confounded incumbent arbiter + map-driven referee (Plan agent, 2026-09-30, NOT frozen)

This condenses the Plan agent's report. The full text is in the session log; see .tmp/PLAN.md.

## Key finding (DEV-1 panels)
Pico sleep share over the episode:

| t (s) | 60 | 120 | 240 | 360 | 480 | 660 |
|---|---|---|---|---|---|---|
| share asleep | 0 | .83 | .48 | .22 | .10 | .05 |

Picos fall asleep INSIDE the all-accept warm-up. The step-2 referee opened units only after 120 s, so it could never
act on the sleep -> nbr chain. The Gate A references (sub:X, noarb) already act from t = 0. So a referee active
from t = 0 on the unchanged P3 surge-L40 is legitimate; the warm-up stays unscored.

## 1. Confounded logging policy: collect_p.IncumbentPolicy
- Active from t = 0. Pressure s = max(own_prb_util, nbr_max_prb_util).
- Accept probabilities:

  | request | s < .6 | s >= .6 |
  |---|---|---|
  | saving (sleep / carrier-off / ptx-down) | .85 | .15 |
  | restore (wake / carrier-on / ptx-up) | .35 | .85 |

  | request | condition met | otherwise |
  |---|---|---|
  | SG raise (own_prot_below_frac > .05) | .85 | .30 |
  | SG lower (own_prot_below_frac = 0) | .85 | .20 |

- Modes are accept or reject only, with a floor of .15. Draws use default_rng([seed, 6622, c, x_idx, t0]). The full
  per-unit row is logged as u["probs"].
- **Variant C2:** the policy also uses an UNLOGGED 30 s trend of nbr prot_below_frac.
- PlaceboIncumbent logs the draw but applies accept.
- **MSCR-CRT v2 needs no change**: per-unit probs rows. The only change is in crt_units.build_unit_data, which must
  prefer u["probs"].
- Use H_pre = 60 so warm-up units survive.
- Validity: under family f's sharp null, the trajectory, ctx and rows are invariant, so the per-unit redraw is exact.
  The policy must read obs only, never its own past modes.

## 2. Discovery under confounding
- **GT:** fresh knockouts on the incumbent base path, with AA continuation. Warm-up plumbing needs a fix: CellKPITap
  counts only scored seconds.
- **K0:** on the confounded placebo.
- **Baselines expected to fail** (corr / INT / SHAP / two-tower / Granger / QACM / PACIFISTA).
- **Honest competitors:** Granger+ctx and an IPW-Wald test on the logged e.

## 3. Referee MapGate(M, theta): cdd_oran/decision/mapgate.py
- Reads only ctx and the request. For request (f, d):
  - **harm:** d * sum over declared rel in {own, nbr} of beta(f, rel, pv) > 0; fall back to v;
  - **costly:** d * sum_rel beta(f, rel, e) >= 0;
  - **defer iff** harm and (costly, or pressure > theta = .05).
- **Arms:** references; GT map (ceiling); MSCR; each baseline's map; random map (tag 6623); blanket no-map; best DEV
  static; Granger+ctx; IPW.

## 4. Criteria (freeze docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md before DISC)
- Discovery: K0; MSCR precision >= .80 and sign >= .90 on referee-relevant cells.
- Decision:
  - E: the MSCR arm is eligible;
  - D1: R(MSCR) - R(best associational map) >= .10, with LB90 > 0;
  - D2: MSCR beats each baseline map (Holm);
  - D3: MSCR beats the incumbent, noarb and blanket.
- PASS = K0 & E & D1 & D2.

## 5. Seeds and cost
- Register 186000-187999 "e6p_confounded", tags 6622-6624.
- Blocks: K-B 186000-079; DISC 186100-699 (600 eps); placebo 186700-739; GT 186740-759; EVAL 187000-079 (+extension
  187080-159). Referee DEV reuses 184200-239.
- About 80 CPU-h, all on Kaggle.

## 6. Kill tests first
- **K-A (~3 CPU-h, 184200-239):** GT map vs own-only GT map vs carrier sign-flipped GT map vs never-sleep static, all
  from t = 0. Kill if the GT map is ineligible, or R(GT) - R(own-only) < .10, or R(GT) - R(sign-flip) < .10.
- **K-B (~2 CPU-h):** MSCR K0 on the confounded placebo; do baseline placebo edges flip >= 1 MapGate decision per
  episode?

## Risks
1. The lever is small, or the guards fail (K-A).
2. Granger+ctx ties MSCR under C1, narrowing the claim to C2.
3. Critiques ("acting in warm-up", "MapGate rigged") and the warm-up tap plumbing.

## Amendment before K-B data (2026-09-30, no K-B data exists yet)
The K-B kill rule "a baseline map flips >= 1 MapGate decision per episode vs the GT map" is passed trivially by an
empty or underpowered map, because missing edges also flip decisions. PRIMARY rule, therefore: `label_attributable`
counts only flips caused by WRONG edges. It is the max of (a) the map vs the same map with only its correct edges (GT
TRUE, right sign), and (b) the GT map plus the method's placebo-declared edges vs the GT map. KILL if MSCR K0 fails,
or if no baseline has >= 1 attributable flip per episode. The literal rule is reported as `label`.
