# Step 2 plan: a regime referee driven by the causal map (Plan agent, 2026-09-30 01:10 NST, NOT frozen)

This is the design to build if step-1 v2 ends PASS or PARTIAL. If it ends KILL, the GT-knockout map becomes the
primary arm (a declared privilege) and the MSCR arm is still reported. Numbers come from gt-1 and
runs/e6p-v2-summary.json.

## 0. What the data already says
Paired knockout, 90 s, per unit:
- **Carrier-off:** +9.4 pv on its own cell and saves 6.6 kJ. There are about 22 per episode.
- **Pico sleep:** +15.4 pv on the neighbours. It **costs** about +5.9 kJ net, because the neighbour macro then turns
  its carrier back on. So refusing sleep under pressure lowers V and also saves energy, and that saving can pay
  for refusing some carrier-offs. This plausibly explains why the constrained oracle's energy retention is 1.26.

Scale of the problem:
- The ES+PowerES saving is about 750 kJ per episode. A retention margin of 0.07 is about 53 kJ, or about 8 refused
  carrier-off windows.
- The loss to recover is Λ ≈ 45 psvr, about 245 protected violated UE-s per episode.
- The paired per-seed SD of (AA − arm) is 26-40 psvr.

## 1. Referee (`cdd_oran/decision/referee_p.py`)
- **Hook.** `UnitArbiter(RefereePolicy, T=10, open_rule="feasible")`, reading only the obs-only `unit["ctx"]`.
  Re-deciding every 10 s turns a refusal into "wait until the pressure drops".
- **Gated actions** (energy-saving direction only):

  | request | options |
  |---|---|
  | pico sleep | accept / defer |
  | macro carrier-off | accept / defer |
  | PowerES ptx-down | accept / soften (half) / defer |

  Wake, carrier-on, ptx-up and SliceGuarantee are always accepted.
- **Regime.** A leaf of an honest depth-2 regression tree, per family, on the ctx features:
  - it predicts the paired Δpv of accepting;
  - it splits on one half of the label episodes (tag 6619) and estimates on the other;
  - each leaf needs ≥ 25 labels and ≥ 6 episodes, with ≤ 4 leaves per family.
- **Uses of the MSCR map.**
  1. **Gating.** A family is gated only if it has a harmful edge into pv or v.
  2. **Features.** Only the relations it has edges into, and only the mediators tied to those KPIs: pv →
     prot_act_ue, prot_below_frac, prot_dem_share; v/load → prb_util, act_ue; e → carriers, asleep, ptx_db.
  3. **Outcome scope.** Labels are summed over the declared relations; the sleep → nbr e edge lets the model see
     the energy saving of refusing sleep.
  4. **Monotone.** Leaf values are isotonic in the edge-sign direction.
- **Policy table.** A knapsack solved by enumeration (≤ 12 leaves × 3 actions):
  - maximise Σ n̄_ℓ b̂_ℓa;
  - subject to energy ≤ β·S_A, and v and rlf increases each ≤ 0.05 × AA;
  - a leaf-action is allowed only if the 90 % lower bound of its benefit is > 0 (episode-cluster bootstrap).

## 2. What-if model
- **Output.** Per (family, leaf, action), the mean 90 s effect on pv, e, v and rlf. ll_viol and non-protected eMBB
  are checked on-policy only.
- **Data (DEV only).**
  - (a) paired knockout labels, gt-1 + gt_ext (the latter only after step-1 v2 is scored);
  - (b) π0 logs with MSCR-v2 design-centred contrasts.
  - Pool (a) and (b) by inverse variance when they agree within 2 SE; otherwise use (a).
- **Noise.** Averages over leaves, EB shrinkage toward the family mean, a lower-bound rule, and no per-unit
  predictions.
- **Optional targeted knockouts.** Sleep-going units, k = 2: 120 episodes ≈ 4.4 CPU-h.
- **β.** The largest β in {.03, .05, .07, .09} with DEV retention ≥ .93 and guardrails ≤ 1.05 × AA.

## 3. Evaluation (`docs/benchmark/E6P_REFEREE_PROTOCOL.md`, frozen before EVAL)
- **Seeds.** Register 184000-185999 as `e6p_referee`, with tags 6618-6621.

  | seeds | use |
  |---|---|
  | 184000-199 | knockouts |
  | 184200-239 | DEV on-policy |
  | 185000-099 | EVAL (N = 100) |
  | 185100-159 | power extension |
  | 185200-299 | step-3 fallback |

  Never 150200-150399, 155200-155399 or 160000-179999.
- **Arms** (paired by seed):
  - freeze, AA, sub/pair arms;
  - QACM;
  - PACIFISTA at δ .25 and .5;
  - the best deployable static (cell-priority plus global envelopes, chosen on DEV);
  - referee-MSCR;
  - hindsight static (reference, 24 seeds);
  - O_tape (cite R 1.35, or run 12 seeds for ~16 CPU-h).
- **Metric.** R as in `analyse_pair_stratum`. Eligibility is energy_ok (retention ≥ .90) and guard_ok (≤ 1.10 × AA).
- **Pass.**
  - E: the referee is eligible.
  - P1: R_ref − max(eligible static) ≥ 0.10, with the paired 90 % lower bound > 0.
  - P2: the referee beats QACM and PACIFISTA, each with lower bound > 0.
  - P3: R_ref ≥ 0.40.
  - Holm correction across P1 and P2. **PASS = E ∧ P1 ∧ P2.**
- **Power.** With paired SD ≈ 35 psvr, N = 100 gives SE ≈ 0.08 R, about 85 % power at ΔR = 0.2.
- **Cost.** ≈ 110 CPU-h (DEV 37, EVAL 36, hindsight 34): two Kaggle batches.

## 4. Step-3 ablation (same pipeline, data, tree, knapsack and β rule)

| arm | map |
|---|---|
| A0 | MSCR-v2 map |
| A1 | no map (all families, all features, network scope) |
| A2 | SHAP-GBDT map |
| A3 | GT-knockout map (privileged ceiling) |
| A4 | a random map of the same size (tag 6621) |

**Causal claim:** A0 is eligible, R(A0) > R(A1), and R(A0) > R(A2), each with the paired lower bound > 0 after Holm.
Supporting evidence: A0 ≥ A4, and A0's calibration slope is in [0.5, 1.5]. If two maps give identical feature sets,
that comparison is void.

## 5. Risks and the cheapest kill tests
**Risks.**
1. Too little lever under the energy budget: R_ref could stay below ~.3.
2. Non-additivity and horizon: many deferrals at once may not add up the way single-unit labels do. The DEV
   calibration slope checks this.
3. The ablation may not discriminate, because pressure features dominate. We would then report exactly that.

**K-L0** (analysis only, < 1 CPU-h, runnable on gt-1 plus π0 logs). Fit the leaves and the knapsack at β = .07, then
compute R_pred = ΣΔV / 245. Kill if any of these holds:
- R_pred < .35;
- R_pred − R_pred(best global envelope) < .10;
- split-half sign agreement of the leaves < .9.

**K-L1** (about 4 Kaggle CPU-h, 20 DEV seeds). Kill if the referee is ineligible at every β, or if R_DEV <
R_best-static + .05.

## Build order (3-4 h)
1. `referee_p.py`: MapSpec, features/scope, honest tree, EffectTable, knapsack, RefereePolicy.
2. `tests/test_referee_p.py`: an all-accept table reproduces AA; the policy is ctx-only; direction gating; the
   knapsack.
3. `scratchpad/e6_dev/e6p_referee.py`, with stages kl0 / kd / dev / eval / summary.
4. The registry entry and the protocol draft.
