# E2 3-arm thesis contrast, END-TO-END on the LIVE trap

Date: 2026-09-06
Env: E2 live-trap (frozen commit `9d87a60`), branch `feat/v2`. Dev-only; nothing committed,
no tracked source modified, `runs/e2slice-recovery/` untouched.

## TL;DR — the thesis FIRED

On the live trap, running the full spine **discovered-structure → world-model → H=1 decision →
REALIZED score on the TRUE env** over the frozen 32 K5-positive committed states:

| arm | mean realized R | mean regret | trap-fall fraction | true-K5 satisfied |
|---|---:|---:|---:|---:|
| do-nothing (floor) | -102.56 | 60.23 | 32/32 (1.00) | 0/32 (0.00) |
| **oracle / complete fan-out** | **-42.33** | **0.00** | **6/32 (0.19)** | **26/32 (0.81)** |
| **discovered structure (decoy)** | **-57.48** | **15.15** | **30/32 (0.94)** | **2/32 (0.06)** |

The **oracle prevents the harmful P0 move** — it keeps true K5 ≤ −25 in 26/32 states (mean true
K5 = −26.2) at regret 0. The **discovered arm systematically falls into the trap** — it leaves K5
violated in 30/32 states (mean true K5 = −2.4, nowhere near the −25 safety line), costing
**+15.15 mean regret** (median +9.0, max +51.2) and realized R that is strictly worse than the
oracle in **25/32** states. The single structural difference driving this is the missing
`P0→K5` edge (absent in 10/10 discovery seeds).

**Null control (K5-negative bank, no trap): oracle and discovered are byte-identical** (both
realize −60.85, regret 0, trap-fall 0/32). Missing `P0→K5` costs nothing exactly where the trap
is inactive — the discovered arm's failure localizes precisely to the trap states.

## What each arm is

All three arms share the same locked call list (`apply_action(P0,v); advance(); advance()`,
2-advance actuation lag, latent noiseless scoring) and the same locked objective `R`
(`cdd_oran/analysis/v2_regret.py`: hinge cost over the full panel `{xApp0,xApp1,xApp2,xApp4}`,
`P0_GRID = linspace(-100,100,101)`). They differ **only in the world model** used to *select*
`do(P0)`; realized R is always scored on the TRUE env.

1. **do-nothing** — hold P0 at its committed value (nearest grid point). Floor.
2. **oracle / complete fan-out** — world model = TRUE `E2V2Env` (complete fan-out incl.
   `P0→K5`). Selects `argmax_v R_true`.
3. **discovered structure** — world model = `E2V2Env(decoy_omit_p0_k5=True)`, i.e. every TRUE
   mechanism kept **except** P0 is frozen inside the K5 term, so the model predicts K5 is
   invariant to P0. Selects `argmax_v R_decoy`.

### Why the decoy env is the faithful model of the *real* discovered masks

I read all 10 discovery outputs `runs/e2slice-redesign/replicate-00..09/discovery_rcot_v2.json`
and confirmed the decision-relevant structure is **uniform across every seed**:

- `P0 → {K0,K1,K2}` recovered in **10/10** seeds (col 0 of rows K0,K1,K2).
- `P0 → K5` recovered in **0/10** seeds.
- K5's discovered parents are at most `{P6}` (seeds 0,1,8,9) or empty (seeds 2–7) — **never P0
  and never any KPI**, so there is no direct *or indirect* P0→K5 path in any seed.
- The only spurious edges are a handful of KPI→KPI false positives (seed0 K1←K3, seed1 K0←K1,
  seed4 K3←K3, seed9 K0←K0/K2←K2/K4←K5). None chains P0 to K5.

Because every seed agrees that **P0 does not reach K5**, the discovered arm's P0-decision is
seed-invariant, and `decoy_omit_p0_k5=True` reproduces it exactly (P0→K0,K1,K2 modeled with the
true mechanism, P0→K5 omitted). The spurious KPI→KPI edges are inert under an env-mechanism world
model (the E2 SCM has `k_t = f(p_{t-1})` with no KPI→KPI channel), so they cannot alter the
result. The discovered arm is therefore reported as a **single deterministic structure**, which
is the correct aggregation of the 10 seeds — not a 10-way average of differing structures.

## States used, and why

The frozen decision-gate bank (`scripts/e2_decision_gate.py`, reused verbatim): for
`env_seed = 0..4095`, `episode=0`, noise OFF, 3 neutral advances, label each committed state by
its true-K5 response geometry over the P0 grid; retain the first **32 K5-positive** and first
**32 K5-negative** states.

- **K5-positive (primary)** = the exact frozen positive-control bank where the trap is live
  (moving P0 can drive K5 across the −25 line; a safe action always exists by construction). This
  is where the thesis must fire.
- **K5-negative (null)** = states where K5 barely responds to P0 (`S5 ≤ 0.05`). The discovered
  arm should cost nothing here. Reported as the control.

## Chosen do(P0) contrast — illustrative K5-positive states

`a_none` = do-nothing (committed P0); `a_O` = oracle choice; `a_D` = discovered choice.
`rzN/rzO/rzD` = realized R. `k5O/k5D` = true K5 at the oracle/discovered choice (≤ −25 = safe).

| seed | a_none | a_O | a_D | rzN | rzO | rzD | true K5@O | true K5@D | discovered trap? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0 | 88 | 48 | −22 | −118.1 | −53.0 | −57.5 | **−32.7** | −0.0 | YES |
| 1 | 40 | 6 | 6 | −102.6 | −52.8 | −52.8 | −0.0 | −0.0 | YES |
| 3 | 8 | −88 | 18 | −113.3 | −64.7 | −90.2 | **−34.5** | −0.0 | YES |
| 4 | 80 | 16 | −26 | −98.8 | **−8.7** | −56.1 | **−25.6** | −0.0 | YES |
| 5 | −20 | −4 | −4 | −101.0 | −59.3 | −59.3 | −0.0 | −0.0 | YES |
| 7 | 60 | −22 | −10 | −113.2 | **−9.2** | −43.4 | **−33.2** | −0.0 | YES |

The pattern: where the oracle can see `P0→K5`, it deliberately steers P0 to push true K5 below
−25 (seeds 0,3,4,7 — sometimes to a P0 far from the local-KPI optimum, e.g. seed 3 picks −88),
accepting a small local-KPI sacrifice for a much better global R. The discovered arm, blind to
`P0→K5`, spends P0 only on the locally-attractive K0/K1/K2 and leaves K5 pinned near 0
(violated). Seeds 1 and 5 are cases where oracle and discovered happen to coincide but the
committed geometry leaves K5 unsatisfiable within reach at that operating point — both still
count as trap states because a safe grid action existed.

## Predicted vs realized — the mechanism of the trap

Mean **predicted** R (arm's own world model at its own choice) vs **realized** R (TRUE env):

| arm | mean predicted | mean realized |
|---|---:|---:|
| oracle | −42.33 | −42.33 (true model ⇒ equal) |
| discovered | −61.63 | −57.48 |

The discovered world model predicts its chosen action is a **+40.9 improvement over doing
nothing** (−61.6 vs the −102.6 floor) — a genuinely attractive move, because it *correctly*
predicts the K0/K1/K2 gains. That is the lure. What it cannot represent is that the same P0 that
buys those local gains leaves the safety KPI K5 in violation; the oracle can, and pays a little
locally to secure K5 globally. Note the live-trap predicted/realized gap for the discovered arm
is modest (−61.6 vs −57.5) rather than a wild over-prediction: because the decoy freezes K5 at
its already-violated committed value (~0), it does not over-promise on K5 — it simply has **no
lever to fix it**. The trap here manifests as "captures the local gains, structurally cannot
secure the global safety KPI," yielding high regret (+15.2) and near-total trap-fall (0.94),
rather than the old dormant-env prototype's "over-predict +16 then realize −52" story. Same
thesis direction; cleaner mechanism.

## Caveats / adaptations

- **Prototypes not persisted.** `scratchpad/ms_world_model_dev.py`, `scratchpad/decision_harness/`
  and `scratchpad/trap_live/` referenced in the task do not exist anywhere in the tree
  (checked worktrees and git history). I rebuilt the spine directly on the FROZEN locked
  machinery — `cdd_oran/analysis/v2_regret.py` (objective R, `score_grid`, `P0_GRID`, 2-advance
  lag) and `scripts/e2_decision_gate.py` (bank + panel) — so the objective and realized scoring
  are the locked ones, not a re-derivation. Runner: `scratchpad/e2_thesis_endtoend.py`; raw
  per-state results: `scratchpad/e2_thesis_results.json`.
- **World model = mechanism-restricted-to-structure.** Both intervention arms get the *exact*
  mechanism for the edges their structure includes; they differ only in whether `P0→K5` is
  present. This is a deliberate structure ablation: it isolates the thesis variable (the missing
  edge) from mechanism-estimation noise. A data-driven world model fit on `rows.npz` would add
  fitting error but cannot change the qualitative result, since K5's discovered parents contain
  no P0 path in any seed (verified) — the discovered model's ∂K5/∂P0 is identically 0 either way.
- **do-nothing floor.** On the K5-positive bank the committed P0 already violates K5 (mean true
  K5 ≈ 0), so do-nothing is the worst arm by construction; both interventions beat it. The
  informative contrast is oracle vs discovered.
- **Seed axes differ.** The decision-gate bank is indexed by an `env_seed 0..4095` scan; the
  discovery replicates are seeds 0..9 of the discovery data generation. What transfers between
  them is the *structure* (the mask), which is seed-invariant in the decision-relevant sense, so
  applying the one discovered structure to all bank states is correct.

## Verdict

**Thesis FIRED.** On the live E2 trap, the complete/oracle fan-out prevents the
locally-attractive-but-globally-harmful P0 move (regret 0, true K5 ≤ −25 in 26/32, trap-fall
0.19), while the discovered structure — missing `P0→K5` in 10/10 seeds — systematically falls
into the trap (regret +15.15, true K5 satisfied in only 2/32, trap-fall 0.94), and the two arms
are identical exactly where the trap is inactive (K5-negative null: regret 0, trap-fall 0). The
binding constraint to the end-to-end thesis is confirmed to be **discovery completeness of the
harmful `P0→K5` edge**, not the world-model or the planner.
