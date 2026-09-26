# 011 — Planner + Objective decision (durable record)

**Status:** DECISION RECORD (not an implementation plan). Codifies a decision made 2026-09-05/06 that
until now lived only in gitignored reports, `.temp/`, auto-memory, and chat logs — never in a git-tracked
doc. This file is the tracked record. Detailed evidence is in the (gitignored) working reports named below.

**Scope caveat (read first):** the quantified fixes below (FM-1/FM-2, `utility_weight`) were validated on
the **RETIRED legacy Environment I–IV**, not the live E-series (E1–E5). Their *direction* is trusted; their
*magnitudes and even applicability to the E-series objective must be re-validated on E-series* before any
of it is built. See `legacy-vs-eseries-boundary` (memory).

---

## 1. Decision: the planner is NOT the bottleneck — stop hunting for better planners

Three-survey falsifiable synthesis + gate (2026-09-05). The four default planners **QACM · CEM · MPPI ·
MCTS** collapse to **one decision rule**: single-parameter, single-step grid-argmin over a flat, saturating,
model-predicted cost. Planner differences are noise; two of three legacy envs sat **below the do-nothing
floor**.

- **FM-6 (STOP holds):** under matched decision-sets, oracle-graph ≯ discovered-graph on the decision. The
  lever is **objective + world-model + graph recovery**, not search.
- Evidence: `reports/2026-09-05-planner-redesign-and-env-boundary/index.html` (gitignored); memory
  `planner-set-redesign-direction`, `bayesian-direction-gated`.

### Keep / cut / reclassify
- **KEEP:** MPPI, CEM, JointMultiNCP; integer-margin on the diagonal CEM/CMA path.
- **CUT:** full/dense CMA-ES covariance; demote MCTS + RecedingHorizonCEM to **ablation flags**; do not
  pursue LM-MA-ES / IPOP-BIPOP / GP-BO / TuRBO / SMAC / diffusion-MPC.
- **CID (causal influence-diagram planner):** earlier slated to build, but adversarial review found it
  **deterministically identical to QACM** on the current single-param path (same lattice, bit-identical
  `score_batch`, same tie-break). It earns nothing until **graph-pruning / joint-queries** land. Not a
  differentiator today. Source: memory `cid-planner-reviewed`.

## 2. Decision: the OBJECTIVE FUNCTION is the real lever — three ranked failure modes

- **FM-1 — flat objective (model-independent).** `cost.py:93` clamps hinge distance to 0 at the
  satisfaction threshold; once an xApp is "satisfied" the objective is indifferent to how far above it sits,
  so the planner stops climbing (`qacm.py:103` strict-`<` argmin) and leaves ~+1.3 utility on the table.
  Fix = additive z-utility term `utility_weight` (**Option A**), validated: legacy Env I **−0.083 → +0.303**
  at λ=0.25. Source: memory `option-a-utility-weight-validated`; `plans/archive/sources/lever2_objective_design.md` (copied from `.temp/new_arch/reports/`).
- **FM-2 — OOD model trust.** Planner optimizes model-predicted KPIs; off-distribution the model is
  confidently wrong. Ensemble-disagreement penalty exists but `ood_threshold: null` (OFF by default).
  Lifting it: legacy Env I **−0.083 → +0.297**.
- **FM-3 — coordination** (the one real coupling) is discovery-hidden AND joint multi-param eval is model-OOD
  for the single-param-trained model (`joint.py:131-158` applies NCPs sequentially by design). The real fix
  is a **world-model multi-intervention capability**, not a planner. (Same conclusion as the E2 P0→K5 work,
  memory `e2-harmful-edge-discovery-gap`.)

## 3. Decision: CMA-ES is CUT — block-covariance buys nothing

Gate `scripts/planner_gate.py`, branch `agent/planner-redesign @ f5db230` (unmerged), TrueSim oracle,
matched budget 256×10, 48 decisions, P1/P3 needle:
- **BlockCMAES − DiagCMAES = −0.0285 ± 0.0427** (within seed spread; zero). The covariance machinery — the
  thing that makes it CMA-ES rather than plain CEM — adds nothing. **CUT.**
- The real gain was **joint-parameter evaluation**: DiagCMAES − JointMultiNCP = **+4.82**, but oracle-only
  and model-OOD by design → a world-model capability, not an optimizer. (Floor: do-nothing −0.1257;
  coarse-grid 41³ = +4.3237.)
- Source: memory `cma-es-gate-falsified`.

## 4. Decision: QACM is a THIRD-PARTY MARKET BASELINE, not our planner

User directive 2026-09-05 (memory `planner-set-redesign-direction`): *"QACM is NOT our planner — it is
third-party (other authors') work. Reclassify it from an ensemble member to a standalone BASELINE."*

**Experimental roster = three arms:**
1. **do-nothing** (scored floor);
2. **our causal solution** (the discovered-structure + world-model + objective-fixed decision);
3. **market baselines**, including **QACM** (verified a genuine varied policy — 191 distinct actions / 210
   panels, `plans/archive/sources/qacm_collapse_check.md`), and do-nothing / random-action references.

`docs/benchmark/SPEC.md` currently uses QACM only as "the SAME single-control planner for both arms" (E2
lines 265-268; E4 `GATE_CONTRACT_E2.md:154`) and **never states it is third-party** — that committed
wording is misleading and should carry a one-line clarification that QACM is a third-party market baseline.

---

## What this means going forward
"Which planner are we going with" = **keep MPPI / CEM / Joint, treat QACM as a market baseline, and do NOT
expect the planner to be where the win comes from.** The wins are: (a) objective function (FM-1/FM-2), (b)
world-model multi-intervention (FM-3), (c) discovery completeness (esp. the harmful fan-out edge). Before
implementing (a), re-validate FM-1/FM-2 on the **E-series** objective (`analysis/counterfactual_metrics.py`,
`SEMANTICS.md:205-228`) — the numbers above are legacy-env and may not transfer.
