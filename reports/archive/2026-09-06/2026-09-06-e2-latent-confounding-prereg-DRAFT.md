# E2 Slice Redesign — Typed Truth under Latent Confounding (PRE-REGISTRATION)

> ## ⚠️ DRAFT — FOR HUMAN REVIEW; **NOT FROZEN**, **NOT COMMITTED**
> This file lives in `reports/` (gitignored). It is a decision document, not the frozen
> contract. Nothing here has been run against E2 truth. The **frozen** version of this
> protocol must be authored and committed under `docs/benchmark/` **before** any recovery
> scoring (see §8 for the exact tracked home and freeze ordering). Do not score, tune, or
> report a recovery number until that commit exists.

**Date:** 2026-09-06 · **Branch:** `feat/v2` · **Author:** research agent (for `internal@brihatech.com`)
**Supersedes (scientifically, not by deletion):** the frozen lagged-KPI E2 discovery protocol
`docs/benchmark/E2_DISCOVERY_PROTOCOL.md` (`protocol_commit 828e345`) and Plan `plans/007-e2-discovery.md`.
The retired pdCor method stays on disk as a documented naive-DAG baseline (§8.5).

---

## 0. One-paragraph summary

The frozen E2 discovery method (U-centered partial distance correlation + per-candidate
permutation null + per-target BH-FDR) over-selects KPI→KPI edges ~9× (17 FP / 36 candidates,
0 TP, seed-0). A 5-lens analysis plus a truth-free calibration study concluded this is **not a
fixable estimator bug** but a scoring target that penalizes a method for reporting associations
that are, in the observable columns, real. We re-pose E2's truth as **typed edges** — directed
**P→KPI (manipulable)** vs bidirected **KPI↔KPI (shared-latent-parent / confounded)** — and
score two separate metrics. **However** (see §2, the pivotal correction) the *current frozen
(6,14) construction does not actually contain confounded sibling pairs among its scored edges*:
its KPI→KPI candidate/target pairs are temporally independent by design, so their FPs are a
calibration artifact, not confounding. Realizing the typed-truth reframe therefore requires a
**new E2 construction** (contemporaneous siblings of a latent previous-state parent), built in a
**two-regime ablation** (latent vs observed shared parent) that *proves* the ↔ edges are genuine
confounding rather than relabeled errors. That construction change is what this document
pre-registers, and it is what keeps the reframe defensible against a goalpost-moving accusation.

---

## 1. Motivation & the structural argument

### 1.1 What the current method does and where it fails
- Truth `_TRUE_ADJACENCY` (`cdd_oran/envs/v2/e2.py:71-78`) has **16 directed P→KPI edges and
  ZERO KPI→KPI edges**. `true_adj_matrix()` (`e2.py:130-135`) writes only those P→K entries.
- Recovery scoring (`cdd_oran/e2slice/evaluate.py:91-94`) therefore counts **every** selected
  KPI→KPI edge as a false positive: `kpi_kpi_fp = int(mask[:, num_params:].sum())`, and
  `recovery_metrics.py:43-49` scores the KPI parent-columns against an all-zero truth block.
- Seed-0 result (`reports/2026-09-06-e2-pdcor-seed0-baseline.md`): P→KPI **P 0.929 / R 0.812 /
  F1 0.867** (13/16 edges, 1 FP) — the planner-relevant part is already good. KPI→KPI: **17 FP /
  36, 0 TP, rejection 0.528** (~9× nominal FDR). The failure is isolated to the Z-dependent
  lagged-KPI candidates.

### 1.2 Why a directed KPI→KPI *non-edge* is not identifiable here (study evidence)
The truth-free calibration study (`runs/calib-study/VERDICT2.md`, numbers in
`report2_numbers.txt`) modeled a confounded null where candidate and target both depend on a
shared latent `U`:
- **Even a proper CI test cannot calibrate under near-determinism.** RCoT (the best of the three
  schemes) stays FP ≈ 0.37–0.51 at n=400 and *worsens to 0.80–0.83 at n=1000* on the true null
  where `U` is fully in `Z` (`VERDICT2.md:16-20`; `report2_numbers.txt:19,99`). Rising FP with n
  is the signature of genuine residual dependence, not a small-sample artifact.
- **When the confounder is missing from the conditioning set, no CI test recovers it:** all three
  schemes reject at 0.64–1.00 on the E2-realistic "proxy" set (`VERDICT2.md:22-30`;
  `report2_numbers.txt:20,100`). More RFF features barely help (Dz sweep 0.61→0.47,
  `report2_numbers.txt:123-133`).
- Marginal permutation (the frozen method's null) is the worst: FP 0.76–0.87 → 1.00 at n=1000
  (`report2_numbers.txt:11-12,91-92`), because permuting the candidate destroys its dependence on
  the conditioning set, sampling *joint* not *conditional* independence
  (`reports/2026-09-06-e2-improvement-synthesis.md:41-44`).

**Structural reading (knowable before any recovery number):** where two variables are both
deterministic functions of a shared latent parent, they are genuinely conditionally dependent
given only the observed columns. A method that reports that association is arithmetically correct
about the observable Markov structure; scoring it as a directed FP against a fully-observed DAG is
scoring the wrong object. This is the mis-posed-target claim. **The fix is to re-type the truth,
not to swap the estimator** — the study shows the estimator swap (RCoT) only halves, never
eliminates, the over-selection.

---

## 2. ⚠️ PIVOTAL CORRECTION — the current construction does NOT contain confounded siblings

This is the single most important finding for keeping the reframe honest, and it *diverges from
the agreed framing*. The agreed plan describes the scored KPI→KPI pairs as "two sibling KPIs both
deterministic functions of an unobserved previous-state parent." **The code does not build that
for the scored pairs.**

From `cdd_oran/e2slice/dataset.py` (docstring `:14-20`, roll loop `:148-153`):
```
x_params[m] = P^m           # committed current params  (target's true parents)
x_kpis[m]   = f(P^{m-1})    # LAGGED KPIs  (the 6 KPI→KPI candidates)
y_kpis[m]   = f(P^m)        # target next-step KPIs
```
Params are drawn **mutually independently every step** (`dataset.py:133-137, inject()`), so
`P^{m-1} ⊥ P^m`. Therefore:
- A scored KPI→KPI edge pairs a **lagged** candidate `K_i = f(P^{m-1})` with a **next-step** target
  `K_j = f(P^m)`. These are driven by *independent* param vectors → **genuinely independent →
  true-negatives by temporal independence**, exactly as the dataset docstring states
  (`dataset.py:19-20`) and Plan 007 §5 asserts (`plans/007-e2-discovery.md:113-116`).
- Shared-latent structure **does** exist — but only *among the 6 lagged candidates themselves*
  (all are `f(P^{m-1})`). It never links a candidate to the target. The recovery scorer only ever
  scores candidate→target pairs, none of which are confounded siblings.

**Consequences:**
1. The current KPI→KPI FPs are a **marginal-permutation calibration artifact**, *not* genuine
   confounding (this matches the synthesis root cause, `improvement-synthesis.md:35-58`). We
   therefore **must not** relabel seed-0's 17 FPs as "true ↔ detections" — that would be the
   goalpost-moving move the discipline forbids.
2. Realizing the typed-truth reframe requires a **new construction** in which the candidate and
   target KPIs are **contemporaneous siblings of a shared previous-state parent** — i.e. score
   `(K_i_t, K_j_t)` pairs, both `= f(P^{t-1})`, with `P^{t-1}` **latent**. Only then is a detected
   KPI↔KPI association a *true* confounded pair.
3. This is a **benchmark-construction decision, not a metric relabel.** It is also what makes the
   reframe reviewer-proof: the ↔ edges are confounded *by construction*, and the two-regime
   ablation (§5) proves it.

> **Sign-off required (Knob 0, the pivotal one):** approve building a new contemporaneous-sibling /
> latent-shared-parent E2 construction, rather than re-typing the frozen lagged (6,14) layout.
> Everything below assumes yes.

A secondary code consequence: the frozen method emits **directed** lagged→next edges from the
temporal ordering (`plans/007-e2-discovery.md:110-112`); it has no way to emit an undirected /
bidirected sibling association. Metric B (§4) needs an **undirected** sibling-association output,
so the discovery layout must expose a symmetric KPI–KPI query. This is a new output channel, not a
rescoring of the existing directed mask.

---

## 3. The new typed truth object (formal definition)

Let `P = {P0..P7}` be the manipulable control parameters and `K = {K0..K5}` the KPIs. The redesign
defines truth as a **typed graph** with two disjoint edge classes, both derivable from
construction metadata **before** any recovery is scored:

### 3.1 Manipulable directed edges `E→ ⊆ P × K`
`E→ = _TRUE_ADJACENCY` restricted to param sources (`e2.py:71-78`, all 16 entries have
`source_index < num_params`). Unchanged from today. `P_a → K_j ∈ E→` iff `P_a` appears in the
Gaussian-bump mechanism for `K_j` (`e2.py:110-121`).

### 3.2 Confounded sibling pairs `S ⊆ (K choose 2)` (bidirected ↔)
A **sibling pair** `{K_i, K_j}` is in `S` iff `K_i` and `K_j` share at least one common
previous-state parent that is **latent in the scored regime**. Derived purely from adjacency
metadata:
```
parents(K_j) = { source : (j, source) ∈ _TRUE_ADJACENCY }
S = { {K_i, K_j} : i < j,  parents(K_i) ∩ parents_latent ≠ ∅
                            and  parents(K_j) ∩ parents_latent ≠ ∅
                            and  a common latent parent is shared }
```
where `parents_latent` is the set of parents held **latent** in a given regime (§5). Under the
current adjacency, the shared parents are the control params (e.g. `P0` parents
`{K0,K1,K2,K5}` via edges `(0,0),(1,0),(2,0),(5,0)` in `e2.py:71-78`); in the redesigned
construction the shared parent is the latent previous-state vector `P^{t-1}` driving the
contemporaneous siblings. **`S` is fixed by construction metadata, not by any observed FP count**
— this is the anti-p-hacking anchor (§7).

### 3.3 What counts as what
- **Manipulable (directed):** a claimed `P_a → K_j`. Scored in Metric A.
- **Confounded (bidirected):** a claimed *undirected* association `K_i — K_j`. Scored in Metric B
  against `S`.
- **Error, always:** a claimed *directed* `K_i → K_j`. A directed KPI→KPI claim is never correct
  (there is no KPI→KPI causal edge in any regime) and is counted as an error in Metric B
  regardless of whether `{K_i,K_j} ∈ S`.

---

## 4. The two metrics + exact scoring rules

Both replace the single "any KPI→KPI = FP" rule at `evaluate.py:91-94`.

### 4.1 Metric A — Manipulable P→KPI directed recovery (planner-relevant)
Exactly today's `ncp_kpi` block (`recovery_metrics.py:48`, parent columns `< num_params`).
- **TP:** `P_a → K_j` selected and `∈ E→`.
- **FP:** `P_a → K_j` selected and `∉ E→`.
- **FN:** `P_a → K_j ∈ E→` and not selected.
- Report precision / recall / F1 per seed + envelope. This is the number the MBP layer consumes
  (§6); target ≈ today's P 0.93.
- **Mapping:** no change to `recovery_by_edge_type`'s NCP block; keep it verbatim.

### 4.2 Metric B — Confounded-sibling detection (correct sign)
Operates on the method's **undirected** KPI–KPI association output (§2, new channel), scored
against `S` (§3.2):
- **TP:** undirected `K_i — K_j` selected and `{K_i,K_j} ∈ S`.
- **FP:** undirected `K_i — K_j` selected and `{K_i,K_j} ∉ S` (a non-sibling pair — genuinely
  independent, so this is a real error).
- **FN:** `{K_i,K_j} ∈ S` and not selected.
- **DIRECTED-CLAIM ERROR:** any directed `K_i → K_j` is tallied separately as `directed_kpi_errors`
  and always counts against the method (never a TP).
- Report sibling precision / recall / F1, plus `directed_kpi_errors`, per seed + envelope.
- **Mapping:** replaces `evaluate.py:91-94`. The old `kpi_kpi_fp = mask[:, num_params:].sum()` and
  `rejection_rate = 1 - FP/36` are **retired** (they encode the all-negatives assumption). The new
  code loads `S` from the frozen truth metadata and computes the four tallies above. Keep the old
  fields in the *retired-baseline* record only (§8.5), never in the new record.

> The two-metric split maps cleanly onto the existing `recovery_by_edge_type` structure
> (`recovery_metrics.py:25-64`): Metric A = its `ncp_kpi` block unchanged; Metric B = a new
> `sibling` block that replaces the meaning of its `kpi_kpi` block.

---

## 5. Two-regime ablation + noise axes (the reviewer-proof validation)

Build the redesigned E2 in **two regimes**, differing only in what is observable:

### 5.1 Regime L (parent LATENT) — ↔ should appear
The shared previous-state parent driving the contemporaneous siblings is **excluded** from the
observed columns. Prediction: a proper association test emits sibling ↔ edges for `{K_i,K_j} ∈ S`
(Metric B recall high), because the confounding is real and unremovable. This is the *positive*
demonstration that near-determinism defeats CI identifiability.

### 5.2 Regime O (parent OBSERVED, ± bounded noise) — ↔ should vanish
Add the shared parent (the previous-state control params) to the observed columns as
conditioners/candidates. Prediction: conditioning on the now-observed parent **removes** the
sibling association → ↔ edges vanish, and a clean directed P→KPI DAG is recovered (Metric A high,
Metric B TP→0, no directed errors). **This is the proof the ↔ in Regime L were genuine confounding,
not relabeled calibration errors** — a relabel could not be made to disappear by observing a
parent.

### 5.3 The real benchmark-defect fix (item 3) — add observable lagged control-params
A real O-RAN RIC logs its own past actions (E2SM-RC / E2SM-CCC control acknowledgements), so the
**previous-step control parameters are observable** and their omission from the candidate set was a
benchmark artifact. Add the observable **lagged params** `P^{t-1}` (or the subset a real RIC logs)
as conditioners/candidates in both regimes. The **truly-internal RAN state** stays latent → honest
residual ↔ in Regime L. (Note per §2: in the *current* construction, adding `P^{m-1}` also directly
removes the marginal-permutation miscalibration among the lagged candidates, so this fix is
independently justified.)

### 5.4 Noise axes
- **Primary E2 stays NOISELESS.** `obs_noise_scale = 0.0` is currently enforced
  (`dataset.py:87-91`, and `E2DatasetConfig.obs_noise_scale = 0.0` default). Keep the primary run
  noiseless: it is the clean demonstration that near-determinism defeats CI identifiability — itself
  a positive result, matching the study (`VERDICT2.md:52-53`: noise never restores calibration
  because confounding lives in the signal).
- **Noisy VARIANT (robustness axis).** Provide a bounded-observation-noise variant, `σ`
  pre-committed and swept, sold as **realism/robustness and to make CI orientation well-posed in
  Regime O** — *not* as the fix. This requires relaxing the `obs_noise_scale != 0.0` guard
  (`dataset.py:87-91`); `E2V2Env` already accepts `obs_noise_scale` (`e2.py:91,97`), so only the
  dataset-config validation changes.
- **σ justification approach (pre-committed, not tuned):** derive `σ` from a physical/SNR argument
  — e.g. a target measurement SNR for O-RAN KPI counters (PRB utilization, throughput, BLER) mapped
  through each KPI's frozen std (`KPI_MEAN_STD`, `e2.py:41-48`) — and sweep a small pre-declared
  grid (e.g. the study's `{0.01, 0.05, 0.1, 0.25}` on standardized KPIs, `report2_numbers.txt`),
  fixing the primary at `σ=0`. The exact `σ` and its source are **Knob 1** (§9).

### 5.5 Ablation GATE (must pass before proceeding — truth-free where possible)
Before scoring any real recovery number, the ablation must show the qualitative signature:
- Regime L: sibling ↔ present for `S`, absent for non-sibling pairs.
- Regime O: sibling ↔ vanish; directed P→KPI DAG clean.
If Regime O does **not** clear the ↔ (i.e. observing the parent fails to remove the association),
the confounding claim is unproven → **STOP and diagnose** (do not proceed to headline scoring).
This gate is the reviewer-proof core; it is largely a construction-consistency check and can be run
on the synthetic regimes before touching the fresh-seed recovery numbers.

---

## 6. Planner anchor (why it matters / why we're fine)

The MBP layer consumes **only directed P→KPI edges** (Metric A). Those recover cleanly (P≈0.93
today, expected to *improve* in Regime O). The planner never consumes KPI→KPI structure, so it is
**robust to the sibling confounding by construction** — the ↔ edges are a discovery-honesty result,
not a planner input. This is the "why we're fine" story: the confounding lives entirely in a channel
the planner does not use.

---

## 7. Method plan

- **RCoT = fast CI oracle only.** RCoT is retained (a) as the fast CI test that makes fresh-seed
  generation feasible (analytic null, linear-N, minutes vs the pdCor ~8.65 h/seed,
  `pdcor-seed0-baseline.md:84-88`), and (b) as evidence that *even a proper CI test cannot calibrate
  under noiselessness* (`VERDICT2.md:16-20`). **RCoT is explicitly NOT the headline** and NOT sold
  as the fix — the study shows it only halves the over-selection.
- **FCI / RFCI / GFCI PAG recovery = OPTIONAL stretch goal**, not a dependency. A PAG naturally
  represents bidirected ↔ edges and would be the "proper" typed-structure learner, but the two
  metrics (§4) + the ablation (§5) stand without it. Pursuing it adds a dependency
  (`causal-learn`/`pgmpy`) — **Knob 3** (§9).
- Retain the retired pdCor mask as the documented naive-DAG baseline (§8.5) for a before/after.

---

## 8. Pre-registration protocol, tracked home & timeline

### 8.1 Tracked home for the FROZEN version (identified)
- `reports/` is **gitignored** (`.gitignore:37`; `git check-ignore reports/x.md` → ignored). This
  DRAFT lives here for review only.
- `docs/benchmark/` is **tracked** (`E2_DISCOVERY_PROTOCOL.md`, `GATE_CONTRACT_E2.md` are committed
  there; `git check-ignore docs/benchmark/x.md` → not ignored).
- **Frozen protocol → new tracked doc `docs/benchmark/E2_TYPED_DISCOVERY_PROTOCOL.md`** (companion
  to the existing `E2_DISCOVERY_PROTOCOL.md`, which is left byte-identical), plus a **new plan
  `plans/008-e2-typed-latent-confounding.md`** (next number after `plans/007`; `plans/` is tracked)
  registered in `plans/README.md`. The as-run result goes to a new
  `docs/benchmark/E2_TYPED_DISCOVERY_RESULT.md`. The dated delta report stays in `reports/`.

### 8.2 Freeze / commit-before-scoring ordering (mandatory, mirrors Plan 007 §"Freeze ordering")
1. Finalize this document → author `docs/benchmark/E2_TYPED_DISCOVERY_PROTOCOL.md` as the executable
   spec (typed truth `S`, both metrics, both regimes, σ grid, method) in **structural terms only —
   no reference to any observed recovery number or the seed-0 FP count**. Review, then **commit it.
   That commit SHA is the freeze**, recorded as `protocol_commit` in every new artifact.
2. Implement: new construction (two regimes) + `S`-derivation from adjacency metadata + the
   two-metric scorer + the undirected sibling channel + fail-closed artifacts (Plan 003 pattern,
   as in `discovery.py:440-548` / `evaluate.py:121-165`). Discovery reads no truth; `S` and the
   metrics read truth only in the post-freeze scorer.
3. Run the **truth-free ablation gate** (§5.5) on the synthetic regimes. It must show the
   latent→↔ / observed→clean signature before any recovery scoring.
4. Generate **fresh seeds** and run discovery, persisting + content-hashing every mask FIRST.
5. **Only then** score Metric A + Metric B against the frozen typed truth into a separate record.

### 8.3 Fresh-seed confirmation (feasible only because the new method is fast)
- **Only seed-0 was ever discovered.** Seeds 1–2 failed to launch (`rc=127`), seeds 3–9 never
  existed (`pdcor-seed0-baseline.md:94-102`). So seed-0's KPI→KPI numbers are the *only* ones ever
  seen.
- Confirm the new typed metrics on **fresh, previously-unscored seeds** generated and scored with
  the fast method (number = **Knob 2**, §9; Plan 007 §6 registered 10). This is feasible precisely
  because RCoT is minutes/seed, not ~8.65 h/seed.
- **Seed-0 is DEMOTED to ILLUSTRATIVE** — used only to narrate the mechanism, never as a headline
  number, because its FP counts were seen before the reframe.

### 8.4 Anti-p-hacking discipline (central)
The reframe is defensible iff **all** of:
- (i) The criterion `S` is grounded strictly on **construction/adjacency facts** (shared-parent
  intersection, §3.2) — derivable before any recovery number — **not** on the observed FP count.
  §2 makes this stronger: we are *not* relabeling seed-0's FPs; we build a new confounded-by-design
  construction whose ↔ pairs are fixed by metadata.
- (ii) The new protocol is **frozen (doc + commit) BEFORE any recovery scoring** (§8.2 step 1).
- (iii) Headline confirmation is on **fresh seeds** (§8.3).
- (iv) Seed-0 is **illustrative only** (§8.3).
- (v) The retired pdCor seed-0 mask is kept as the documented **naive-DAG baseline** (§8.5) for an
  honest before/after.

### 8.5 Retired pdCor baseline (kept)
`reports/2026-09-06-e2-pdcor-seed0-baseline.md` and the frozen
`docs/benchmark/E2_DISCOVERY_PROTOCOL.md` (`protocol_commit 828e345`) stay on disk unchanged as the
naive fully-observed-DAG baseline: P→KPI P 0.929, KPI→KPI 17 FP/36, rejection 0.528. The new result
doc cites these as "before."

### 8.6 Timeline (indicative)
1. Human signs off Knobs 0–4 (§9). → 2. Author + review + **commit** frozen protocol. → 3.
Implement construction + scorer + tests (gates: `ruff`, `ty`, `pytest`, fail-closed load). → 4.
Truth-free ablation gate (§5.5). → 5. Fresh-seed generate + discover + persist/hash. → 6. Score
both metrics. → 7. Result doc + delta report + merge to `feat/v2`.

---

## 9. Open questions / knobs requiring the human's sign-off

- **Knob 0 (PIVOTAL):** Approve the **construction change** — build a new contemporaneous-sibling /
  latent-shared-parent E2 (two regimes), rather than re-typing the frozen lagged (6,14) layout
  (§2)? Everything depends on this.
- **Knob 1 — σ value & source:** the exact bounded observation-noise `σ` for the noisy variant and
  its physical/SNR justification, plus the pre-declared sweep grid; primary stays `σ=0` (§5.4).
  Requires relaxing `dataset.py:87-91`.
- **Knob 2 — number of fresh seeds** to generate + score with the fast method for headline
  confirmation (Plan 007 registered 10; only seed-0 ever ran) (§8.3).
- **Knob 3 — FCI/RFCI/GFCI PAG stretch:** pursue the proper typed-structure learner (adds a
  `causal-learn`/`pgmpy` dependency), or stop at the two metrics + ablation (§7)?
- **Knob 4 — exact lagged-param columns to add** as observable conditioners: which components of
  `P^{t-1}` a real O-RAN RIC logs via E2SM-RC / E2SM-CCC, i.e. which stay observable vs latent
  (§5.3). This choice defines `parents_latent` and hence `S`.

---

## 10. What could still sink this (risks)

1. **Regime O fails to clear the ↔.** If observing the shared parent does *not* remove the sibling
   association (e.g. residual determinism, imperfect conditioning), the confounding claim is
   unproven and the ablation gate (§5.5) STOPs the work. This is the load-bearing risk.
2. **Goalpost-moving accusation persists.** Even with §2's honest construction change, a reviewer
   may argue the redesign was reverse-engineered from seed-0's failure. Mitigation: the ↔ criterion
   `S` is a pure function of adjacency metadata frozen before scoring, and headline numbers are
   fresh-seed only. The DRAFT records the seen seed-0 numbers explicitly so nothing is hidden.
3. **Noiselessness defeats the oracle in Regime O too.** The study shows even RCoT worsens with n
   on the confounded null (`VERDICT2.md:16-20`). If Regime O's CI orientation is not well-posed
   even with the parent observed, the noisy variant (Knob 1) becomes load-bearing for orientation —
   which risks the "noise as fix" optics we are trying to avoid. Mitigation: keep the noiseless
   primary as the *identifiability-limit* result and frame noise strictly as orientation-enabling.
4. **Directed-vs-undirected output mismatch.** The frozen method emits directed lagged→next edges
   only (§2). Metric B needs a symmetric sibling channel; if the new discovery layout cannot cleanly
   emit undirected associations, Metric B is ill-defined. Mitigation: design the sibling channel
   explicitly in step 2 (§8.2), pre-frozen.
5. **`S` is trivially small under the current adjacency.** Shared-parent siblings via `P0` give a
   modest `S`; if the redesigned construction yields too few sibling pairs, Metric B has low power.
   Mitigation: choose the latent-parent structure (Knob 4) so `S` is non-trivial and pre-declared.
6. **Scope creep into the E2 decision gate.** The E2 *decision* null (`GATE_CONTRACT_E2.md`, commit
   `8d70a4f`) is FAIL and stays untouched (Plan 007 firewall, `plans/007-e2-discovery.md:38-44`).
   This redesign is discovery-method work only; it must not be presented as E2 decision-value
   evidence.

---

*End of DRAFT. Do not freeze or commit this file. The frozen protocol belongs at
`docs/benchmark/E2_TYPED_DISCOVERY_PROTOCOL.md` + `plans/008-*.md`, committed before any scoring.*
