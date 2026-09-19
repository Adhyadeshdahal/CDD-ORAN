# E5 as an effect-quantification problem — re-examining "effect estimation is the wrong family" (2026-09-06)

Research lane 3 of 3, read-only. Reframes the END GOAL **E5** around the layer a prior pass
(`reports/2026-09-06-causal-upgrade-synthesis.md`) dismissed: **effect estimation / intervention-effect
quantification** over a discovered structure. Cites specs and code; changes no frozen code.

**One-line answer.** The prior dismissal was correct *for the task it was judging* (structure **discovery**
on E2 — replacing RCoT) and **wrong for E5**. E5's "globally harmful" is definitionally a statement about
the **magnitude and sign of a shared knob's effect propagated across its complete causal fan-out and
aggregated into one objective** — i.e. exactly an interventional-effect-quantification problem layered on the
discovered graph. Effect quantification is a **first-class part of the E5 solution**, not a wrong turn. But
the *specific tools* the user linked (Tencent, RAPIDS/DoubleML) remain a poor fit for a different reason than
the prior pass gave — not "wrong family," but "right family, wrong regime" (they are big-data ATE scalers;
E5 is a small, deterministic-to-mildly-noisy, known-mechanism SCM). The genuinely relevant effect-estimation
machinery is **DoWhy-GCM-style interventional propagation** (E2/E3/E5) and **DML/deconfounding** (E4 only).

---

## 1. How E5 defines local-vs-global harm and the shared-knob decision

E5 is the **composition capstone** (`SPEC.md:556-599`): it introduces **no new mechanism**, composing the
three already-validated axes plus moderate observation noise:

- a **shared-control fan-out block** from E2 (`P_shared -> {multiple xApps}`, `SPEC.md:566`);
- a **temporal KPI chain** from E3 (`K -> K -> K` at one-step lag, `SPEC.md:567`);
- a **confounding block** from E4 (`Z -> A_behavior`, `Z -> K_out`, `A -> K_out`, `SPEC.md:568`).

Its **primary metric is aggregate AND worst-case decision regret** over the composed decision population, ID
and OOD (`SPEC.md:586-588`). Success = "discovered ≈ oracle with bounded aggregate and worst-case regret ≪
combined decoy, graceful ID→OOD degradation" (`SPEC.md:590-593`).

The **local-vs-global-harm mechanism E5 inherits is defined most sharply by E2** (`SPEC.md:215-285`,
`GATE_CONTRACT_E2.md`), whose scientific hypothesis is verbatim the END-GOAL question:

> "When one shared control knob fans out to multiple xApps, a model that reasons over the **complete** causal
> fan-out avoids a locally attractive single-xApp action that is globally harmful, whereas a model that sees
> only an **incomplete** fan-out takes it." (`SPEC.md:217-221`)

The decision is **fully specified as an effect-magnitude/sign question**:

- **The shared knob** is `P0`, which fans out to `{K0,K1,K2,K5}` (`GATE_CONTRACT_E2.md:62-64`). "Locally
  attractive" = moving `P0` improves the xApps on `K0/K1/K2`; "globally harmful" = the **same** `P0` move
  pushes `K5` (satisfy-**below** xApp4, `directions=[...,1]`, `GATE_CONTRACT_E2.md:40`) up past its
  threshold, a harm the local view never prices.
- **The decoy is precisely an effect-estimation error.** It omits **exactly one fan-out edge `P0->K5`**
  (`SPEC.md:243-245`, `GATE_CONTRACT_E2.md:69-74`) and therefore **mispredicts the effect of `do(P0=v)` on
  `K5` as zero**. It "picks the locally attractive `P0` move; the complete-fan-out oracle predicts the `K5`
  harm and avoids it" (`SPEC.md:245-246`). The trap is a *quantitative fan-out prediction* failure, not a
  discovery failure — the decoy and oracle score the **identical** panel; only the predicted `K5` effect
  differs (`GATE_CONTRACT_E2.md:69-74`, matched decision population `SPEC.md:247-252`).
- **"Global" is an aggregation over the fan-out.** The decision is scored by the locked objective `R`
  (`SEMANTICS.md:198-239`, `GATE_CONTRACT_E2.md:130-141`): a hinge cost summing per-xApp standardized
  satisfaction over **every** xApp sharing the knob,
  `cost = Σ_i w_i·distance_i·s − (Σ_i ok_i)²`, `R = −cost`, select `argmax_v R`. "Globally harmful" is
  literally: the `v` that maximizes the *incomplete-fan-out* `R` is not the `v` that maximizes the
  *complete-fan-out* `R`, because the omitted `P0->K5` term flips the aggregate.
- **The gate measures a normalized effect gap.** Positive/negative controls are labeled purely by
  `K5`-response geometry — `z_min <= θ5-0.5 ∧ z_max >= θ5+0.5` (a state where `do(P0)` *realizes* `K5` harm)
  vs `S5(s) <= 0.05` (`do(P0)` inert on `K5`) (`GATE_CONTRACT_E2.md:84-99`). PASS iff
  `mean_positive(gap_norm) >= 0.10 ∧ mean_negative(gap_norm) <= 0.01` (`GATE_CONTRACT_E2.md:122`), where
  `gap_norm` is the decoy-vs-oracle realized-regret difference over the TRUE action-return range `D(s)`.

**E3 sharpens the same shape across time**: an immediate `K0` benefit propagates through `K0->K1->{K2,K3}`
into a **larger delayed combined loss** on two monitored services (`SPEC.md:289-318`,
`GATE_CONTRACT_E3.md:30-51`). Here "globally harmful" = the **cumulative** return over the delayed window is
worse even though the immediate step looks good — a fan-out that is **temporal** (down the chain) as well as
**cross-sectional**. The oracle is the finite-horizon `H=3` cumulative-return maximizer
(`GATE_CONTRACT_E3.md:72-85`).

**Net:** across E2/E3/E5 the shared-knob decision is *identical in kind* — enumerate `v` on the grid, for each
`v` **propagate the intervention `do(P0=v)` through the complete fan-out** (cross-sectional in E2, temporal in
E3, both + confounding in E5), **aggregate into `R`**, take `argmax`. The whole benchmark is an
effect-quantification-for-decision test wearing a discovery hat.

---

## 2. Effect-quantification methods that fit E5 — and an honest re-verdict on the tools

### 2.1 What the codebase already does (and its gap)

The **oracle world-model already IS complete-fan-out effect propagation** — on the TRUE simulator.
`cdd_oran/benchmark/rollout.py` (`enumerate_open_loop` / `paired_regret`) rolls each candidate action
sequence forward on an independent TRUE-SCM clone, scores the SEMANTICS §1.2 window, and returns cumulative
regret vs the planner. That *is* `do(P0=v)` propagated through every downstream KPI and KPI→KPI chain and
aggregated into `R` — the reference answer E5 needs. The scientific question is whether a **learned /
discovered** model can reproduce it.

The **learned** side (`cdd_oran/models/cdl.py`, scored by `cdd_oran/planners/qacm.py:70`) does
effect quantification only **weakly**:

- CDL is a **one-step structure-conditioned next-state predictor** (`predict_next_state`,
  `cdl.py:640-688`): per-KPI-child MLPs gated by a sampled structure. QACM scores each candidate `(bin,index)`
  by a **single** `predict_next_state` call and aggregating xApp cost (`qacm.py:62-113`). That is a **one-step
  fan-out effect estimate under a learned mechanism** — adequate for E2 (`H=1`) but it does **not** propagate
  through the E3/E5 KPI→KPI chain (no multi-step rollout of predicted KPIs fed forward).
- The known multi-intervention gap (`memory/cma-es-gate-falsified.md`): the shipped joint planner applies
  block NCPs **sequentially, never overwriting param slots** because "the trained model was fit on
  SINGLE-param interventions, so joint multi-param actions are OOD." The **+4.82 coordination win is
  oracle-only**; on the learned model, joint multi-param **effect prediction is OOD**. The memory's own verdict:
  *"the coordination lever is a **world-model capability** (predict joint multi-param interventions), NOT the
  planner optimiser… the lever is causal-model/discovery quality, not the search algorithm."* This is the
  effect-quantification layer named explicitly as the bottleneck.
- `memory/cid-planner-reviewed.md`: the "causal influence-diagram planner" is **deterministically equal to
  QACM today** — it adds value only once **graph-pruning or joint queries** land. Both are effect-quantification
  upgrades (which params to intervene on; how joint interventions propagate), not search upgrades.

**So the E5 effects layer = "turn the one-step learned predictor into a faithful interventional
world-model that propagates `do(P0=v)` (and joint / multi-step / deconfounded interventions) through the
discovered fan-out and aggregates into `R`."** This is the world-model the planner memory keeps pointing at.

### 2.2 The method families and where each fits

| Method family | What it computes | Fits which E-env | Why |
|---|---|---|---|
| **Structural interventional propagation** (do-operator simulated on a fitted SCM: fit each `f_child`, then roll `do(P0=v)` forward through the graph, chaining predicted KPIs) | `E[K_j \| do(P0=v)]` for every downstream `j`, incl. through KPI→KPI chains | **E2, E3, E5 (core)** | Mechanisms are **known-form & (near-)deterministic**; direction is **known by time order**; causal sufficiency holds for the observed params. The effect is *propagated*, not statistically *estimated from confounded data*. This is exactly `rollout.py` but on a **learned** `f`. |
| **do-calculus / interventional distribution** (`P(K \| do(A))`) | identified interventional query | **E4 (needed), E2/E3/E5 (trivially identified)** | In E2/E3 params are exogenous roots → `P(K\|do(P))=P(K\|P)`, identification is free. E4 is the one env where `P(K_out\|do(A))` is **not** identified from observational data (`SPEC.md:398-403`); identification comes from the randomized-interventional subset. |
| **ATE / interventional-mean reduction** | `E_Z[K_out \| do(A=a)]` scalar per action | **E4 (the built score), E2 aggregate** | E4 literally scores decisions on `K_score(a)=E_Z[K_out\|do(A=a)]=-a` (`SPEC.md:531`, `GATE_CONTRACT_E4.md`). E2's `R` is an aggregate of per-knob interventional effects. |
| **CATE / conditional effects** | `E[K \| do(A), X=x]` (state-conditional) | **E3/E5 (mild), E4 (secondary)** | The effect of `do(P0)` is state-dependent (Gaussian-bump mechanisms), so per-state (per-`s`) effects matter — but here `X` is the fully-observed committed state, so this is "evaluate `f` at `s`," not a heterogeneity-estimation problem. |
| **Double/Debiased ML (DML)** | de-biased ATE/CATE under high-dim nuisance confounding | **E4 ONLY (and even there, marginal)** | DML's whole reason for existing is to **partial out confounders** with flexible ML nuisance models. E2/E3 have **no confounder** — DML solves a problem they don't have. E4 **does** have `Z`, but `Z` is *latent* and E4's identification is designed to come from the **randomized subset**, not from adjusting on observables — so DML's backdoor-adjustment machinery is only partially on-target. |

**Bottom line on methods:** the E5 core is **structural interventional propagation on a fitted/discovered
SCM** (rung-2 do-queries simulated through the graph) — *not* observational treatment-effect estimation.
Treatment-effect estimators (DML/DR/IV) are relevant **only at E4's confounded sub-block**, and even there the
design leans on the interventional subset rather than pure deconfounding.

### 2.3 Honest re-verdict on the specific tools

**The prior pass's framing was E2-discovery-myopic.** It judged the tools solely as *structure-discovery*
candidates and said "effect estimation… NOT structure discovery — wrong tool family"
(`reports/2026-09-06-causal-upgrade-synthesis.md:59-62`, and the systems lane). That verdict is right *for
replacing RCoT* and beside the point *for E5's effects layer*. Re-judged as **effect-quantification**
candidates for E5:

- **NVIDIA RAPIDS + DoubleML (the user's link).** Confirmed: it is **DML (`DoubleMLPLR`) estimating an ATE
  for a single treatment**, where **cuML GPU-accelerates the nuisance ML models** (RandomForest), ~12× on
  10M rows; it does **no** structure discovery and **no** interventional propagation through a graph
  ([NVIDIA blog](https://developer.nvidia.com/blog/faster-causal-inference-on-large-datasets-with-nvidia-rapids/)).
  **Re-verdict: still not adopted, but for the honest reason** — it is a **big-data confounded-ATE scaler**.
  E5 has ≤~14 nodes, thousands of transitions, **known mechanisms**, and **no observational confounding except
  E4**. DML earns its place *only* on the **E4** sub-block (recover the true `A->K_out` slope `α=-1` against
  the `Z`-induced pooled slope `+3.29`, `GATE_CONTRACT_E4.md`), and even there the identification is designed
  to come from the randomized subset. The GPU angle is irrelevant at this scale. So: **relevant in kind
  (effect estimation is first-class), not relevant in tool** — the datasets are too small and mostly
  unconfounded for DML-at-SQL-scale to buy anything. This is a *sharper* dismissal than "wrong family."

- **Tencent fast-causal-inference (the user's link).** An **OLAP/SQL-scale** causal-inference engine (DML,
  ATE, distributed) for big-data statistical models
  ([repo](https://github.com/Tencent/fast-causal-inference)). **Re-verdict: not relevant to E5** — same
  regime mismatch as RAPIDS, amplified: it targets warehouse-scale SQL analytics, the opposite of a small
  simulator SCM. **Confirmed genuinely off-target**, but again for the regime reason, not because effect
  estimation is the wrong idea.

- **DoWhy / DoWhy-GCM + EconML (NOT in the user's links, but the right tools if any).** This is the honest
  correction the prior pass missed. **DoWhy-GCM fits a structural causal model to data and answers rung-2
  interventional (`do`) and rung-3 counterfactual queries, propagating an intervention through the graph to
  downstream nodes**, plus attribution
  ([DoWhy what-if docs](https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/what_if/index.html),
  [DoWhy-GCM paper arXiv:2206.06821](https://arxiv.org/pdf/2206.06821)). That is **precisely E5's fan-out
  quantification** — it is `rollout.py`'s propagation generalized to a *fitted* SCM. **EconML** supplies the
  DML/DR/IV/meta-learner estimators for E4, callable through DoWhy's model→identify→estimate→**refute**
  pipeline; the **refutation** methods (placebo treatment, add-random-common-cause, dummy-outcome) map cleanly
  onto our **factor-removal controls** (E2 drop-`K5`, E4 `λ=0`) as an off-the-shelf validation vocabulary
  ([DoWhy docs](https://www.pywhy.org/dowhy/v0.8/)). **Re-verdict: genuinely relevant as a design reference /
  possible adoption**, especially DoWhy-GCM for interventional propagation + refutation. Caveat: our
  simulator gives us the mechanisms directly, so we may *implement* the propagation natively (we already do,
  in `rollout.py`) and use DoWhy-GCM mainly as (a) the learned-arm fan-out estimator and (b) a
  standard-vocabulary refutation harness — not a wholesale dependency.

**Summary re-verdict:** *Effect estimation is a first-class part of E5* — the prior "wrong turn" conclusion
does not hold at the end goal. *The user's two specific links stay unadopted*, but the correct reason is
**regime** (big-data confounded-ATE scalers vs a small known-mechanism SCM), and their *family* (interventional
effect quantification) is exactly what E5 needs, best represented by **DoWhy-GCM interventional propagation**
(E2/E3/E5) and **EconML/DML deconfounding** (E4).

---

## 3. How structure discovery + effect quantification + the planner compose into E5

The three layers are **necessary-but-each-insufficient** and compose as a pipeline (each already has a seam in
the code):

1. **Structure discovery** (lane 2 / RCoT-v2, `plans/008-009`, `evaluate.py` enum graph → CDL
   `enumeration_graph`, `cdl.py:298-304`) supplies the **fan-out topology**: which KPIs `P0` touches, which
   KPI→KPI edges carry delayed harm. It answers *where the fan-out goes*. It does **not** answer *how much /
   which sign* — a topology with the `P0->K5` edge present but a mispredicted magnitude still picks the harmful
   action. (E2's decoy is a topology hole; E5's harder failure is a *magnitude* hole even with correct
   topology.)

2. **Effect quantification** (the layer this report argues for) turns the topology into **signed, magnitude-
   accurate interventional effects**: for each `v`, `E[K_j | do(P0=v)]` for every downstream `j`, **chained
   through KPI→KPI edges** for the temporal fan-out, **deconfounded** (E4 block) where a latent biases the
   behavior policy. This is the **world-model the planner queries** — currently the weak one-step CDL predictor
   (`cdl.py:640`), the thing `memory/cma-es-gate-falsified.md` and `memory/planner-set-redesign-direction.md`
   both name as the real lever ("world-model multi-intervention capability").

3. **The planner** (QACM baseline / the influence-diagram planner, `qacm.py`, `memory/cid-planner-reviewed.md`)
   **aggregates the fan-out effects into the global objective `R` and takes `argmax_v`**. Given a *faithful*
   effects layer this is nearly trivial (grid argmax over `R`); given an *incomplete* one it takes the locally
   attractive, globally harmful action — which is exactly the decoy. The planner is **not** the lever
   (three memories converge on this); it is the aggregation-and-selection step that a good effects layer makes
   easy.

**The composition, stated as E5's decision rule:**
`argmax_v R( aggregate_over_fanout( propagate_do(P0=v) through discovered_structure ) )`.
Discovery gives `discovered_structure`; effect quantification gives `propagate_do(...)` (the world-model);
`R` and `argmax` are the planner. **The fan-out-quantification IS the world-model the planner needs** — the
report's core claim, and the direct answer to the memory thread's open question.

The **oracle** already executes this with the TRUE `f` (`rollout.py`); the **decoy** executes it with one edge
zeroed; the **proposed arm** must execute it with the **discovered structure + a learned/fitted effects
layer**. E5's regret gap is exactly the gap between a faithful and a crippled effects layer.

---

## 4. What to prototype for the effects / world-model layer

Ordered, truth-free-prototype-first (matches the project's freeze discipline; none of this touches frozen
gate values):

1. **Multi-step interventional rollout of the learned model (highest value, unblocks E3/E5).** Wrap CDL's
   one-step `predict_next_state` in a **chained rollout** that feeds predicted KPIs forward through the
   discovered KPI→KPI edges — i.e. a learned-model analogue of `rollout.py`'s `enumerate_open_loop`. Bar:
   on E3's frozen SCM, the chained-learned fan-out must reproduce the TRUE cumulative-return ranking of `v`
   well enough to recover the FH-vs-FM gap (`GATE_CONTRACT_E3.md:155-174`). This is the single missing piece
   between "one-step E2 predictor" and "temporal-fan-out E3/E5 world-model."

2. **Joint / multi-param intervention prediction (closes the `memory/cma-es-gate-falsified.md` gap).** Train
   or structure CDL so `do(P_i, P_j)` predictions are **in-distribution** (train on joint interventions, or
   compose per-edge structural effects rather than sequential-no-overwrite). Bar: recover a positive
   coordination gap on the true shared-control block **without** the oracle (the +4.82 currently exists only
   under the oracle). This is the "world-model multi-intervention capability" named as the headline lever.

3. **Deconfounded effect estimator for the E4 block (the one place DML/EconML genuinely fit).** Prototype a
   structure-restricted regression (the `D=1`-restricted arm already contemplated, `GATE_CONTRACT_E4.md`) vs
   an **EconML DML/DR learner** on the matched 90:10 obs+interventional corpus; both must recover
   `α=-1` against the pooled `+3.29`. Use it as the E4 "aware" arm's effects layer. Keep it **scoped to E4** —
   do not import DML where there is no confounder.

4. **Refutation/validation harness borrowed from DoWhy's vocabulary (cheap, high-credibility).** Re-express
   the existing factor-removal controls (E2 drop-`K5` → `gap_norm<=0.01`; E4 `λ=0` → gap collapses to 0) as
   **placebo / add-random-common-cause / dummy-outcome refutations**. This gives the effects layer a standard,
   reviewer-legible validation story at ~no cost and reuses machinery we already trust.

5. **(Reference only, do not vendor at this scale) DoWhy-GCM as the fitted-SCM propagator.** Evaluate
   DoWhy-GCM's `interventional_samples` / graph-propagation as the learned-arm world-model for a single env,
   purely to check whether it beats our native chained-rollout on faithfulness; adopt the *pattern* (fit
   `f_child`, simulate `do`, propagate), not necessarily the dependency — our simulator already hands us the
   mechanisms, and `rollout.py` already propagates.

**Non-goals (honest guardrails):** do **not** add benchmark noise to "make DML apply"
(`causal-upgrade-synthesis.md` guardrail); do **not** import Tencent/RAPIDS SQL-scale DML (regime mismatch,
§2.3); do **not** treat the planner/optimizer as the lever (three memories falsify that). The lever is a
**faithful interventional world-model over the discovered fan-out** — which is effect quantification, promoted
to first-class exactly as this lane argues.

---

## Sources

- Specs: `docs/benchmark/SEMANTICS.md` (§1-§3 timing/objective/regret), `docs/benchmark/SPEC.md` (E2 217-285,
  E3 289-386, E4 390-553, E5 556-599), `docs/benchmark/GATE_CONTRACT_E2.md`, `docs/benchmark/GATE_CONTRACT_E3.md`.
- Code: `cdd_oran/benchmark/rollout.py` (oracle fan-out propagation), `cdd_oran/models/cdl.py` (one-step learned
  world-model), `cdd_oran/planners/qacm.py` (candidate scoring / aggregation).
- Memory: `cma-es-gate-falsified.md`, `planner-set-redesign-direction.md`, `cid-planner-reviewed.md`,
  `causal-planner-direction-explored.md` (world-model multi-intervention = the lever).
- Prior pass re-examined: `reports/2026-09-06-causal-upgrade-synthesis.md`.
- External tools: [NVIDIA RAPIDS + DoubleML blog](https://developer.nvidia.com/blog/faster-causal-inference-on-large-datasets-with-nvidia-rapids/),
  [Tencent/fast-causal-inference](https://github.com/Tencent/fast-causal-inference),
  [DoWhy](https://www.pywhy.org/dowhy/v0.8/),
  [DoWhy what-if / GCM interventions](https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/what_if/index.html),
  [DoWhy-GCM paper arXiv:2206.06821](https://arxiv.org/pdf/2206.06821).
</content>
</invoke>
