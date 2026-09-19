# E5 causal representation & the shared-knob decision (LANE 1, 2026-09-06)

Read-only research. No code changed. Grounds the project's end-goal decision problem in the frozen
E-series specs and maps the causal representation + pipeline it demands against what we already have
(the RCoT edge test).

## 0. Framing reconciliation (read first — the labels matter)

The task states the end goal as: *"Can COMPLETE CAUSAL FAN-OUT prevent a locally-attractive-but-
globally-harmful SHARED-KNOB action?"* In the frozen repo specs **that exact sentence is E2's
scientific hypothesis**, not E5's:

- `docs/benchmark/SPEC.md:216-222` — E2 hypothesis: *"When one shared control knob fans out to
  multiple xApps, a model that reasons over the **complete** causal fan-out avoids a locally
  attractive single-xApp action that is globally harmful, whereas a model that sees only an
  **incomplete** fan-out takes it."*
- `cdd_oran/envs/README.md:29` labels E2 verbatim: *"Shared-control conflict — can complete causal
  fan-out prevent a harmful shared-knob action?"*
- `docs/benchmark/SPEC.md:556-599` — **E5 is the composition capstone**: the E2 shared-knob fan-out
  decision *plus* E3's temporal KPI→KPI chain *plus* E4's latent confounding *plus* moderate
  observation noise, scored by aggregate **and worst-case** regret, ID and OOD.

**Reconciliation used throughout this report:** the shared-knob decision *is* the end goal, and E5 is
where it must survive in its hardest form. E2 tests the decision with the simplifications the task
warns against (known time order, causal sufficiency, noiseless deterministic KPIs, KPI→KPI edges that
are all true-negatives). E5 is the same decision after those crutches are removed. So "complete causal
fan-out" for the end goal =
**E2 breadth (one knob → many KPIs) + E3 depth (real multi-hop KPI→KPI chains) + E4 orientation/
deconfounding (sign is not free from time) + robustness to noise.** Judged against E5, E2 recovery is
necessary but nowhere near sufficient.

---

## 1. What E3 / E4 / E5 actually introduce (grounded in the frozen specs)

### E3 — temporal depth, and the first REAL KPI→KPI fan-out
`SPEC.md:289-386`, `docs/benchmark/GATE_CONTRACT_E3.md`.

- **SCM** (`GATE_CONTRACT_E3.md:28-51`, every coefficient exactly 1, no noise):
  `K0(t)=P0(t-1); K1(t)=K0(t-1); K2(t)=P1(t-1)+K1(t-1); K3(t)=P2(t-1)+K1(t-1); K4(t)=P3(t-1)`.
  True graph `P0→K0→K1→{K2,K3}` plus `P1→K2, P2→K3, P3→K4`.
- **The new thing:** genuine within-chain **KPI→KPI edges that fan out** — the internal conduit K1
  fans into two delayed monitored services K2 and K3 (`SPEC.md:308-312`). Each KPI→KPI edge advances
  one step ([SEM] PD2), so the immediate gain at K0 and the delayed combined loss at {K2,K3} are
  separated in time by chain depth. This is exactly the structure **E2 lacks** (E2: *"No KPI→KPI edge
  required"* `SPEC.md:233`).
- **Decision = cumulative regret over the horizon** (`SPEC.md:358-366`), not a single static
  transition. Oracle searches an **open-loop** action sequence over `V^3` (101³ ≈ 1.03M sequences,
  `GATE_CONTRACT_E3.md:72-84`).
- **Trap = 2×2 design** (`SPEC.md:331-347`): full-vs-truncated structure × H=1-myopic-vs-H=depth.
  Primary contrast **FH vs FM** isolates temporal value; **FH vs TH** (drop the fan-out pair
  `{K1→K2, K1→K3}`) isolates *structure*. Decision gate PASS iff `mean(gap_H_norm) ≥ 0.10` AND
  control `mean(|gap_T_norm|) ≤ 0.01` (`GATE_CONTRACT_E3.md:151-174`).
- **Build status:** env `cdd_oran/envs/v2/e3.py` + `scripts/e3_scm_gate.py`/`e3_decision_gate.py`
  exist; **no `e3slice` discovery/decision pipeline** (`envs/README.md:30`).
- **Known implementation prerequisite** (`SPEC.md:368-376`): the named `RecedingHorizonCEM` returns
  only the **first** action and repeats one held action across the horizon
  (`cdd_oran/planners/horizon.py:49-73,115-164`), so it does **not** emit the oracle's `a1..aH`
  open-loop action class. A sequence-emitting open-loop planner must be built before E3 runs.

### E4 — action-relevant confounding (breaks causal sufficiency and sign)
`SPEC.md:390-552`, `docs/benchmark/GATE_CONTRACT_E4.md`.

- **SCM** (`GATE_CONTRACT_E4.md:28-42`): one acted param `A=P0`, one KPI `K_out`, one **latent** `Z`
  on all three edges `Z→A_behavior`, `Z→K_out`, `A→K_out`; `K_out(q+1)=alpha·A + theta·Z + c`,
  `alpha=-1`, `theta=+2.5`. `Z` is never a param/KPI, never exposed to a planner → observed adjacency
  is just `(K0,P0)`; the two Z-edges are latent SCM metadata.
- **The new thing:** the observational association has the **wrong sign**. The naive 90:10 pooled
  line has slope `b_pool = +3.29` (`GATE_CONTRACT_E4.md:100-105`) while the true `alpha = -1` — a
  robust sign reversal, not a knife-edge (`GATE_CONTRACT_E4.md:107-109`). **Time order does not orient
  this** (both A and K_out are same-transition-lagged); identification comes only from the
  **randomized-interventional subset** where action assignment is independent of `Z`
  (`SPEC.md:399-402`, `rho_obs:rho_do = 90:10`).
- **Data model:** obs/do mixture with per-transition **intervention indicators**; every learned arm
  gets the identical corpus (`SPEC.md:440-452`). Decisions scored under `P(K_out|do(A=a))`.
- **Structural gate** deterministic + `λ=0` control (`GATE_CONTRACT_E4.md:156-205`): PASS iff
  `mean(gap_norm) ≥ 0.10` AND `mean(|gap_norm_control|) ≤ 0.01`. **Trained-arm decision value is
  DEFERRED and may honestly be null** — prior 5-seed deconfounding tests had all CIs including 0
  (`SPEC.md:491-495`, `memory/causal-planner-direction-explored.md`).
- **Build status:** env `cdd_oran/envs/v2/e4.py` + `scripts/e4_structural_gate.py`; **no `e4slice`**.

### E5 — integrated stress (the capstone; composition only)
`SPEC.md:556-599`.

- **No new equations.** One graph at moderate density carrying: an **E2 shared-control fan-out block**
  (`P_shared → {multiple xApps}`), an **E3 KPI→KPI temporal chain** (`K→K→K`, one-step lag), an **E4
  confounding block** (`Z→A_behavior, Z→K_out, A→K_out`), and **moderate observation noise ON** (unlike
  E1 — a config-sweep value, not a mechanism) (`SPEC.md:563-569`).
- **Combined trap** = incomplete fan-out (E2) ∧ myopic/truncated chain (E3) ∧ correlational
  observational structure (E4), all at once (`SPEC.md:577-579`).
- **Primary metric = aggregate AND worst-case decision regret**, over the composed
  conflict/temporal/interventional decision population, **ID and OOD** under the single common
  support-shift rule of SEMANTICS §7 (`SPEC.md:586-588`, `583-584`).
- **Gating discipline:** each sub-mechanism must have **already passed** its own SCM + decision-gap +
  control gate before E5; E5 introduces no mechanism not already validated in E2–E4, else it is
  rejected outright (`SPEC.md:571-572,595-599`).
- **Expected result / failure mode:** discovered ≈ oracle with bounded aggregate & worst-case regret
  ≪ combined decoy, graceful ID→OOD; a **collapse under composition** (a mechanism that held alone but
  fails when combined) is a legitimate integration-negative result (`SPEC.md:590-593`).
- **Build status:** research-only, design notes; **not built** (`envs/README.md:32`).

**Net:** E5 turns exactly the four crutches the task warns about back off — real KPI→KPI fan-out
chains (E3), latent confounders / non-free orientation (E4), noise ON, and a shared knob whose harm
must be aggregated across breadth *and* depth *and* deconfounded.

---

## 2. The causal representation + decision rule the shared-knob question demands

To *reject a locally-attractive-but-globally-harmful shared-knob action*, the decision layer needs
four things. E2 supplies the first two cheaply; E5 makes all four load-bearing.

1. **Oriented, complete fan-out topology of the knob.** The full set of KPIs causally downstream of
   the shared knob `P0`, **including multi-hop KPI→KPI paths** (E3), not just the direct
   `P0→K_i` edges (E2). "Complete" is the whole point: E2's decoy omits exactly one edge `P0→K5`
   (`GATE_CONTRACT_E2.md:69-76`) and therefore mispredicts the harm and takes the myopic action; the
   complete-fan-out oracle predicts the K5 harm and avoids it. In E5 the missing consequence can hide
   two hops down a KPI→KPI chain, so the topology must be *transitively* complete.
   - **Orientation source differs by env:** E2/E3 get orientation **free from time** (`t→t+1`); the
     3-lane synthesis confirms FCM/PC orientation machinery is *redundant* there
     (`reports/2026-09-06-causal-upgrade-synthesis.md:8-22`). **E4 breaks this** — a latent confounder
     makes the observational edge point the wrong way in *sign*; orientation/identification must then
     come from **interventional data**, not time.

2. **Intervention semantics + effect propagation.** The ability to propagate `do(P0=v)` to **all**
   affected KPIs at their correct lags — traversing KPI→KPI chains (E3) and integrating out the latent
   `Z` (E4, `K_score(a)=E_Z[K_out|do(A=a)]`, `GATE_CONTRACT_E4.md:116-119`). This is a **world-model**
   capability, and it is the project's identified real blocker (see §3).

3. **Signed effect magnitudes, not edge existence.** RCoT answers *"is there an edge?"* (binary
   adjacency). The decision needs the **signed, quantitative** effect of each fan-out path to weigh
   local gain against aggregate harm. E4 makes this decisive: correlation returns slope `+3.29`, the
   true effect is `-1` — an edge-presence test that is *right about the edge* is still catastrophically
   *wrong about the decision* if it inherits the observational sign.

4. **A local-vs-global decision rule.** Compare the **local objective gain** of the myopic action
   against the **aggregate + delayed global harm** summed across the full fan-out and over the horizon,
   and reject when global harm exceeds local gain. This is exactly the frozen regret comparator:
   cumulative latent return `G = Σ_h R(k_{t+h})` on the **full matched panel** (SEMANTICS §1.3/§2,
   `SEMANTICS.md:94-104,205-228`), with the oracle maximizing `G` over the declared action class and
   `regret = G_oracle − G_planner`, unclamped (`SEMANTICS.md:243-254`). The oracle "compares local vs
   global" *by construction* because it scores the complete objective over the complete horizon; the
   decoy loses precisely because its world model is structurally incomplete (E2), myopic/truncated
   (E3), or correlational (E4). E5's metric adds **worst-case** regret (`SPEC.md:586-588`), so it is
   not enough to be right on average — the representation must prevent the *worst* myopic trap.

**Where the per-edge RCoT test fits, and what more is needed** (answers task Q2):
RCoT is the **skeleton/existence primitive for one edge** — time-oriented CI testing. It is stage 1 of
the discovery pipeline and is sufficient for E1/E2's *skeleton under time + causal sufficiency*. It
does **not** by itself provide: (a) orientation when time is not enough (E4 latent confounding);
(b) reliable KPI→KPI chain recovery under determinism-faithfulness (E3/E5 — flagged harder in
`…synthesis.md:80-82`); (c) **effect magnitudes/signs** (RCoT is a test, not an estimator);
(d) **intervention propagation** across multi-hop chains and joint/multi-param interventions; (e) a
**planner/decision layer** that turns propagated effects into a reject/accept; (f) graceful behavior
under **observation noise** (E5 turns it on — and genuine noise actually *re-enables* the FCM methods
E2 could skip, `…synthesis.md:80-82`).

---

## 3. End-to-end pipeline, and the gap between what we have and what E5 needs

Target pipeline (this is where structure discovery MEETS planning/effects — task Q3):

```
obs + interventional data (with do/obs indicators, moderate noise)
  → discovered ORIENTED structure   (skeleton + orientation + latent-edge handling)
  → fitted EFFECT model             (signed magnitudes, deconfounded via the do-subset)
  → intervention PROPAGATION engine (multi-hop KPI→KPI at correct lags; joint/multi-param do)
  → PLANNER over the action class   (open-loop a1..aH; cumulative G on the full matched panel)
  → shared-knob DECISION            (reject the locally-attractive/globally-harmful action)
```

**What we already HAVE (building blocks that scaffold forward):**

- **RCoT-v2 per-edge CI test** — `cdd_oran/e2slice/discovery_rcot_v2.py` (frozen, `block_perm_reps 99→299`),
  banked + tested; outputs a **binary time-oriented adjacency mask**. The **run is parked** (~5–18 d on
  a 16 GiB box; `plans/007…`, `envs/README.md:29`). This is stage-1 skeleton discovery, valid where
  time + causal sufficiency hold (E1/E2).
- **E1 recovery gate PASSED** via v2 partial-correlation discovery, including both **KPI→KPI edges**
  (`docs/benchmark/E1_DISCOVERY_RESULT_V2.md:10-14`) — so E2–E5 are no longer blocked by an unmet E1
  recovery gate. (The clean-env KPI→KPI recovery primitive exists.)
- **The decision-scoring kernel** — `cdd_oran/benchmark/rollout.py`
  (`rollout_open_loop`/`enumerate_open_loop`/`paired_regret`, `SEMANTICS.md:416-422`): the shared
  open-loop executor, the exhaustive open-loop oracle, and unclamped paired regret. The comparator
  math for the decision rule (§2.4) already exists.
- **Analytic SCM + decision/structural gates for E2, E3, E4** — `scripts/e{2,3,4}_*_gate.py`. **Caveat:
  these validate that the *benchmark expresses the trap*, using the TRUE SCM oracle and a hand-built
  decoy world model. They do NOT run a discovered structure through a trained planner.** The
  discovered-structure→decision path is *not* exercised by any gate today.
- **Env classes E1–E4 built** (`cdd_oran/envs/v2/e1..e4.py`); planner zoo (`cdd_oran/planners/`):
  QACM (now reclassified as a **baseline**, not ours — `memory/planner-set-redesign-direction.md`),
  CEM, MPPI, Joint, RecedingHorizonCEM, and CID (unmerged; **deterministically ≡ QACM today** —
  `memory/cid-planner-reviewed.md`).

**What we must BUILD (the gap):**

| Capability | Status | Needed for | Source |
|---|---|---|---|
| E5 env (composition) + `e3slice`/`e4slice` discovery+decision pipelines | **absent** | E3/E4/E5 | `envs/README.md:30-32` |
| Orientation/identification under **latent confounding** (use the do-subset, not time) | absent | E4, E5 | `SPEC.md:399-402`; synthesis §3 point 3 |
| KPI→KPI chain discovery robust under **determinism-faithfulness / noise** | RCoT alone insufficient | E3, E5 | `…synthesis.md:49-55,80-82` |
| **Effect-magnitude/sign estimation** on the discovered graph (not just edges) | absent | E4, E5 (sign is decisive) | `GATE_CONTRACT_E4.md:96-109` |
| **Multi-intervention world model** (propagate `do(P0)` to all fan-out KPIs; joint/multi-param) | absent — **the identified real blocker** | E2 fan-out, E3 chains, E5 | `memory/cma-es-gate-falsified.md`; `planner-set-redesign-direction.md` |
| **Discovered-structure → planner decision path** for trained arms | gates are analytic-oracle-only | E2–E5 (the actual claim) | `scripts/e2_decision_gate.py` header |
| **Sequence-emitting open-loop planner** matching oracle `a1..aH` | absent (RecedingHorizonCEM emits first action only) | E3, E5 | `SPEC.md:368-376` |

**The single most load-bearing gap** is the **multi-intervention world model**. Three independent
threads converge on it: the CFCP coordination win (+2.22) is real but oracle-only and discovery-hidden
(`memory/causal-planner-direction-explored.md`); the CMA-ES gate falsified the *optimiser* as the
lever and re-localized coordination to *"predict joint multi-param interventions"*
(`memory/cma-es-gate-falsified.md`); and the planner-redesign synthesis concluded *"the planner is NOT
the bottleneck … the real fix = world-model multi-intervention capability"*
(`memory/planner-set-redesign-direction.md:37-49`). A shared-knob fan-out decision **is** a
multi-intervention prediction problem: you cannot weigh the aggregate harm of `do(P0)` unless the model
can predict `P0`'s effect across *every* downstream KPI at once. This is the piece that makes E2's
building blocks either scaffold toward E5 or dead-end.

---

## 4. Implications — what to build NOW so E2 scaffolds toward E5 (not a dead-end at E2)

Ranked by leverage toward the E5 end goal:

1. **Build the discovered-structure → decision path end-to-end on E2 first, and treat edge-recovery as
   stage 1 of a pipeline, not the deliverable.** Today the E2 decision gate is analytic-oracle-vs-
   analytic-decoy; the *scientific claim* ("discovered fan-out lets a planner reject the harmful move")
   is untested. Wire `discovered mask → effect model → propagation → planner → paired_regret` on E2
   (the simplest env), reusing `rollout.py`. This makes the whole pipeline real on the easy case, then
   E3/E4/E5 reuse it. **Do not stop at the RCoT recall number** — a green E2 skeleton that never drives
   a decision cannot compose into E5.

2. **Prioritize the multi-intervention world model** (§3). It is the identified real blocker and the
   literal capability the fan-out decision needs. Without it, the coordination/fan-out win stays
   oracle-only and E5 cannot be attempted honestly. Frame it as a causal-modeling thread (train
   on / structure for joint & multi-hop interventions), not a planner-optimiser thread.

3. **Add effect-magnitude/sign estimation on the discovered graph now.** E4 proves edge-existence is
   insufficient and correlation gets the *sign* wrong. Fitting signed, deconfounded effects (using the
   interventional subset for identification) is required for E4/E5 and harmless for E2 (it just
   confirms the known signs). This is the difference between "there is an edge" and "moving the knob
   this way costs more than it gains."

4. **Plumb interventional data (obs/do split + per-transition intervention indicators) into the
   discovery/effect stack now.** E4/E5 need it, and it is the *only* identification route when time
   does not orient (latent confounding). Build it into the dataset layer while E2 is the test case so
   E4/E5 inherit it.

5. **Build the sequence-emitting open-loop planner** matching the oracle's `a1..aH` action class
   (`SPEC.md:368-376`). E3 and E5 are cumulative-horizon decisions; the current RecedingHorizonCEM
   cannot make the locked open-loop claim. This is a concrete, self-contained prerequisite.

6. **Keep RCoT as the CI oracle but adopt the two carry-forward discovery upgrades**, per the 3-lane
   synthesis (`…synthesis.md:34-55`): (A) tiered PC-stable **low-order skeleton** search with per-target
   BH-FDR kept on top, and (B) a **determinism-native functional-support** signal. Both extend to the
   KPI→KPI chains E3/E5 add, and (B)'s determinism assumption is exactly what E5's noise will stress —
   so build (A)/(B) knowing E5 may require the noise-appropriate FCM (ANM/PNL) that E2 could skip
   (`…synthesis.md:80-82`).

7. **Do NOT add noise to the E2 benchmark to "simulate" E5** (`…synthesis.md:29-31`: counter-productive,
   risks a gameable benchmark). Instead make the *method* noise-ready and let E5 turn noise on via its
   config sweep, as specified.

8. **Respect the composition gating discipline** (`SPEC.md:571-572`): E5 is only attemptable after
   E2/E3/E4 each pass their SCM + decision-gap + control gates independently. That ordering is also the
   correct build order — each env's *decision pipeline* (not just its analytic gate) is a prerequisite,
   and only E2's exists in part. A collapse-under-composition at E5 is itself a publishable result
   (`SPEC.md:590-593`), so the goal is a faithful pipeline, not a forced positive.

---

## Sources

Specs/contracts: `docs/benchmark/SPEC.md` (E2 216-285, E3 289-386, E4 390-552, E5 556-599, cross-cut
603-621); `docs/benchmark/SEMANTICS.md` (§1.1-1.3 47-104, §2 198-239, §3 243-258, §10 405-412, ptr
416-422); `docs/benchmark/GATE_CONTRACT_E2.md`, `GATE_CONTRACT_E3.md`, `GATE_CONTRACT_E4.md`;
`cdd_oran/envs/README.md`; `plans/README.md`, `plans/007-e2-discovery.md`.
Code: `cdd_oran/e2slice/discovery_rcot_v2.py`, `scripts/e2_decision_gate.py`,
`cdd_oran/planners/*`, `cdd_oran/benchmark/rollout.py` (per SEMANTICS ptr).
Reports/memory: `reports/2026-09-06-causal-upgrade-synthesis.md`;
`memory/planner-set-redesign-direction.md`, `cma-es-gate-falsified.md`, `cid-planner-reviewed.md`,
`causal-planner-direction-explored.md`; `docs/benchmark/E1_DISCOVERY_RESULT_V2.md`.
