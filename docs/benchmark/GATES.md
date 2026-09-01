# Phase-4 Decision Gates (preregistered)

**Status:** Preregistered before any simulator search. Thresholds, decoys, and state banks
are fixed here. Failed gate attempts are recorded rather than tuning equations toward a
positive result.

Every gate below uses the **latent noiseless** comparator (SEMANTICS §3): regret =
`G_oracle − G_planner`, cumulative over the scored window `k_{t+1..t+H}`, both returns
produced by cloned rollouts on the same `(env_seed, episode)` under identical exogenous
draws. The finite-horizon oracle searches the open-loop action sequence over the 101-point
grid (SEMANTICS §3).

---

## 1. SCM gate (WP3 deliverable)

The procedure WP3 implements:

- **Direct-child at exact lag.** A declared edge `K_c <- P_i` passes only if `|Δ_c| ≥
  min_eff = 0.05` (standardized latent units), where `Δ_c = |k_c(t+1; do high_i) −
  k_c(t+1; do low_i)|`, at **every** state in the bank, for the `high_i` intervention.
- **Descendant over horizon.** A KPI->KPI descendant `K_d` reached through `h` KPI edges
  responds at `t+1+h`. The descendant horizon is the chain depth. Checked at its own lag,
  not at `t+1`.
- **Interior points.** `N_states = 8` latent states from reset + 3 neutral steps (noise
  OFF). These are interior, not the reset state.
- **Min-effect tolerance.** `min_eff = 0.05`. A declared edge that moves its child by less
  than `min_eff` at any state is a FAILURE.
- **Negative controls.** For each `do(P_i)`, every non-descendant KPI must satisfy
  `|Δ| ≤ tol_zero = 1e-6` at all lags up to `H_desc`. Movement above `tol_zero` at a
  non-descendant is a FAILURE.
- **Edge-ablation / direct-parent intervention (verifies DIRECT KPI->KPI parentage, not
  mere ancestry).** A param-ancestor drive alone cannot tell a declared `K0->K2` from an
  undeclared direct `P0->K2`: after the bank equilibrates on the manifold `K0 = P0`, both
  move `K2` when `P0` moves, and a shadow that zeroes the KPI coefficient removes both
  paths (demonstrated false-pass, review BLOCKER 1). For **each** declared KPI->KPI edge
  `K_s -> K_d`, therefore, intervene **directly on the latent source** `prev_kpis[s]`,
  pushing it OFF the manifold (`K_s = P_s ± 0.3`, so `K_s ≠ P_s`) while holding **all
  params and every other latent KPI fixed**, then advance one step and read the child.
  The declared edge is **confirmed** iff:
  1. **True SCM** — the child moves by `≥ min_eff` (a real `K_s -> K_d` edge is present);
  2. **Shadow SCM** (that one KPI-parent coefficient zeroed, `b_d = 0`) — the same direct
     source move leaves the child inert (`|Δ| ≤ tol_zero`); the KPI-parent contribution is
     gone;
  3. the **direct param->child effect** `do(P_d)` is **unchanged** between true and shadow
     (`|Δ_shadow − Δ_true| ≤ tol_zero`), isolating the KPI-parent path from the param path;
  4. **negative control** — a direct off-manifold move of a **non-parent** source KPI
     (e.g. `K1` for `K2`) produces **no** effect on the child (`|Δ| ≤ tol_zero`): no
     undeclared cross-KPI edge exists.
- **Latent noiseless.** `noise_scale = 0`. Observation noise is excluded entirely.
- **Analytical match (on AND off manifold).** The closed-form next-latent-KPI (independent
  of the env code) matches the rolled-out value to `tol_zero`, evaluated on the neutral
  on-manifold bank states **and** on preregistered OFF-manifold `(prev_params, prev_kpis)`
  states where `K0 ≠ P0` and `K1 ≠ P1`. The off-manifold evaluation is what pins the
  coefficients and lags (a wrong direct coefficient or a wrong-lag / cross-KPI read agrees
  on the manifold but diverges off it).

**Acceptance:** Green iff, at **all 8 states**, (a) every declared edge passes the
minimum-effect rule at its correct lag; (b) every non-descendant passes the `do(P_i)`
negative control; (c) every declared KPI->KPI edge passes the direct off-manifold
edge-ablation check (true moves, shadow inert, direct-param path invariant, non-parent
control inert); (d) the analytical match holds on- and off-manifold. Failure of any clause
blocks WP3, with the specific clause and edge flagged.

---

## 2. Decision-gap gate

### 2.1 Preregistration invariants

All thresholds, decoy rules, state banks, and acceptance criteria below are fixed **before
any simulator search**. Failed gate attempts are recorded; equations are not tuned toward a
positive result.

### 2.2 E1 exemption

E1 is **exempt** from the decision-gap and control gates (MAJOR 9 / SPEC E1). Its only gate
is SCM correctness (above). E1 carries no decoy arm.

### 2.3 E2 — incomplete fan-out decoy (SPEC E2)

**Decoy construction rule:** The decoy is an incomplete fan-out world model that omits
**EXACTLY ONE edge: `P0 -> K5`** (the fan-out edge into xApp4). This edge is fixed before
any search. The decoy mispredicts the effect of moving `P0` on `K5` (predicts no effect)
and picks the locally attractive `P0` move; the complete-fan-out oracle predicts the `K5`
harm and avoids it.

**State bank.** A fixed set of `P0`-shared decisions containing:
- **Positive controls:** states where `do(P0)` realizes harm to `K5`.
- **Negative controls:** states where `do(P0)` is inert on `K5`.

**Threshold.** `τ_E2` = minimum normalized realized regret. Set before any simulator
inspection. Anchor target: at least the validated CFCP shared-control effect on TRUE blocks.

**Acceptance.** Mean realized regret gap `(decoy − oracle) ≥ τ_E2` on positive controls
**AND** `≤ tol` (no gap) on negative controls.

**Matched population.** The full xApp objective and full conflict panel (including xApp4/K5)
are IDENTICAL for oracle and decoy arms. The decoy only mispredicts K5; it does NOT remove
K5 from realized scoring. Regret is computed on the full objective under the true SCM.

**Planner.** The SAME single-control planner for BOTH arms (MAJOR 8).

### 2.4 E3 — myopic/chain-truncated decoy (SPEC E3)

Everything below is FROZEN by `docs/benchmark/GATE_CONTRACT_E3.md` (ruling
`review-e3-redesign.md`, RATIFIED fan-out redesign; `num_kpis` 4->5). The obsolete single-edge
`K1->K2` decoy is superseded.

**Decoy construction rule (2×2 design):** Four cells under the same model capacity and objective:

|                          | **H = 1 (myopic)** | **H = causal-depth horizon (H=3)** |
|--------------------------|--------------------|------------------------------------|
| **Full structure**       | cell FM            | cell FH (proposed)                 |
| **Truncated structure**  | cell TM            | cell TH                            |

Full structure is the depth-three cascade `P0->K0->K1->{K2,K3}` (K1 conduit, K4 non-descendant
control). The **truncated** analytic model drops **exactly the fan-out pair `{K1->K2, K1->K3}`**
(adjacency edges `(2,5),(3,5)`), zeroing only K1's contribution to K2 and K3; TRUE realization
and oracle always retain both edges. The **PRIMARY contrast is FH vs FM** — full structure held
fixed, horizon varied. This isolates temporal decision value.

**State bank.** `env_seed=0..4095`, `episode=0`, noise off, 3 neutral advances; valid iff
`P1>=L AND P2>=L`, `L=0.5+0.5*sqrt(2/12)`; retain the first `N=32` ascending (contract "State
bank"). Panel IDs `(0,2,3)`; excluded IDs `(1,4)`.

**Threshold.** `tau_E3 = 0.10`, `tol = 0.01`, `tol_zero = 1e-12` (contract "Decision gate").

**Acceptance.** PASS iff `mean_bank(gap_H_norm) >= 0.10` AND `mean_bank(abs(gap_T_norm)) <= 0.01`.
PRIMARY contrast FH vs FM; secondary FH vs TH isolates causal structure; TM is the floor. A
normalized null is a valid boundary result.

**Standalone SCM gate (prerequisite).** Before the decision gate, the SCM-correctness gate
(`scripts/e3_scm_gate.py`, contract "Standalone SCM gate") verifies the seven-edge adjacency,
the P0 descendant timing (K0@t+1, K1@t+2, K2&K3@t+3), K4 inertness under do(P0), off-manifold
direct-parentage for `K0->K1/K1->K2/K1->K3`, and that the decision decoy differs from TRUE by
exactly `(2,5),(3,5)`. Any failed clause blocks decision-gate execution.

### 2.5 E4 — correlational-on-observational decoy (SPEC E4)

**Decoy construction rule.** A correlational structure trained on the identical mixture of
observational (`Z`-confounded) and randomized-interventional transitions. It absorbs the
`Z`-induced spurious association and recommends the intervention that looks best
observationally but is worse (or sign-reversed) interventionally.

**Additional requirement (MINOR 14).** E4 requires **sign-reversal or positive
interventional regret** on the interventional evaluation distribution `P(K_out | do(A = a))`.

**Locked confounder parameters.** All of the following are fixed BEFORE any generation and
are NOT tuned:
- Behavior-policy equation `π_b(a | Z)`.
- Latent `Z` distribution `P(Z)`.
- Confounding coupling level(s).
- Fixed state bank.
- PRIMARY acceptance criterion (sign-reversal OR positive interventional action regret, with
  a fixed magnitude).
- Minimum action-disagreement prevalence.

**State bank.** Fixed set of states at which the interventional contrast is scored.

**Threshold.** `τ_E4` = minimum action-disagreement prevalence (fraction of bank decisions
where correlational arm and interventional oracle choose different actions), with
sign-reversal or positive interventional regret of at least the locked magnitude.

**Acceptance.** On the fixed state bank: correlational/decoy action and interventional
oracle action differ with sign-reversal or positive interventional regret of at least the
locked magnitude, on at least `τ_E4` of decisions.

**Matched-data contract.** Every learned arm receives IDENTICAL training corpus: same
observational transitions, same randomized-interventional transitions, same train/validation
splits, same intervention indicators, same action coverage. Only the structural assumption
varies.

**Built structural design-validation gate + λ=0 control (FROZEN, `GATE_CONTRACT_E4.md`).**
The tonight-built unit is the **structural (design-validation) gate**, NOT the trained-arm
evaluation. Everything below is FROZEN by `docs/benchmark/GATE_CONTRACT_E4.md` (ruling
`.herdr/reports/rule-e4-design.md`, which corrected two proposal blockers). It is deterministic
and uses NO training.

- **Minimal confounded SCM** (`cdd_oran/envs/v2/e4.py`): 1 param `A=P0`, 1 KPI `K_out=K0`, latent
  `Z` on `Z->A_behavior`, `Z->K_out`, `A->K_out`; `K_out=-A+2.5*Z` at one-step lag;
  `A_behavior=clip(0.5+lambda*Z+eta,0,1)`; obs/do modes on `rho_obs:rho_do=90:10`. Observed
  adjacency is only `(0,0)`; the two `Z`-incident edges are latent metadata.
- **Naive decoy** = exact 90:10 OLS pooler that ignores `D` and omits latent `Z`:
  `K_hat_naive(a)=c_pool+b_pool*a`, `b_pool=+3.291569669211391` (sign-reversed from the true
  `-1`), `c_pool=-2.145784834605695`. Aware/oracle model `K_hat=-a`.
- **Bank.** `env_seed=0..4095`, `episode=0`, noise off, 3 neutral advances; retain the first
  `N=64` states valid under the frozen `z_true` geometry rule (`m=0.50`); expected seeds `0..63`.
  Fewer than 64 ⇒ feasibility FAIL.
- **Gap.** Each model selects the locked hinge `argmax` over `V={0,..,1.00}` (smallest index on
  ties); realize on the true interventional-mean SCM `K_score(a)=-a`;
  `gap_norm=(regret_true(naive)-regret_true(oracle))/D(s)=0.6695395708852947`, `D(s)=35.641...`.
- **Threshold.** `tau_E4=0.10`, `tol=0.01`, `tol_zero=1e-12`.
- **λ=0 factor-removal control.** Remove exactly `Z->A_behavior` (`lambda=0`), all else fixed;
  the pooler recovers slope `-1`, `a_naive_control=0`, and `gap_norm_control=0`.
- **Acceptance.** `scripts/e4_structural_gate.py` — **PASS iff both**
  `mean_bank(gap_norm)>=0.10` **AND** `mean_bank(abs(gap_norm_control))<=0.01`. Both clauses are
  wired into `passed` and the process exit status; an injected wrong control (a `lambda!=0` env
  that does not remove the confounding edge) FAILS. A normalized null is a valid boundary result.

**DEFERRED (requires training).** The trained-arm decision-value evaluation — structure-aware
`D=1` regression, correlational pooler, discovered arm, dense arm on the matched corpus
(`pi_E4=0.50`, `tau_E4_trained=0.10`) — is NOT built tonight and is honestly allowed to be null;
it must not trigger changes to `Z`, `lambda`, the mixture, or the bank.

### 2.6 Per-env summary

| Env | Decoy rule                     | Regret type        | State bank | Threshold |
|-----|--------------------------------|--------------------|------------|-----------|
| E1  | EXEMPT                         | —                  | —          | —         |
| E2  | Omit `P0 -> K5`               | Single-decision H=1 | +/− controls | `τ_E2`  |
| E3  | Myopic H=1 / drop fan-out `{K1->K2,K1->K3}` | Cumulative H=3 | Cascade states | `τ_E3` |
| E4  | Correlational observational    | Interventional      | Fixed bank | `τ_E4` + sign-reversal |

---

## 3. Control gate

Removing the defining factor removes or greatly reduces the gap.

### 3.1 Per-env manipulation and metric

| Env | Defining factor      | Manipulation                            | Metric (reduced gap)                    |
|-----|----------------------|-----------------------------------------|-----------------------------------------|
| E1  | EXEMPT               | — (MAJOR 9)                             | —                                       |
| E2  | Shared-control fan-out | Remove `P0` from the conflict panel    | Regret gap (decoy − oracle) → ≤ tol     |
| E3  | Temporal cascade depth (fan-out) | Drop the fan-out pair `{K1->K2,K1->K3}` (truncated model) | Cumulative regret gap `abs(gap_T_norm)` → ≤ tol |
| E4  | Action-relevant confound | Remove `Z -> A_behavior` edge          | Interventional regret gap → ≤ tol       |

### 3.2 Acceptance (all envs except E1)

The gap between oracle and decoy must be **greatly reduced** (to within tolerance) when
the defining factor is removed. The manipulation is a single-factor change; no second axis
is varied. E1 is exempt (MAJOR 9).

---

## 4. Meta-rules

1. **Thresholds and decoys are fixed before any simulator search.** They appear in this
   document and are not updated after the first simulator run.
2. **Failed gate attempts are recorded** (in the handoff report) rather than tuning equations
   toward a positive result.
3. **One stress axis per env.** A decoy that needs a second stress axis is rejected
   (Decision Gates).
4. **Negative controls mandatory.** Every gate includes negative controls where the gap
   must be absent.
5. **Latent noiseless scoring.** All oracle/regret comparisons use the latent noiseless
   state (SEMANTICS §3, PD3).
