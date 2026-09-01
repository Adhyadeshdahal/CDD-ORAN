# E3 Phase-4 Gate Contract (FROZEN preregistration)

Frozen pre-code from `review-e3-redesign.md`; no value may change after the first run.

This artifact transcribes the reviewer's **"Frozen E3 contract value block"** (ruling
`review-e3-redesign.md`, LOAD-BEARING-PROVEN, 0 blockers) VERBATIM. It fixes every equation,
adjacency edge, threshold, decoy rule, state bank, margin, standardization, and tolerance for the
E3 temporal-depth fan-out gate **before any E3 environment, planner, oracle, or gate execution**.
A failed SCM gate, feasibility check, primary gap, or truncated control is recorded as-is; a
normalized null is a valid boundary result. The frozen block supersedes the obsolete single-harm
E3 SCM and the single-edge-decoy portions of `rule-e3-forks.md` (items 6 and 10's old single-edge
identity only; all planner, horizon, objective, oracle, normalization, and threshold rulings
remain active).

This unit builds the **environment + structural (SCM-correctness) validation** only. The decision
gate (state bank + 2×2 decision-gap + truncated control) is a SEPARATE later unit; its frozen
values are transcribed here for reference and MUST NOT change after this unit's first run.

---

## Status and anti-tuning rule

Freeze before any E3 environment, planner, oracle, or gate execution. A failed SCM gate,
feasibility check, primary gap, or truncated control is recorded as-is. Do not alter equations,
fan-out count, panel, edges, thresholds, standardization, bank, horizon, grid, denominator,
`tau_E3`, or tolerances after the first run. A normalized null is a valid boundary result.

## SCM

```text
num_params = 4
num_kpis   = 5
ID ranges  = P0,P1,P2,P3 in [0,1]

K0(t) = P0(t-1)
K1(t) = K0(t-1)
K2(t) = P1(t-1) + K1(t-1)
K3(t) = P2(t-1) + K1(t-1)
K4(t) = P3(t-1)
```

Every displayed coefficient is exactly 1.0; there are no intercepts or process noise. P0 is the
only acted coordinate. P1/P2/P3 are held within each decision.

Adjacency `(kpi_index,source_index)`, with KPI sources `4+i`:

```text
[(0,0), (1,4), (2,1), (2,5), (3,2), (3,5), (4,3)]
```

The TRUE graph is `P0->K0->K1->{K2,K3}`, plus `P1->K2`, `P2->K3`, and `P3->K4`.

## Truncated model and four cells

The truncated analytic planner model removes exactly adjacency edges `(2,5)` and `(3,5)` and
sets only the corresponding K1 terms in K2/K3 to zero. TRUE realization and oracle always retain
both edges.

```text
FH = full analytic model      + exhaustive H=3 sequence optimization
FM = full analytic model      + three predicted-state greedy H=1 choices
TH = truncated analytic model + exhaustive H=3 sequence optimization
TM = truncated analytic model + three predicted-state greedy H=1 choices
```

Every arm emits the complete sequence before TRUE execution. FM/TM roll predicted model state and
never inspect TRUE outcomes. A greedy choice at position h maximizes the next corresponding member
of the common scored trajectory; it must not execute an extra two-advance standalone rollout that
shifts the sequence timeline. All action ties use lexicographically smallest grid indices.

## Action and trajectory

```text
shared param = P0 (param_id=0), no joint action
H = 3
gamma = 1
V = {0,0.01,...,1.00}
oracle/FH/TH sequence class = V^3 (101^3 sequences)
```

The TRUE oracle exhaustively maximizes cumulative return over all 1,030,301 sequences. Use three
`apply_action;advance` pairs plus one action-free terminal `advance`; k1 is unscored and only
latent noiseless `{k2,k3,k4}` is scored. Candidate sequences restore independent clones of the
same committed state and exogenous coordinates. Regret is not clamped.

## xApps, panel, and objective

```text
xapp_kpi_indices    = [(0,), (1,), (2,), (3,), (4,)]
kpi_to_xapp         = {0:0, 1:1, 2:2, 3:3, 4:4}
xapp_param_indices  = [(0,), (0,), (1,), (2,), (3,)]
directions          = [0,0,1,1,0]
kpi_thresholds      = [0.5,0.5,1.0,1.0,0.5]
R panel IDs         = (0,2,3)
excluded IDs        = (1,4)
```

Use the same panel for every planner-model score, TRUE oracle score, realized return, and D3.
On raw latent KPIs, each panel xApp standardizes exactly once. With `theta_i=0`, unit weights,
`s=10`, and higher return better:

```text
direction 0: distance=max(-z,0), ok=(z>=0)
direction 1: distance=max( z,0), ok=(z<=0)
R(k) = (sum ok)^2 - 10*sum(distance)
G = R(k2)+R(k3)+R(k4)
```

Robust/risk aggregation, observation noise, env reward, K1, and K4 are excluded from the primary
gate objective.

## Exact standardization

Authoritative steady-state ID moments after exactly three neutral advances:

```text
mu = [0.5, 0.5, 1.0, 1.0, 0.5]
sigma = [sqrt(1/12), sqrt(1/12), sqrt(2/12), sqrt(2/12), sqrt(1/12)]
```

Evaluate square roots at full float precision. Do not paste six-decimal approximations. Monte
Carlo, if retained, is diagnostic only and is never compared to `tol_zero`.

## State bank

```text
candidate env_seed = 0..4095
episode = 0
obs_noise_scale = 0
neutral advances = 3
m = 0.50 standardized
L = 0.5 + m*sqrt(2/12) = 0.7041241452319316...
valid iff P1 >= L AND P2 >= L
traversal = ascending seed
retain first N = 32
```

Use the exact expression for L, not rounded `0.7041`. The expected deterministic retained seeds
are the 32 listed in the ruling's Part A; implementation must reproduce them:

```text
16,62,81,99,101,117,121,132,133,145,175,176,186,196,208,217,223,224,238,252,270,312,
355,365,370,373,375,386,401,414,415,436
```

If fewer than 32 valid states exist, fail feasibility without expanding the pool, changing
comparison precision, relaxing m, or changing the band. Labels use only committed TRUE-SCM
parameter geometry, never planner actions, returns, regret, or D3.

## Decision gate

For each retained state and sequence `a in V^3`, let `G_true(s,a)` be latent TRUE-simulator
cumulative return on `{k2,k3,k4}`.

```text
G_star(s)        = max_a G_true(s,a)
D3(s)            = max_a G_true(s,a) - min_a G_true(s,a)
regret_true(X,s) = G_star(s) - G_true(s, sequence_X)

gap_H(s)         = regret_true(FM,s) - regret_true(FH,s)
                 = G_true(FH,s) - G_true(FM,s)
gap_H_norm(s)    = gap_H(s)/D3(s)

gap_T_norm(s)    = (G_true(TH,s)-G_true(TM,s))/D3(s)

tau_E3   = 0.10
tol      = 0.01
tol_zero = 1e-12
```

If any `D3(s)<=tol_zero`, fail as degenerate without replacing the state. **PASS iff both**
`mean_bank(gap_H_norm)>=0.10` and `mean_bank(abs(gap_T_norm))<=0.01`. Both clauses must be wired
into the executable `passed` result and process exit status. Report all four sequences, TRUE
returns/regrets, D3, raw/normalized gaps, and FH-vs-TH as diagnostics; do not tune on them.

## Standalone SCM gate

The SCM gate runs before the decision gate, latent and noiseless:

```text
states = env_seed 0..7, episode 0, reset + 3 neutral advances
do values = 0.25 and 0.75 for each parameter
H_desc = 3
min_eff = 0.05 standardized units
tol_zero = 1e-12
off-manifold source offsets = +/-0.3
```

Required checks at all eight states:

- Direct param children at t+1: P0->K0, P1->K2, P2->K3, P3->K4.
- P0 descendant timing: K0 at t+1, K1 at t+2, K2 and K3 at t+3. Earlier/later movement must
  match the closed form.
- Every non-descendant remains within tol_zero through H_desc; in particular K4 is inert under
  do(P0).
- For each KPI-parent edge K0->K1, K1->K2, and K1->K3, directly set only the committed source KPI
  to its on-manifold value plus/minus 0.3 and advance once. The TRUE child must move by at least
  min_eff; a one-edge shadow with only that coefficient zero must be inert within tol_zero; the
  child's direct param path, where present (P1->K2 or P2->K3), must be invariant within tol_zero;
  and an identically moved non-parent KPI must leave the child inert within tol_zero. K0->K1 has
  no direct child-param path, so that subcheck is explicitly not applicable.
- Independently computed closed-form latent transitions must match simulator output within
  tol_zero on neutral, intervention, and off-manifold states.
- TRUE adjacency must equal the seven-edge list exactly; each truncated shadow must differ by
  exactly its intended one edge. The decision decoy must differ from TRUE by exactly the ruled
  pair `(2,5),(3,5)`.

Any failed SCM clause blocks decision-gate execution. OOD is deferred; both Phase-4 gates are
ID-only.

## Implementation pointers (this unit)

- Env: `cdd_oran/envs/v2/e3.py` — `E3V2Env(V2Env)`, `num_params=4`, `num_kpis=5`, the five
  mechanisms above, coefficient knobs `c10/c25/c35` (single-edge shadows), `truncate_fanout`
  (decision decoy: zeroes K1's contribution to K2 and K3, drops `(2,5),(3,5)`), exact-`sqrt`
  standardization, and `true_adj_matrix()`.
- SCM gate: `scripts/e3_scm_gate.py` — the standalone protocol above; injectable
  `env_factory`/`shadow_factory`; `RESULT: PASS/FAIL` + exit status.
- Tests: `tests/test_e3_scm_gate.py`.
