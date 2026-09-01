# E4 Structural Gate Contract (FROZEN preregistration)

Frozen pre-code from `rule-e4-design.md`; no value may change after the first run.

This artifact transcribes the reviewer's **"Frozen E4 Contract Value Block"** (ruling
`.herdr/reports/rule-e4-design.md`, FIX-NEEDED → corrected contract frozen; the proposal
`.herdr/reports/e4-research.md` had 2 blockers the ruling CORRECTED) VERBATIM. It fixes every
equation, coefficient, policy, mixture, bank, action, threshold, and tolerance for the E4
action-relevant-confounding **structural design-validation gate** before any E4 environment or
gate execution. A failed feasibility, SCM, primary-gap, or control check is recorded as-is; a
normalized null is a valid boundary result.

This unit builds the **environment + structural (design-validation) gate + λ=0 control** only.
The trained-arm decision-value evaluation (structure-aware vs correlational vs discovered vs
dense, on the matched 90:10 corpus) is **DEFERRED** (requires training) and is transcribed here
only as the deferred list; its defaults MUST NOT change after this unit's first run.

---

## Status and anti-tuning rule

**Status: FROZEN PRE-CODE.** This block supersedes the unfrozen values in
`.herdr/reports/e4-research.md` for the tonight structural gate. A failed feasibility, SCM,
primary-gap, or control check is recorded as-is. No equation, coefficient, policy, mixture,
bank, action, threshold, or tolerance below may change after the first implementation run.

## SCM and timeline

```text
num_params = 1                         # P0 = A
num_kpis   = 1                         # K0 = K_out
A decision grid V = {0,0.01,...,1.00}
H = 1, gamma = 1

Z_q   ~ Normal(0,1)
eta_q ~ Normal(0,0.5^2)

K_out(q+1) = alpha*A_q + theta*Z_q + c
alpha = -1.0
theta = +2.5
c     = 0.0
```

The graph contains `Z->A_behavior`, `Z->K_out`, and `A->K_out`. `Z` is latent: it is neither a
parameter nor KPI and is never exposed to a planner. The observed env adjacency contains only
`(K0,P0)=(0,0)`; the two latent edges are mandatory SCM metadata and gate assertions.

For transition coordinate `q`, the environment draws/stores `Z_q` before action assignment. An
observational action and its delayed outcome use the same `Z_q`. A do-action is independent of
`Z_q`, but its delayed outcome still uses `Z_q`. Pending and committed `Z` must be included in
snapshot/restore. The H=1 realization uses `apply_action;advance` for unscored k1, followed by one
action-free terminal `advance` for scored k2, exactly as `SEMANTICS.md:106-127`.

## Behavior and matched obs/do protocol

```text
lambda primary = 1.0
lambda sweep   = {0.0,0.5,1.0,1.5}
A_behavior     = clip(0.5 + lambda*Z + eta, 0, 1)
rho_obs:rho_do = 90:10
D=0 observational: A=A_behavior, same Z enters K_out
D=1 interventional: A sampled uniformly over V independently of Z
```

Use dedicated non-aliasing tape namespaces/slots for `Z`, `eta`, randomized do-action, transition
noise, and observation noise. Assert byte-identical exogenous coordinates across arms taking
different actions. Every future learned arm receives the identical generated corpus, row order,
`D` flags, train/validation split, and do-grid coverage.

For exact clipped-normal moments, let `W=lambda*Z+eta`, `s2=lambda^2+0.25`,
`r=0.5/sqrt(s2)`, standard normal CDF `Phi`, density `phi`, and
`p=2*Phi(r)-1`. Then:

```text
E[A_behavior]   = 0.5
Var(A_behavior) = s2*(p - 2*r*phi(r)) + 0.25*2*(1-Phi(r))
Cov(A_behavior,Z) = lambda*p
```

At primary `lambda=1`:

```text
p                  = 0.34527915398142284
Var(A_behavior)    = 0.1916935893025823
Cov(A_behavior,Z)  = 0.34527915398142284
```

The do stream samples the 101 grid points uniformly, so `E[A_do]=0.5` and
`Var(A_do)=17/200=0.085`. For the naive OLS pooler that ignores `D` and omits latent `Z`, the exact
90:10 population line is:

```text
Var_pool(A) = 0.9*0.1916935893025823 + 0.1*0.085
            = 0.18102423037232407

Cov_obs(A,K) = alpha*Var(A_behavior) + theta*Cov(A_behavior,Z)
             = 0.6715042956509748

Cov_pool(A,K) = 0.9*Cov_obs(A,K) + 0.1*alpha*Var(A_do)
              = 0.5958538660858774

b_pool = Cov_pool/Var_pool = 3.291569669211391
c_pool = E[K] - b_pool*E[A] = -0.5 - 0.5*b_pool
       = -2.145784834605695
K_hat_naive(a) = c_pool + b_pool*a
```

This is a robust sign reversal, not a knife-edge. At `lambda=0`, `A_behavior` is independent of
`Z`; both observational and do slopes equal `alpha=-1`, so the pooled line is exactly `-a`.

## Exact standardization and objective

Authoritative scoring integrates `Z` to its mean and uses the continuous ID reference
`A~Uniform[0,1]`:

```text
K_score(a) = E_Z[K_out|do(A=a)] = -a
mu_out     = -0.5
sigma_out  = sqrt(1/12) = 0.28867513459481287  # evaluate exact sqrt in code
direction  = 0                              # satisfy-above
threshold_raw = 0.0
theta_std = (0-mu_out)/sigma_out = sqrt(3)
panel = (K_out,) only
```

Use the locked SEMANTICS hinge exactly once on raw latent/interventional-mean KPI, with unit weight,
`s=10`, and higher return better. Observation noise, sampled Z, env reward, and alternate utility
sums are excluded from the primary gate.

## Geometry bank

```text
candidate env_seed = 0..4095
episode = 0
obs_noise_scale = 0
neutral advances = 3
m = 0.50 standardized
N = 64
```

For each committed state, evaluate only the declared true interventional geometry over `V`:

```text
z_true(a) = (-a-mu_out)/sigma_out
valid iff z_true(0) >= theta_std
      AND z_true(1) <= theta_std-m
      AND z_true(0)-z_true(1) >= 2*m
```

The minimal SCM has no decision context, so this rule intentionally verifies action relevance but
does not manufacture context strata. Traverse seeds ascending and retain the first 64 valid states
(expected seeds `0..63`). If fewer than 64 are available, fail feasibility without expanding the
pool or relaxing the rule. Labels may not inspect decoy actions, returns, regret, or `D(s)`.

## Structural gate

The aware/oracle analytic model uses `K_hat_oracle(a)=-a`. The naive analytic model uses the exact
primary pooled line above. Both select `argmax` of the same locked hinge over `V`, with the smallest
grid index on ties. Every selected action is then realized and scored on the true interventional
mean SCM.

At the frozen primary values:

```text
a_oracle = 0.00
naive zero crossing = -c_pool/b_pool = 0.6519032103974248
a_naive  = 0.66  # first grid action satisfying the naive hinge

G_true(0)    = 1
G_true(a>0)  = -10*a/sigma_out
D(s)         = max_a G_true(a)-min_a G_true(a)
             = 1 + 10/sigma_out
             = 35.64101615137755

gap_raw(s)  = G_true(a_oracle)-G_true(a_naive)
gap_norm(s) = gap_raw(s)/D(s)
            = 0.6695395708852947
```

Executable definitions:

```text
G_star(s)        = max_a G_true(s,a)
D(s)             = max_a G_true(s,a)-min_a G_true(s,a)
regret_true(X,s) = G_star(s)-G_true(s,a_X)
gap_norm(s)      = (regret_true(naive,s)-regret_true(oracle,s))/D(s)

tau_E4   = 0.10
tol      = 0.01
tol_zero = 1e-12
```

If any retained `D(s)<=tol_zero`, fail degenerate without replacing the state. Primary structural
clause passes iff `mean_bank(gap_norm)>=tau_E4`.

## Factor-removal control and acceptance

Remove exactly `Z->A_behavior` by setting `lambda=0`; leave `theta`, `Z->K_out`, `A->K_out`, the
90:10 mixture, bank, panel, scoring, and action grid unchanged. Recompute the exact population
pooler. It has slope `-1`, emits `a_naive_control=0`, and therefore gives `gap_norm_control=0`.

**PASS iff both** `mean_bank(gap_norm)>=0.10` and
`mean_bank(abs(gap_norm_control))<=0.01`. Both clauses must drive the result and process exit.
Injected wrong controls must fail. Report exact moments, slopes/intercepts, actions, true returns,
regrets, `D`, raw/normalized gaps, bank seeds, and both acceptance clauses.

## Explicitly deferred

- all model fitting and planner training;
- the aware `D=1` regression, correlational learned pooler, discovered arm, and dense arm;
- learned action disagreement and regret (`pi_E4=0.50`, `tau_E4_trained=0.10` defaults);
- finite-sample uncertainty, train/validation behavior, and seed aggregation;
- positive-lambda sweep results beyond deterministic analytic diagnostics;
- sampled-Z/full-distribution regret, prediction metrics, ID/OOD, and richer-context sensitivity;
- any claim that structural causal inductive bias improves learned decisions.

## Implementation pointers (this unit)

- Env: `cdd_oran/envs/v2/e4.py` — `E4V2Env(V2Env)`, `num_params=1`, `num_kpis=1`, the confounded
  mechanism above, dedicated non-aliasing latent/eta/do tape slots, obs/do modes, pending +
  committed `Z` in snapshot/restore, `score_interventional_mean` (`K_score=-a`), latent-edge
  metadata `LATENT_EDGES`, observed adjacency `[(0,0)]`, `true_adj_matrix()`.
- Structural gate: `scripts/e4_structural_gate.py` — the design-validation protocol above;
  injectable `env_factory`/`control_env_factory`; `RESULT: PASS/FAIL` + exit status.
- Tests: `tests/test_e4_structural_gate.py`.
