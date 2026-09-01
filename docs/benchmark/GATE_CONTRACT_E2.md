# E2 Phase-4 Gate Contract (FROZEN preregistration)

Frozen pre-sim from reviewer ruling `review-e2-buildplan.md`; no value may change after the
first gate run (SPEC E2 281-285, GATES 2.1/4).

This artifact fixes every threshold, decoy rule, state bank, margin, and normalization for the
E2 shared-control decision-gap gate **before any E2 simulation or regret inspection**. A failed
gate is recorded as-is; a null is a valid boundary result. Any post-run change to a value here
is a new blocker and invalidates the preregistered E2 result.

---

## 1. Mechanism family (Q3 — FROZEN)

Reuse all six Env II Gaussian-bump equations **exactly**, including `safe_exp`, coefficients,
thresholds, directions, ID ranges, and xApp maps (`env_ii.py:12-105`). No E1-style linear
substitution. Reimplemented on the v2 `V2Env` base (apply/advance, one-step latency,
coordinate-keyed CRN, noise-off latent scoring); legacy `env_ii.py` untouched.

`safe_exp(x) = x if |x| > 0.1 else 0.1`. Mechanisms (params `P0..P7`, one-step lag):

```
K0 = 80  * exp(-(P0)^2        / (2·safe_exp(P1)^2))
K1 = 100 * exp(-((P0+P2))^2   / (2·safe_exp(P1)^2))
K2 = 120 * exp(-((P0+45))^2   / (2·safe_exp(P3)^2))
K3 = 120 * exp(-((P5+P1-30))^2/ (2·safe_exp(P4)^2))
K4 = 150 * exp(-((P5+P1-50))^2/ (2·safe_exp(P4)^2))
K5 = -35 * exp(-((P7+P0-25))^2/ (2·safe_exp(P6)^2))   # P0->K5 : decoy-omitted edge
```

ID param ranges (non-ood branch, `env_ii.py:56-65`):
```
P0[-100,100] P1[-10,50] P2[-20,20] P3[-60,60] P4[-20,20] P5[-50,150] P6[-60,65] P7[-100,150]
```

xApp maps / directions / thresholds (`env_ii.py:74-87`):
```
xapp_kpi_indices = [(0,),(1,),(2,),(3,4),(5,)]      # xApp3={K3,K4}; xApp4={K5}
kpi_to_xapp      = {0:0,1:1,2:2,3:3,4:3,5:4}
directions       = [0,0,0,0,0,1]                    # xApp4/K5 satisfy-below
kpi_thresholds   = [55,95,85,75,80,-25]
```

### Frozen standardization constants (Q4)

Computed **once** via `get_env_ii_mean_std(param_ranges=ID, seed=0, num_samples=1_000_000)`
(the exact `compute_kpis_ii` reduction) and pasted as literals. Per-KPI `(mean, std)`:

```
K0: (21.482606, 27.694787)
K1: (26.776967, 34.651755)
K2: (40.931657, 44.767996)
K3: (14.975340, 32.369335)
K4: (18.755966, 40.460422)
K5: (-10.140367, 12.583744)
```

Panel xApps are single-KPI, so `xapp.mean/std` = the panel KPI's `(mean, std)` (legacy
`_mean_for_indices`, `base.py:149-158`). Provenance: `seed=0, num_samples=1_000_000`.

## 2. Adjacency (Q2 — FROZEN)

Code adjacency (`env_ii.py:88-105`) is authoritative. TRUE `P0 -> {K0,K1,K2,K5}` (no
`P0->K3`). Full edge list `[(kpi,source)]`:
```
(0,0)(0,1) (1,0)(1,1)(1,2) (2,3)(2,0) (3,1)(3,4)(3,5) (4,1)(4,4)(4,5) (5,0)(5,6)(5,7)
```

## 3. Decoy rule (FROZEN)

Drop **exactly one edge `(5,0)` = `P0->K5`**, fixed before search. Realized as a selection
world model that **freezes the committed pre-decision `P0` inside the K5 mechanism only** (P6,P7
live); all other mechanisms remain TRUE. The decoy is used **only** to select the decoy arm's
action; realized scoring is always on the TRUE SCM, and xApp4/K5 stays in both arms' objective
(SPEC E2 240-252; GATES 2.3 matched population).

## 4. State/conflict bank (Q5 — FROZEN)

K5-response geometry only; never inspect actions, regret, or oracle/decoy returns for labels.

Candidates: `env_seed = 0..4095`, `episode = 0`, reset with observation noise **off**, then
**exactly 3 neutral advances**; take the committed state `s = (prev_params, prev_kpis)`.

For each `s`, hold every committed coordinate except P0 fixed and evaluate the TRUE noiseless K5
mechanism on the fixed 101-point inclusive grid `V = {-100,-98,...,98,100}`:
```
z5_s(v) = (K5_true(s; do(P0=v)) - mu5) / sigma5      mu5=-10.140367, sigma5=12.583744
theta5  = (-25 - mu5) / sigma5                       = -1.180859
z_min = min_v z5_s(v)   z_max = max_v z5_s(v)   S5(s) = z_max - z_min
```
xApp4 is satisfy-below, so larger K5 is harmful.

- **Positive:** `z_min <= theta5 - 0.50` AND `z_max >= theta5 + 0.50`.
- **Negative:** `S5(s) <= 0.05`.
- **Grey zone:** every other state excluded, never relabelled.

Traverse seeds ascending; retain the **first 32 positive and first 32 negative**. Final bank
size **exactly 64**. If the pool lacks 32 of either class, E2 **fails preregistration
feasibility**; do not expand the pool or relax a margin.

## 5. Normalization, thresholds, PASS/FAIL (Q6 — FROZEN)

Full locked panel `{xApp0, xApp1, xApp2, xApp4}`, locked latent objective `R` (§6). Selection
is `argmax_v R` (= `argmin_v cost`; BLOCKER 1). For each retained state, over `v in V`:
```
D(s)        = max_v R_true(s,v) - min_v R_true(s,v)
regret_true(a) = max_v R_true(s,v) - R_true(s,a)
gap_raw(s)  = regret_true(a_decoy) - regret_true(a_oracle)     # a_oracle=argmax R_true
gap_norm(s) = gap_raw(s) / D(s)
```
`D(s)` is computed from the TRUE objective over the preregistered action class, independent of
the decoy-selected action. If any retained state has `D(s) <= 1e-12`, the gate **fails as
degenerate**; do not drop or replace it. Numerical equality checks use `tol_zero = 1e-12`,
separate from the scientific no-gap tolerance.

```
tau_E2   = 0.10
tol      = 0.01
tol_zero = 1e-12
```

**PASS** iff `mean_positive(gap_norm) >= 0.10` AND `mean_negative(gap_norm) <= 0.01`
(consequently pos/neg mean separation `>= 0.09`). The `0.01` band is a scientific
negligible-effect band, not a Monte-Carlo bound (latent deterministic, CRN-paired).

`tau_E2=0.10` / `tol=0.01` are unitless fractions of each decision's complete TRUE-SCM
action-return range `D(s)`; the legacy +2.22 result is the **rationale** for studying this
family, not a cross-unit threshold.

## 6. Locked objective `R` (SEMANTICS §2)

Panel `{xApp0,xApp1,xApp2,xApp4}`, on the **raw latent** KPI vector `k`, `w_i=1`, `s=10.0`,
`γ=1`:
```
u_i = (mean_of_xapp_raw_kpis(k) - xapp.mean_i) / xapp.std_i
theta_i = (threshold_i - xapp.mean_i) / xapp.std_i
dir 0: distance_i = max(theta_i - u_i, 0)   ok_i = (u_i >= theta_i)
dir 1: distance_i = max(u_i - theta_i, 0)   ok_i = (u_i <= theta_i)
cost(k) = Σ_i w_i·distance_i·s − (Σ_i ok_i)^2
R(k)    = −cost(k)          (higher R is better; select argmax_v R)
```

## 7. Factor-removal control (GATES §3.1 — FROZEN)

Remove the shared-control factor by dropping xApp4/K5 from the panel (panel → `{xApp0,xApp1,
xApp2}`), recompute `gap_norm` on the positive controls; it must fall to `<= tol = 0.01`
(same normalized tolerance). Single-factor manipulation; nothing else changes.

## 8. Horizon / planner

H = 1, single-control, single decision on `P0`. Gate selector = exhaustive H=1 single-control
planning (`argmax_v R` over `V`), rolling the SEMANTICS §1.1 call list (one warm-up `advance`
UNSCORED, one terminal `advance` SCORED at `k2`). Downstream trained arm:
`cdd_oran.planners.qacm.QACM.act(..., conflict_param_index=0, ...)`. `JointMultiNCPPlanner` is
not used.
