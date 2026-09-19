# E4 deconfounding — feasibility sketch (TRUTH-FREE)

Date: 2026-09-07. Status: **DEV feasibility check only.** Not the scored E4 gate arm, not a
reusable module — both remain **USER-GATED** per `docs/benchmark/GATE_CONTRACT_E4.md` "Explicitly
deferred". Nothing here is committed, no tracked source was modified, no module was built.
Throwaway sketch: `<scratchpad>/e4_feas/feas.py`.

## Question
Can a deconfounded identification strategy recover the true structural slope (alpha ≈ −1) on the
**real E4 DGP**, where the naive correlational pooler catastrophically cannot (≈ +3.29)?

## Confirmed E4 confounding mechanism (read from `cdd_oran/envs/v2/e4.py`, not paraphrase)
- SCM: `K_out(q+1) = alpha*A_q + theta*Z_q + c`, with `alpha=-1.0`, `theta=+2.5`, `c=0.0`.
- `Z ~ N(0,1)` **latent** (never a param/KPI, never exposed); `eta ~ N(0, 0.5^2)`.
- OBS (`mode="obs"`): `A_behavior = clip(0.5 + lambda*Z + eta, 0, 1)`, lambda=1 — the **same Z**
  then enters `K_out`, so Z is an action-relevant backdoor confounder (Z→A_behavior, Z→K_out).
- DO (`mode="do"`): action drawn uniformly over the do-grid **independent of Z**; Z still enters
  `K_out`. This is the identification lever — do() breaks the Z→A edge.
- True structural effect / authoritative scoring: `score_interventional_mean(a) = alpha*a + c = −a`.
- Frozen naive 90:10 pooled line (contract): `b_pool = 3.291569669211391` — a robust **sign
  reversal** vs the true `alpha = −1`, not a knife-edge.

## Method (pre-declared, no tuning)
- Corpus per seed generated from the **real `E4V2Env`** via its own
  `generate_action → apply_action → advance` loop (same route as the gate's `generate_corpus`).
- Matched **90:10** obs:do mix: `n_obs=18000`, `n_do=2000`. Seeds = `env_seed 0..9` (10 independent
  corpora; the coordinate-keyed tape makes a time-walk i.i.d.). Single-core (`*_NUM_THREADS=1`).
- (a) **pooler** = OLS `K ~ A` over the full pooled corpus (ignores D, omits latent Z).
- (b) **deconfounded** = OLS `K ~ A` over the **D=1 (interventional) subset only** — the route the
  contract designs for ("aware D=1 regression uses only the interventional subset").
- (c) **oracle backdoor** (REFERENCE ONLY; Z is latent so this is not deployable) = OLS `K ~ A + Z`
  on obs rows, to confirm the backdoor set {Z} also identifies alpha.

## Per-seed slope estimates

| seed | pooler (K~A pool) | deconf (K~A \| D=1) | backdoor (K~A+Z) |
|-----:|------------------:|--------------------:|-----------------:|
| 0 | 3.2866 | −1.3555 | −1.0000 |
| 1 | 3.2926 | −0.8701 | −1.0000 |
| 2 | 3.3493 | −1.0010 | −1.0000 |
| 3 | 3.2979 | −0.8937 | −1.0000 |
| 4 | 3.3055 | −0.9092 | −1.0000 |
| 5 | 3.2761 | −1.4501 | −1.0000 |
| 6 | 3.3177 | −0.9867 | −1.0000 |
| 7 | 3.2902 | −1.3714 | −1.0000 |
| 8 | 3.3069 | −0.8912 | −1.0000 |
| 9 | 3.3234 | −0.6940 | −1.0000 |

## Mean / 95% CI across 10 seeds

| estimator | mean | 95% CI | sd | CI excludes 0 | CI < 0 (correct sign) |
|-----------|-----:|--------|----|:-------------:|:---------------------:|
| pooler   | **+3.3046** | [+3.2914, +3.3178] | 0.021 | yes | **no (wrong sign)** |
| deconf   | **−1.0423** | [−1.2011, −0.8834] | 0.256 | **yes** | **YES** |
| backdoor | −1.0000 | [−1.0000, −1.0000] | 0.000 | yes | yes |

Pooler mean +3.30 reproduces the frozen +3.2916 (tiny +0.013 finite-sample offset at exact 90:10).

## Verdict: **VIABLE route** (not a null)
The deconfounded D=1-restricted regression **recovers the correct sign and slope**
(mean −1.04, 95% CI [−1.20, −0.88]) exactly where the pooler is sign-reversed at +3.30. The
deconf CI **excludes 0 and lies entirely below 0**, robustly across all 10 seeds. The route works
**in principle on the E4 DGP**.

Honest caveats (report as-is, no escalation):
- Per-seed deconf slope is **noisy** (range −0.69 … −1.45, sd 0.256): the D=1 residual is
  `theta*Z` (var 6.25) and only 2000 do-rows enter each corpus, so a single seed can land as far as
  −0.69. Robustness here comes from the multi-seed mean, not any single seed. At the deferred gate's
  finite-sample n / seed budget this borderline per-seed spread is the risk to watch — the prior
  5-seed deconfounding null (CIs included 0) is consistent with under-powered do-subsets, not with
  the identification failing.
- The oracle `K~A+Z` gives exactly −1 because the mechanism is noiseless given Z; it only confirms
  {Z} is a valid backdoor and is **not a deployable estimator** (Z is latent). The deployable route
  is the do-restricted regression (b), which is what the "VIABLE" verdict rests on.

## Scope reminder
This sketch answers the feasibility question only. The **scored E4 gate arm** (structure-aware vs
correlational vs discovered vs dense on the matched 90:10 corpus, `pi_E4≈0.50`,
`tau_E4_trained≈0.10`) and any **reusable corpus/estimator module** remain **DEFERRED / USER-GATED**
and were deliberately NOT built here. No freeze, no pre-registration, nothing committed.
