# FROZEN pre-registration — E2 discovery via RCoT (conditional-independence test)

> **SUPERSEDED by RCoT-v2 (2026-09-22).** This is RCoT-**v1** (`discovery_rcot.py`, low recall R 0.163). The
> canonical frozen E2 method is **RCoT-v2** (`E2_RCOT_DISCOVERY_PROTOCOL_V2.md`, plan 009,
> `cdd_oran/e2slice/discovery_rcot_v2.py`, `block_perm_reps` 99→299, F1 0.855). Use v2 for the live pipeline;
> v1 kept as the retired predecessor.

> **STATUS: FROZEN.** This document is the frozen pre-registration contract, human-approved with the
> primary null pinned to **`block_perm`** (§7.3). **Committing this document is the freeze point:** its
> commit SHA is recorded as `protocol_commit` inside every `discovery_rcot.json` artifact, and the
> executable (`cdd_oran/e2slice/discovery_rcot.py`) is locked to the §11 constants at that same freeze
> (its `PROTOCOL_COMMIT` = this document's SHA, config re-derived fail-closed on load). From this commit
> forward the anti-p-hacking ordering below is in force: no constant may change after any E2 result is
> inspected.

This contract is **frozen before it is executed**. Its commit SHA is recorded as `protocol_commit`
inside every artifact this method produces (the persisted `discovery_rcot.json` mask and its
descendants). The ordering is the anti-p-hacking guarantee: the discovery *method* is fixed on disk
before it is run against E2 truth — that is, before it produces any recovery number. E2 ground truth
is read **only after** the mask has been persisted and hashed.

Nothing in this protocol may be revised to improve a recovery number after this method's recovery
result has been inspected. If the first frozen execution recovers the graph imperfectly (or a guard
fires), that outcome is **recorded and reported as-is** — a null or partial recovery is a valid
scientific result.

Every design choice below is justified in **structural** terms — the declared Gaussian-bump
mechanism *class*, the mixed-scale param/KPI layout, and the numerics of the RCoT construction —
never by any specific E2 edge, adjacency, or coefficient value. This document does not enumerate
E2's true edges, does not import `E2V2Env`, and does not read `true_adj_matrix()`; it is derived only
from the declared generative mechanism, the declared param/KPI layout, and truth-free calibration on
a synthetic testbed (§12).

## Revision history

- **2026-09-06, initial FROZEN version (authored pre-execution; human-approved, primary null `block_perm`).**
  First authored version of the E2 **RCoT**
  discovery contract, frozen by the commit that introduces it. This is a **FRESH** pre-registration at a
  **clean pre-truth point**: NO RCoT
  recovery number against real E2 truth has been computed. It is a sibling to the frozen pdCor
  contract `E2_DISCOVERY_PROTOCOL.md` (`protocol_commit 828e345`), and it deliberately **shares** that
  document's freeze discipline, persistence pattern, fail-closed loading, candidate/target layout,
  conditioning set (the other 13 candidates), per-target BH-FDR selection, relative-collapse guard,
  and guard-fraction HALT. It **replaces** the pdCor edge score (a nonlinear conditional-*association*
  measure whose marginal-permutation null over-selects lagged KPI→KPI edges ~9×) with **RCoT**, a
  genuine **conditional-independence** test whose null preserves the candidate|Z coupling and is
  (near-)nominal and flat in n. The rationale is the a-priori structural argument of §3 plus the
  truth-free calibration evidence of §12, both written before any real-E2 RCoT recovery was run and
  before any real-E2 ground truth was read for this method.

---

## 1. Freeze preamble

- **FROZEN.** This contract is frozen by the commit that introduces it; that commit's SHA is the freeze
  point, recorded as `protocol_commit` in every produced artifact. At the same freeze
  `cdd_oran/e2slice/discovery_rcot.py` is locked to the §11 constants (`PROTOCOL_COMMIT` set, config
  re-derived fail-closed on load) with the primary null pinned to `block_perm`.
- Discovery reads **no environment truth** and imports no true-adjacency symbol. It must not import
  `E2V2Env` or call `true_adj_matrix()`. Dimensions come from the persisted row shapes. (The shipping
  module already satisfies this: it conditions on `np.delete(xs, i, axis=1)` and touches no truth.)
- E2 ground truth is read only after the discovery mask is persisted and hashed. Recovery scoring
  lives in a separate record (§9) that can never feed back into selection.
- Imperfect recovery is a recorded result, not a defect to tune away. Revising the method after its
  recovery number is seen is forbidden (§13, §15).
- **All numerical constants below become FROZEN at the freeze.** Once frozen they MUST NOT be reduced,
  raised, or otherwise changed after any E2 result — including any E2 dataset, score, or recovery
  number — has been inspected. The complete frozen-constant table is §11.

## 2. What this plan is — and is NOT (the firewall)

Two separate things must never be conflated, and this separation is a STOP condition (§15):

1. **E2 RCoT discovery is METHOD validation** — a corrected, better-calibrated discovery test for the
   nonlinear, non-monotonic E2 mechanism class. A green E2 discovery would extend the label-free
   pipeline's validated operating range to nonlinear mechanisms with a *conditional-independence* test.
   It says **nothing** about whether E2's decision gap is real.

2. **E2's preregistered DECISION gate is a recorded NULL and STAYS one.** `GATE_CONTRACT_E2.md`
   (committed `8d70a4f`) tested a shared-control decision gap; the recorded result is a REAL,
   adversarially reviewed **FAIL**, encoded `xfail(strict=True)`. This protocol does **not** touch,
   reopen, improve, or re-tune that gate. E2 discovery is **methodological/exploratory — it is NOT
   evidence for the E2 decision-value hypothesis.** A *failed* E2 discovery means "this method did not
   establish nonlinear recovery here," NOT "the E2 decision hypothesis is disproven." A *green* E2
   discovery means "the pipeline recovers this nonlinear structure," NOT "the E2 decision gap is real."
   Presenting E2 discovery as decision-value evidence, or reopening/re-tuning the recorded E2 decision
   null, is a STOP (§15).

## 3. Root cause — why RCoT, and why a marginal-permutation null MUST be replaced (a-priori, structural)

Every E2 KPI is a **Gaussian bump** in its parents (declared mechanism class, `cdd_oran/envs/v2/e2.py`,
Env II verbatim), schematically:

```
K_j = A_j * exp( -( sum(shift_parents_j) - c_j )^2 / ( 2 * safe_exp(width_parent_j)^2 ) )
```

where `A_j`, `c_j` are constants, `shift_parents_j` are the parents that enter the numerator offset,
`width_parent_j` is the parent that sets the bump width, and `safe_exp(x) = x if |x| > 0.1 else 0.1`.
No reference is made here to *which* params are the shift or width parents of *which* KPI — only the
mechanism *class* is used. Two structural properties of this class defeat any **linear** residual
score (partial correlation, OLS `|beta|`, Pearson/Spearman): the shift parents enter through an
approximately even function so linear/rank correlation is ~0 while the functional dependence is total;
and width parents enter the denominator as a heteroscedastic scale channel that a linear conditional-
mean score sees almost nothing of. So the E2 edge score must be a **general nonlinear dependence
measure**, not a linear one.

**But the choice of a nonlinear score is not enough — the NULL matters.** The frozen sibling scores
`|pdCor|` (a nonlinear conditional-*association* measure) and tests it with a **marginal permutation**
of the candidate column. That permutation is **miscalibrated for a lagged-KPI candidate**: a lagged-KPI
candidate shares its generating exogenous params with sibling lagged KPIs that live in the conditioning
set `Z`. A marginal (global) permutation destroys the candidate|Z coupling, so the permutation null
collapses too tight and a tiny partial statistic rejects — the mechanism behind the observed ~9×
over-selection of lagged KPI→KPI edges. (This is the marginal-vs-conditional gap the frozen sibling
itself flagged as N1/§7.1.)

**RCoT (Randomized Conditional Correlation Test) is a genuine CONDITIONAL-independence test** that
fixes this at the null level. It residualizes **both** the candidate and the target on random Fourier
features (RFF) of `Z`, then tests the residuals for dependence, so the candidate|Z coupling is
**preserved** and the null is (near-)nominal and **flat in n** (§12). This is the a-priori structural
justification for §6, stated in structural terms only, with no reference to any recovered edge or
magnitude. (Linear scores, marginal/unconditional dependence, and the marginal-permutation pdCor null
are rejected as primary; pdCor is retained only as the recorded, retired naive baseline, §14.)

## 4. Data generation (frozen)

The observational dataset is the **same frozen E2 dataset** used by the pdCor sibling: generated from
the `E2V2Env` **TRUE SCM with the decoy OFF** (`decoy_omit_p0_k5 = False`); the decoy device belongs to
the decision gate, not to discovery.

- **Exogenous parameters.** For each row, the eight exogenous params `P0..P7` are sampled **mutually
  independently**, each **uniform over its registered ID range**, so that any non-parent has exact
  **zero population dependence** on a target. The registered ID ranges (declared layout,
  `E2V2Env.id_ranges`) are `P0 [-100,100]`, `P1 [-10,50]`, `P2 [-20,20]`, `P3 [-60,60]`,
  `P4 [-20,20]`, `P5 [-50,150]`, `P6 [-60,65]`, `P7 [-100,150]`.
- **Alignment across the one-step actuation latency.** Per row, the input vector is the committed
  pre-actuation state `X = [P0..P7 (t) | K0..K5 (t)]` (14 columns), and the target is the resulting
  next KPI vector `Y = [K0..K5 (t+1)]` (6 columns), matching E1's `s_t -> k_{t+1}` layout.
- **Noiseless.** `obs_noise_scale = 0.0`; scoring is on the noiseless latent KPI values. E2's
  difficulty comes from nonlinearity and conditioning, **not** from added noise.
- **Rows.** `N_rows_per_seed = 4000`.
- **Content hash.** The generated dataset is content-hashed; the hash is a parent hash of the
  discovery artifact (§8) and binds it fail-closed.

## 5. Candidate graph (frozen)

Identical layout to the pdCor sibling — score-matrix shape **`(6, 14)`**:

- Inputs (candidates) `X = [P0..P7, K0_t..K5_t]` — 8 params + 6 **lagged** KPIs (14 total).
- Targets `Y = [K0_{t+1}..K5_{t+1}]` — 6 next-state KPIs.
- **48 NCP→KPI candidates** (`6 x 8`) and **36 lagged KPI→KPI candidates** (`6 x 6`), **84 candidates
  per seed** in total.
- The 6 KPI candidates are the **lagged** `K_t`, **not** contemporaneous KPIs — there are **no
  same-time KPI→KPI edges**. RCoT is nondirectional; **direction comes solely from the frozen temporal
  ordering `t -> t+1`.**
- Ground truth (used only for post-persistence recovery scoring, §9, never for selection) contains
  true NCP→KPI edges and **0 true KPI→KPI edges** — stated only as the structural fact that all 36
  KPI→KPI candidates are true-negatives the method must reject; *which* NCP→KPI edges are true is
  deliberately **not** enumerated in this contract. The KPI→KPI false-positive count (of 36) and
  rejection rate are reported explicitly (§9, §10). Because the real E2 mechanism has **zero** true
  KPI→KPI edges, the injected-edge power numbers in §12 are a **robustness probe**, not a real recovery
  target; the real recovery targets are the NCP→KPI parents.

## 6. Edge test — RCoT (Randomized Conditional Correlation Test)

For each of the 84 (target KPI `j`, candidate `i`) pairs, RCoT tests the **conditional-independence**
hypothesis `cand_i ⟂ target_j | Z`, with `Z` = the **stack of all 13 candidates EXCEPT `i`**
(identical conditioning to the pdCor sibling, `np.delete(xs, i, axis=1)`). Unlike `|pdCor|`, RCoT is a
genuine CI test, not merely an association score. The exact numerics below are **transcribed from the
already-built, adversarially-reviewed module** `cdd_oran/e2slice/discovery_rcot.py` (which itself lifts
them verbatim from the verified reference `runs/calib-study/calib_study.py`); they are not invented
here.

### 6.1 Per-column z-standardization (frozen, and NOT an affine no-op)

Before any test, **every candidate column (14) and every target column (6) is z-standardized once, up
front, using that column's mean and population standard deviation (`ddof = 0`)** on the frozen dataset
(`_standardize` in the module; zero-variance columns are rejected).

This standardization is a **DELIBERATE, non-invariant preconditioning of the multivariate conditioner
`Z` — NOT an affine no-op.** The 13-dim `Z` bandwidth is a **single joint Euclidean median-heuristic
`sig`** computed over all 13 columns at once. Real E2 columns span wildly different scales (params
~±100, KPIs ~±40); without standardization the largest-scale columns dominate the median pairwise
distance and swamp the smaller-scale columns. Per-column z-standardization **rebalances that joint
bandwidth** so every column contributes on a comparable scale. Only the 1-D candidate and 1-D target
legs are affine-invariant (their per-variable `sig` rescales with the data); the multivariate-`Z` leg
is **not** — an adversarial review measured `Δp` up to 0.307 on the analytic joint-`Z` p-value from
per-column rescaling of `Z`. The **false "affine no-op" claim was caught in that review and
corrected**: the review further verified standardization **keeps/improves** calibration on the faithful
E2 null (e.g. NULL_k3 FP@.05 0.093 → 0.067 at n=400). It stays because it is beneficial and correct,
not because it is numerically inert.

### 6.2 The RCoT numerics (write them out)

- **Median-heuristic bandwidth.** For each leg (`X`, `Y`, `Z`), the RBF bandwidth `sig` is the median
  of pairwise Euclidean distances over the first ≤200 rows (`_median_sigma`; `med` if `> 0` else `1.0`).
- **Random Fourier features (RFF).** For a variable `V` of intrinsic dimension `dim`,
  `f = sqrt(2/D) * cos(V @ W + b)` with `W ~ Normal(0,1)/sig` shape `(dim, D)` and `b ~ U(0, 2π)`
  length `D`, then mean-centered per column (`_rff`). Feature counts: **`dxy = 5`** for the candidate
  and target legs; **`dz = 25`** for the `Z` conditioner. `dz = 25` is a **deliberate FROZEN value**:
  the calibration study selected it for near-nominal FP that is **flat in n**. **Higher `dz`
  OVER-rejects at small n** (more RFF capacity → the residualization overfits the null coupling and the
  statistic inflates), which is exactly the failure mode being avoided — so `dz` is **not** a tunable
  "fix" to be raised; it is a locked constant.
- **Residualization.** Draw `fx = RFF(X)`, `fy = RFF(Y)`, `fz = RFF(Z)`. Solve the ridge least-squares
  `G = fz.T @ fz + ridge * n * I` (with **`ridge = 1e-6`**), `Bx = solve(G, fz.T @ fx)`,
  `By = solve(G, fz.T @ fy)`; residuals `rx = fx - fz @ Bx`, `ry = fy - fz @ By` (`_rff_residuals`).
- **Statistic.** `T = n * ||Cxy||_F^2`, where `Cxy` is the mean over rows of the per-row Kronecker
  product `kron(rx, ry)` (`_rcot_statistic`): `prod = kron(rx, ry)` per row, `cmean = prod.mean(0)`,
  `T = n * (cmean @ cmean)`.

### 6.3 Relative residual-collapse guard (frozen)

RCoT is undefined when the candidate and target are (near-)fully explained by `Z`. The guard fires when

```
sqrt( (||rx||^2 / ||fx||^2) * (||ry||^2 / ||fy||^2) ) <= residual_epsilon
```

with **`residual_epsilon = 1e-12`** (`_residual_collapse`) — the RFF residuals of BOTH the candidate
and the target have collapsed relative to their raw RFF energy. This is the RCoT analogue of the pdCor
sibling's relative denominator guard. On a guard fire the pair gets **`statistic = NaN`, `p = 1`**, is
**reported and counted as guarded**, and is **never selected** (fail-closed) — so it can never lower
another candidate's BH rank in its favour.

## 7. Null variants and selection

### 7.1 Two null variants

The module implements two null distributions for the RCoT statistic:

- **`block_perm` — block-conditional permutation null (RECOMMENDED PRIMARY).** Build a greedy
  nearest-neighbour tour of `Z` and split it into contiguous blocks of length **`block_size = 25`**
  (`z_blocks`); consecutive tour entries have similar `Z`, so permuting the candidate **within** blocks
  preserves the candidate|`Z` coupling. Recompute the RCoT statistic **`block_perm_reps = 99`** times
  with a **FIXED RFF basis** (same per-pair seed each rep, so only `fx` changes); the p-value is
  `(1 + #{stat_perm >= stat_0}) / (block_perm_reps + 1)`. Calibration (§12): **EXACT-nominal — at/below
  nominal and flat in n.** A degenerate block-perm null is fail-SAFE (every permutation reproduces
  `stat_0` → `p → 1`, never selects), so no extra HALT is needed.
- **`analytic_hbe` — analytic Hall–Buckley–Eagleson weighted-χ² null (FAST SENSITIVITY VARIANT).**
  Form `Sigma = (prodc.T @ prodc) / n` from the centered products, take its positive eigenweights, and
  return the survival of a matched Gamma via a regularized-incomplete-gamma computation (`_hbe_pvalue`,
  `_gamma_sf`). Fast; **mildly liberal, and its liberality GROWS with n on strong-sibling nulls**
  (§12). Its only degenerate case (empty positive spectrum) already returns `p = 1`.

### 7.2 Selection — per-target BH-FDR (frozen)

Within each target row `j`, the 14 candidate p-values are subjected to **Benjamini–Hochberg FDR control
at `q = 0.05`** (`m = 14` tests per target), reusing the **exact frozen `bh_fdr_reject`** helper
imported from the pdCor `discovery.py`. A candidate edge is **selected iff it passes BH** within its
target's row; multiplicity is controlled **per target, never pooled** across targets. Guarded cells are
forced to `0`. The persisted `binary_mask` is the row-wise stack of the BH decisions, shape `(6, 14)`.

### 7.3 Primary `null_method` (FROZEN = `block_perm`)

> **FROZEN (human-approved).** The primary/LIVE frozen null is **`block_perm`** — it is EXACT-nominal
> and flat in n (§12), precisely the calibration property RCoT exists to buy, and its deployment cost
> (~3.2 min/seed, §12) is trivial. `analytic_hbe` is retained as a fast, clearly-labelled **sensitivity**
> variant only, **never the selector**. The frozen module default and the config lock pin
> `null_method = "block_perm"`; the 10-seed run and all recovery scoring use `block_perm`.

## 8. Persistence (before any ground truth is read)

The discovery artifact `discovery_rcot.json` is written **atomically** and is the point of no return.
**Ground truth only after persistence.** Loading recomputes `content_hash` and refuses to load on any
content-hash / shape mismatch or if a guarded cell is selected (`load_discovery_rcot`, fail-closed). It
records at least: `schema_version` (`"…​.rcot"`); `frozen` flag; parent `dataset_hash`; `n_rows` and
`seed`; per-column standardization moments (mean/std for the 14 candidate + 6 target columns);
`score_method` (`"rcot_conditional_independence"`); `null_method`; `dz`, `dxy`, `ridge`, `q`,
`block_perm_reps`, `block_size`, `residual_epsilon`, `rng_seed`; `statistic` `(6,14)` (NaN where
guarded); `pvalues` `(6,14)` (1.0 where guarded); `guarded_mask` `(6,14)`; `binary_mask` `(6,14)`;
`candidate_shape`; `git_sha` / `git_dirty`; and `content_hash` (canonical SHA-256 over all other
fields).

> **Freeze fields.** The persisted record carries **`protocol_commit`** = this document's commit SHA and
> **`frozen: true`**, mirroring the pdCor sibling's `PROTOCOL_COMMIT` field (see §13). `load_discovery_rcot`
> refuses to load any record whose `protocol_commit` or frozen §11 constants do not match the module.

## 9. Recovery scoring (post-persistence ONLY)

After the discovery mask is persisted and hashed — **and only then** — map the `(6, 14)` discovered mask
into the full `(14, 14)` graph (`P + K = 8 + 6`) expected by `recovery_by_edge_type` (only the child
KPI rows `8..13` carry predicted edges; the 8 param-child rows are empty), read
`E2V2Env().true_adj_matrix()` for the first time into a **separate** record that can never feed back
into selection, and report: precision / recall / F1 **overall** and for **NCP→KPI**; the **KPI→KPI
false-positive count (of 36)** and **rejection rate = 1 − FP/36**; per-target metrics; and
`protocol_commit` stamped into the recovery record. The recovery record loads fail-closed. This reuses
the pdCor sibling's `evaluate.py` recovery machinery; the RCoT mask is evaluate-compatible
(`discovered_mask_array`).

## 10. Reporting format (frozen)

- **Per (target, candidate):** RCoT `statistic`, p-value, BH decision, selected bit; guarded cells
  flagged and counted (NaN statistic, `p = 1`, never selected).
- **Per target:** the BH-FDR `q`, the selected parents, the guarded count.
- **Aggregate:** precision / recall / F1 overall and NCP→KPI; **KPI→KPI FP count (of 36) and rejection
  rate**; per seed and across the 10-seed envelope; **no seed dropped**; the `null_method` used.

## 11. Frozen constant table

| Constant                | Value                                                | Section |
|-------------------------|------------------------------------------------------|---------|
| `decoy`                 | OFF (`decoy_omit_p0_k5 = False`)                     | §4      |
| `N_rows_per_seed`       | `4000`                                               | §4      |
| `obs_noise_scale`       | `0.0` (noiseless)                                    | §4      |
| candidate shape         | `(6, 14)` — 48 NCP→KPI + 36 KPI→KPI = 84             | §5      |
| conditioning set `Z`    | the other 13 candidates (`np.delete(xs, i, axis=1)`) | §6      |
| standardization         | per-column z-score, `ddof = 0`, frozen dataset       | §6.1    |
| bandwidth               | median-heuristic (`_median_sigma`, ≤200 rows)        | §6.2    |
| `dxy`                   | `5` (candidate & target RFF features)                | §6.2    |
| `dz`                    | `25` (Z-conditioner RFF features; higher over-rejects at small n) | §6.2 |
| `ridge`                 | `1e-6` (fz residualization ridge)                    | §6.2    |
| statistic               | `T = n * ||Cxy||_F^2`                                | §6.2    |
| `residual_epsilon`      | `1e-12` (relative residual-collapse guard)           | §6.3    |
| `block_perm_reps`       | `99` (block_perm null)                               | §7.1    |
| `block_size`            | `25` (Z-neighbour block length, block_perm null)     | §7.1    |
| BH-FDR `q`              | `0.05` (per target, `m = 14`)                        | §7.2    |
| `rng_seed`              | `0` (base RFF/permutation seed)                      | §6/§7   |
| `max_guard_fraction`    | `0.05` → HALT at `>= ceil(0.05*84) = 5` of 84        | §12/§15 |
| seeds                   | `r = 0..9`; `env_seed = weight_seed = r`             | §13     |
| `sampling_seed`         | `0` (frozen base, inherited from the pdCor dataset)  | §13     |
| `null_method`           | **`block_perm`** (FROZEN — human-approved)            | §7.3    |

All constants above are **truth-free** and were pinned before any real-E2 RCoT recovery was examined.
At this freeze they are locked; none may change after any E2 result is inspected.

## 12. Truth-free calibration evidence + cost gate (pre-registered de-risking)

These are the **pre-registered de-risking numbers**, all measured on the **SHIPPING standardized path**
against a **synthetic faithful testbed** and **NO real-E2 truth**. The testbed's KPI mechanism was
cross-checked **byte-identical** to `E2V2Env._update_kpis` (max abs diff `1.4e-14`). The raw study
lives in `runs/calib-study/` (gitignored); the study code is **not** copied into this doc. NO RCoT
recovery number against real E2 truth has been computed — this is the clean pre-truth point.

**False-positive rate @ 0.05 on true-null cells** (reps = 500 analytic / 150 block_perm; n = 400 / 1000 / 2000):

| Null           | ANCHOR              | NULL_k3             | NULL_k0             |
|----------------|---------------------|---------------------|---------------------|
| `analytic_hbe` | 0.050 / 0.078 / 0.046 | 0.068 / 0.106 / 0.136 | 0.078 / 0.130 / 0.164 |
| `block_perm`   | 0.047 / 0.040 / 0.033 | 0.013 / 0.080 / 0.107 | 0.013 / 0.013 / 0.020 |

- `analytic_hbe`: **liberality GROWS with n** on strong-sibling nulls (NULL_k3/NULL_k0 climb with n).
- `block_perm`: **at/below nominal and flat in n** — the calibration property RCoT exists to buy.

**Power @ 0.05 (n = 1000):**

| Null           | parent | injected edge a = 0.10 / 0.20 / 0.40 |
|----------------|--------|--------------------------------------|
| `analytic_hbe` | 0.987  | 0.600 / 0.897 / 0.987                |
| `block_perm`   | 0.967  | 0.458 / 0.792 / 0.950                |

(Real E2 has **ZERO** true KPI→KPI edges, so injected-edge power is a **robustness probe**, not a real
recovery target. The real recovery targets are NCP→KPI parents, where **both** variants are ~0.97–0.99.)

**Cost gate (single-core, deployment n = 4000):**

| Null           | per cell   | per seed (84 cells)      | all 10 seeds |
|----------------|------------|--------------------------|--------------|
| `analytic_hbe` | 0.0091 s   | 0.76 s                   | ~7.6 s       |
| `block_perm`   | 2.32 s     | **194.6 s (~3.2 min)**   | **~32 min**  |

Both are **trivial** vs the retired pdCor method's ~8 h/seed. `block_perm` at ~32 min for the full
10-seed envelope is the recommended primary.

**Module self-checks (independent RCoT null):** analytic FP 0.065, block_perm FP 0.037, power 0.99.
All 16 unit tests pass single-core.

## 13. Freeze / lock mechanism

The freeze executes these steps in order:

1. **Commit this document.** Its git commit SHA is the freeze point.
2. **Record the SHA in code.** `PROTOCOL_COMMIT = "<this document's full sha>"` is set in
   `cdd_oran/e2slice/discovery_rcot.py`, mirroring `discovery.py`'s `PROTOCOL_COMMIT = "828e3458…"`.
   Every `discovery_rcot.json` records this `protocol_commit` and sets `frozen: true`, so the frozen
   contract provably predates any result.
3. **Lock the config.** `frozen_config()` returns the §11 values with `null_method = "block_perm"`; the
   canonical persist path (`write_discovery_rcot`) uses it, and `load_discovery_rcot` **re-derives every
   frozen §11 constant and the `protocol_commit`, refusing to load on any mismatch** — matching
   `discovery.py`'s frozen-constant re-derivation. The primary `null_method` (§7.3) is pinned to
   `block_perm` here.

Committing the code in step 2–3 (with `PROTOCOL_COMMIT` set to the step-1 SHA) follows the document
commit, so the executable provably references a contract that predates it, and no run precedes the lock.

## 14. Retention of the retired pdCor baseline

The frozen pdCor method (`E2_DISCOVERY_PROTOCOL.md` / `discovery.py`, `protocol_commit 828e345`) stays
**untouched on disk** as the recorded, **retired naive baseline**. Its recorded naive result stands as
the point of comparison and is **not** re-run or re-tuned by this protocol:

- seed-0: **P 0.929 / R 0.813 / F1 0.867**, **KPI→KPI 17 FP / 36**, **8.65 h**.

RCoT does not modify, reopen, or improve that record; it is a separate, better-calibrated method
pre-registered afresh.

## 15. Run + recovery-scoring plan (post-freeze)

Executed **only after** the freeze (§13):

1. **Truth-free smoke** (tiny n, few reps, no truth) to confirm the pipeline runs within resource
   limits. Throwaway constants; results discarded; §11 constants never reduced.
2. **Discovery, all 10 seeds** (`r = 0..9`) at the frozen constants with the frozen primary
   `null_method`. At `block_perm` this is **~32 min total** (§12). Persist + content-hash every
   `discovery_rcot.json` mask FIRST; verify all 10 fail-closed-loadable.
3. **Numerical HALTs (broken-run guards, NOT scientific nulls):** NaN/inf outside the guard path →
   HALT; residual-collapse guard fires on `>= ceil(0.05*84) = 5` of 84 candidates in a seed → **HALT
   that seed** and diagnose the RFF residualization / standardization (parity with the pdCor §11 guard-
   fraction HALT). Record guarded counts; never silently drop edges.
4. **ONLY THEN** run recovery scoring (§9) vs `E2V2Env().true_adj_matrix()` into a separate fail-closed
   record; report per-seed + aggregate P/R/F1 overall and NCP→KPI, KPI→KPI FP-count / rejection rate,
   `null_method`; **no seed dropped**. Compare against the retired pdCor baseline (§14).
5. Write the as-run result doc and a dated delta report **only after** results exist.

## 16. Caveats, interpretation limits, and STOP conditions

**Interpretation limits:**

1. **CI test, but direction is not certified.** RCoT tests conditional independence; **direction** is
   supplied **only** by the frozen temporal ordering `t -> t+1` (§5), not by the test.
2. **Full-conditioning caveat.** Conditioning each candidate on all 13 others can overcondition or open
   collider paths when lagged-KPI candidates are descendants; the test certifies conditional
   independence, not causal structure.
3. **Noiseless-control caveat.** E2 is noiseless; do NOT add noise, alter the SCM, add the decoy, or
   restrict the candidate graph to manufacture a cleaner recovery. Because each `Y_j` is a deterministic
   function of a subset of `Z`, some non-parent behavior may be non-clean — reported as-run.
4. **Firewall.** E2 discovery is method validation, **NOT** evidence for the E2 decision-value
   hypothesis. The recorded E2 decision null (`GATE_CONTRACT_E2.md`, `8d70a4f`) stands untouched. A
   partial or null recovery is a valid recorded boundary result, not a defect to tune away.

**STOP conditions (scientific — halt and escalate rather than proceed):**

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The score, null, or threshold is chosen or adjusted to hit specific E2 edges, or revised after its
  recovery number is inspected — including raising `dz` as a "fix".
- Recovery is inconsistent across seeds and the response is to drop or reweight seeds.
- Any attempt to present E2 discovery as evidence for the E2 decision-value hypothesis, or to reopen /
  re-tune / "improve" the recorded E2 decision null.
- Adding noise, the decoy, or any SCM change to manufacture a cleaner recovery.
- A training/planning arm is added without the project intentionally registering a *new* E2 benchmark
  first.

- If a reviewer rejects this contract **before** its first execution, revise it and record why in the
  Revision history. **After** first execution, do not revise it to improve recovery.
