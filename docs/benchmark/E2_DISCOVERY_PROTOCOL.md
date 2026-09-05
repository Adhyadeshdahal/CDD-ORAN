# Frozen E2 discovery protocol — U-centered partial distance correlation (contract)

This contract is **frozen before it is executed**. Once committed, this document's commit SHA is
recorded as `protocol_commit` inside every artifact this method produces (the persisted discovery
mask and its descendants). The ordering is the anti-p-hacking guarantee: the discovery *method* is
fixed on disk before it is run against E2 truth — that is, before it produces any recovery number.
E2 ground truth is read **only after** the mask has been persisted and hashed.

Nothing in this protocol may be revised to improve a recovery number after this method's recovery
result has been inspected. If the first frozen execution recovers the graph imperfectly (or a guard
fires), that outcome is **recorded and reported as-is** — a null or partial recovery is a valid
scientific result and is expected to be less clean than E1 (§13).

Every design choice below is justified in **structural** terms — the declared Gaussian-bump
mechanism *class* and the algebra of the distance-geometry construction — never by any specific E2
edge, adjacency, or coefficient value. This document does not enumerate E2's true edges, does not
import `E2V2Env`, and does not read `true_adj_matrix()`; it is derived only from the declared
generative mechanism and the declared param/KPI layout.

## Revision history

- **2026-09-05, initial freeze (pre-execution).** First authored version of the E2 discovery
  contract. Supersedes nothing; it is a *fresh* freeze for a nonlinear, non-monotonic mechanism
  class, sibling to `E1_DISCOVERY_PROTOCOL_V2.md`. It shares E1's freeze discipline, persistence
  pattern, and fail-closed loading, but replaces the linear partial-correlation score (E1 §3) with a
  nonlinear U-centered partial distance correlation, and replaces the per-target `largest_gap`
  selector (E1 §4) with a per-candidate permutation null under per-target BH-FDR. The rationale is
  the a-priori structural argument of §3, written before any E2 discovery ran and before any E2
  ground truth was read.
- **2026-09-05, pre-execution amendment (before any execution).** Added N1 (permutation FDR is
  approximate — marginal-permutation vs conditional-null gap, §7.1/§15), N2 (noiseless-determinism
  × full-conditioning caveat, §15.3), N3 (pinned the exact `SeedSequence` derivation, §12).
  Structural/honesty/reproducibility only — no method or frozen-constant change. Made on
  orchestrator+user review before the freeze commit and before any E2 execution.

---

## 1. Freeze preamble

- This contract is frozen before execution. Its git commit SHA is the freeze point and is recorded
  as `protocol_commit` in every produced artifact.
- Discovery reads **no environment truth** and imports no true-adjacency symbol. It must not import
  `E2V2Env` or call `true_adj_matrix()`. Dimensions come from the persisted row shapes.
- E2 ground truth is read only after the discovery mask is persisted and hashed. Recovery scoring
  lives in a separate record (§9) that can never feed back into selection.
- Imperfect recovery is a recorded result, not a defect to tune away. Revising the method after its
  recovery number is seen is forbidden (§13, §15).
- **All numerical constants below are FROZEN.** They MUST NOT be reduced, raised, or otherwise
  changed after any E2 result — including any E2 dataset, score, or recovery number — has been
  inspected. The complete frozen-constant table is §14.

## 2. What this plan is — and is NOT (the firewall)

Two separate things must never be conflated, and this separation is a STOP condition (§15):

1. **E2 discovery is method validation, extending E1's cleared discovery-method prerequisite to a
   nonlinear, non-monotonic mechanism class.** A green E2 discovery would extend the label-free
   pipeline's validated operating range to nonlinear mechanisms. It says **nothing** about whether
   E2's decision gap is real.

2. **E2's preregistered DECISION gate is a recorded NULL and STAYS one.** `GATE_CONTRACT_E2.md`
   (committed `8d70a4f`) tested a shared-control decision gap; the recorded result is a REAL,
   adversarially reviewed **FAIL**, encoded `xfail(strict=True)`. This protocol does **not** touch,
   reopen, improve, or re-tune that gate. E2 discovery is **methodological/exploratory — it is NOT
   evidence for the E2 decision-value hypothesis.** A *failed* E2 discovery means "this method did
   not establish nonlinear recovery here," NOT "the E2 decision hypothesis is disproven." A *green*
   E2 discovery means "the pipeline recovers this nonlinear structure," NOT "the E2 decision gap is
   real." Presenting E2 discovery as decision-value evidence, or reopening/re-tuning the recorded
   E2 decision null, is a STOP (§15).

## 3. Root cause: why a linear residual score MUST NOT be carried over (a-priori, structural)

Every E2 KPI is a **Gaussian bump** in its parents (declared mechanism class,
`cdd_oran/envs/v2/e2.py`, Env II verbatim), schematically:

```
K_j = A_j * exp( -( sum(shift_parents_j) - c_j )^2 / ( 2 * safe_exp(width_parent_j)^2 ) )
```

where `A_j`, `c_j` are constants, `shift_parents_j` are the parents that enter the numerator offset,
`width_parent_j` is the parent that sets the bump width, and `safe_exp(x) = x if |x| > 0.1 else 0.1`.
No reference is made here to *which* params are the shift or width parents of *which* KPI — only the
mechanism *class* is used. Two structural properties of this class defeat any linear residual score
(partial correlation, OLS `|beta|`, Pearson/Spearman):

- **Non-monotonic, near-even dependence in the shift parents.** The bump rises to a peak and falls;
  where a shift parent's registered range straddles the center `c_j`, the induced dependence is
  approximately symmetric about the peak, so **linear and rank correlation are ~0 while the
  functional dependence is total**. A parent that enters the offset through an (approximately) even
  function of its own range has near-zero partial correlation despite being a true parent.
- **Width parents enter the denominator (a scale channel).** A width parent modulates the bump's
  spread, not the conditional mean linearly. Its influence is a heteroscedastic / scale effect that
  a linear conditional-mean score sees almost nothing of.

Therefore the E2 edge score must be a **general statistical dependence measure** — sensitive to
arbitrary non-monotonic and scale-channel association — not a linear one. This is the a-priori
justification for §6, stated in structural terms only, with no reference to any recovered edge or
magnitude. (Linear scores, marginal/unconditional dependence, and pooled thresholding are rejected
as primary; they may appear only as clearly labelled diagnostics that never alter the persisted
mask.)

## 4. Data generation (frozen)

The observational dataset is generated from the `E2V2Env` **TRUE SCM with the decoy OFF**
(`decoy_omit_p0_k5 = False`); the decoy device belongs to the decision gate, not to discovery.

- **Exogenous parameters.** For each row, the eight exogenous params `P0..P7` are sampled **mutually
  independently**, each **uniform over its registered ID range**, so that any non-parent has exact
  **zero population dependence** on a target. The registered ID ranges (declared layout,
  `E2V2Env.id_ranges`) are:

  | Param | Range          | Param | Range          |
  |-------|----------------|-------|----------------|
  | P0    | `[-100, 100]`  | P4    | `[-20, 20]`    |
  | P1    | `[-10, 50]`    | P5    | `[-50, 150]`   |
  | P2    | `[-20, 20]`    | P6    | `[-60, 65]`    |
  | P3    | `[-60, 60]`    | P7    | `[-100, 150]`  |

- **Alignment across the one-step actuation latency.** Per row, the input vector is the committed
  pre-actuation state `X = [P0..P7 (t) | K0..K5 (t)]` (14 columns), and the target is the resulting
  next KPI vector `Y = [K0..K5 (t+1)]` (6 columns). `Y = f(P_t)` under the frozen SCM
  (one-step actuation latency), and the lagged `K_t` in `X` is the KPI vector present at the start of
  the step. Rows are aligned so that `X` is `s_t` and `Y` is `k_{t+1}`, matching E1's `s_t -> k_{t+1}`
  layout.
- **Noiseless.** `obs_noise_scale = 0.0`; scoring is on the noiseless latent KPI values, as in E1.
  E2's difficulty comes from nonlinearity and conditioning, **not** from added noise (§13).
- **Rows.** `N_rows_per_seed = 4000`.
- **Content hash.** The generated dataset is content-hashed; the hash is a parent hash of the
  discovery artifact (§8) and binds it fail-closed.

## 5. Candidate graph (frozen)

The candidate directed graph is the **temporal 14-wide layout**, score-matrix shape **`(6, 14)`**:

- Inputs (candidates) `X = [P0..P7, K0_t..K5_t]` — 8 params + 6 **lagged** KPIs (14 total).
- Targets `Y = [K0_{t+1}..K5_{t+1}]` — 6 next-state KPIs.
- This yields **48 NCP->KPI candidates** (`6 x 8`) and **36 lagged KPI->KPI candidates** (`6 x 6`),
  **84 candidates per seed** in total.
- The 6 KPI candidates are the **lagged** `K_t`, **not** contemporaneous KPIs — there are **no
  same-time KPI->KPI edges**. Distance correlation is nondirectional; **direction comes solely from
  the frozen temporal ordering `t -> t+1`.**
- Ground truth (used only for post-persistence recovery scoring, §9, never for selection) contains
  **16 true NCP->KPI edges and 0 true KPI->KPI edges** — stated as expected counts only; *which*
  edges are true is deliberately **not** enumerated in this contract. All 36 KPI->KPI candidates are
  true-negatives the method must reject; the KPI->KPI false-positive count (of 36) and rejection
  rate are reported explicitly (§9, §10).

## 6. Edge score — U-centered partial distance correlation, `|pdCor|`

The edge score is the **magnitude of the U-centered partial distance correlation**, a nonlinear
conditional-**association** score. It is built from the U-centered (unbiased) estimator of squared
distance covariance (Székely–Rizzo 2013/2014) and the partial-distance-correlation projection
construction (Székely–Rizzo 2014).

**This is a nonlinear conditional-ASSOCIATION score, NOT a conditional-independence certificate.**
Ordinary distance correlation's "zero iff independence" does **not** transfer to the partial
statistic: `pdCor = 0` is **not** equivalent to conditional independence (Székely–Rizzo 2014,
explicit). The selection rule (§7) is therefore a benchmark heuristic, not a CI test (§15 caveats).

### 6.1 Distance geometry (frozen choices, not "parameter-free")

The estimator removes a kernel bandwidth, but the distance geometry is a **frozen choice**, pinned
here and re-derivable from the persisted artifact:

- **Standardization.** Every column of the frozen dataset (all 14 candidate columns and all 6 target
  columns) is z-standardized using that column's mean and population standard deviation
  (`ddof = 0`) computed on the frozen dataset. Distance correlation is scale-invariant per variable;
  standardization is fixed for numerical conditioning and to place the stacked conditioning space
  (below) on a common scale.
- **Distance.** Euclidean distance with **distance exponent `alpha = 1`** (i.e. `|u - v|`, not
  `|u - v|^alpha` with `alpha != 1`).
- **The 13-dim conditioning vector.** For a given (target `j`, candidate `i`), the conditioning
  variable `Z` is the **stack of all 13 candidates EXCEPT `i`** (every candidate column other than
  `i`), each standardized as above, assembled column-wise into a single point in `R^13`. Distances
  on `Z` are Euclidean in `R^13` (`alpha = 1`). Thus the candidate `X_i` (1-dim), the target
  `Y_j` (1-dim), and the conditioning `Z` (13-dim) each yield an `n x n` distance matrix.

### 6.2 The estimator (write it out)

For a sample of `n` rows and a variable `V` with `n x n` Euclidean distance matrix `a_{kl}`, the
**U-centered** matrix `Ã` is (Székely–Rizzo 2013/2014):

```
Ã_{kl} = a_{kl} - (1/(n-2)) * sum_m a_{km} - (1/(n-2)) * sum_m a_{ml}
                + (1/((n-1)(n-2))) * sum_{m,o} a_{mo}     for k != l,
Ã_{kk} = 0.
```

The **U-centered inner product** of two U-centered matrices is the unbiased U-statistic (requires
`n >= 4`):

```
(Ã · B̃) = (1 / (n (n - 3))) * sum_{k != l} Ã_{kl} B̃_{kl}.
```

`(Ã · B̃)` is the U-statistic (unbiased) estimator of squared distance covariance
`dCov*^2`, and `sqrt((Ã · Ã))` is the corresponding norm `|Ã|`. Let `Ã`, `B̃`, `C̃` be the
U-centered distance matrices of the candidate `X_i`, the target `Y_j`, and the conditioning `Z`
respectively.

**Partial construction (projection).** Project `Ã` and `B̃` onto the orthogonal complement of `C̃`
in the U-centered inner-product space (Székely–Rizzo 2014):

```
Pxz = Ã - ((Ã · C̃) / (C̃ · C̃)) * C̃          # candidate, conditioning projected out
Pyz = B̃ - ((B̃ · C̃) / (C̃ · C̃)) * C̃          # target, conditioning projected out
```

(when `(C̃ · C̃) > 0`; the guarded case is §6.3). The **partial distance correlation** is the signed
cosine of these projected matrices:

```
pdCor(X_i, Y_j ; Z) = (Pxz · Pyz) / ( |Pxz| * |Pyz| )
```

with `|Pxz| = sqrt((Pxz · Pxz))`, `|Pyz| = sqrt((Pyz · Pyz))`.

- **The statistic is signed and may be negative.** The **edge score is `abs(pdCor)`**; the signed
  value is retained as a reported diagnostic (§10) and is **never clipped** to 0.
- Persisted per candidate: signed `pdCor[j, i]` and the score `scores[j, i] = |pdCor[j, i]|`, both
  `(6, 14)`.

### 6.3 Denominator guard (RELATIVE — frozen)

`pdCor` is undefined when the projected self-norms collapse relative to the raw self-norms. With
`vA_projected = (Pxz · Pxz)`, `vB_projected = (Pyz · Pyz)`, `vA_raw = (Ã · Ã)`, `vB_raw = (B̃ · B̃)`,
the guard fires when:

```
sqrt(vA_projected * vB_projected) <= denominator_epsilon * sqrt(vA_raw * vB_raw)
```

with **`denominator_epsilon = 1e-12`** (a *relative* tolerance; the raw self-norms set the scale, so
the guard is invariant to the overall magnitude of the U-centered distances). On a guard fire:

- `pdCor[j, i] = NaN` (and `scores[j, i] = NaN`);
- the candidate is **reported as guarded** and **counted** (§10);
- the edge is **not selected** (fail-closed);
- the candidate **enters the per-target BH-FDR as `p = 1`** (fail-closed), so it can never be
  selected and never lowers another candidate's BH rank in its favour.

This is the E2 analogue of E1's collinearity STOP: a degenerate candidate is rejected, never
worked around. (A guard rate above the frozen fraction is a numerical HALT, §11.)

## 7. Threshold — per-candidate permutation null, per-target BH-FDR (frozen)

Selection is label-free and truth-free. It never pools scores across targets.

### 7.1 Per-candidate permutation null

For each (candidate `i` -> target `j`), the **conditioning vector `Z` and the target `Y_j` are held
fixed** (so `B̃`, `C̃`, and `Pyz` are unchanged), and the **candidate column `X_i` is permuted** over
its rows. For each permutation, `Ã` is recomputed from the permuted candidate, `Pxz` is recomputed,
and `|pdCor|` is recomputed. Frozen constants:

- **`B_perm = 999`** permutations.
- **`permutation_seed = 0`** (the permutation RNG is seeded from this frozen value; it uses **no
  ground truth**).

The one-sided p-value is:

```
p = (1 + count( null_|pdCor| >= observed_|pdCor| )) / (B_perm + 1)
```

(a guarded candidate does not run a null; it takes `p = 1` per §6.3). A degenerate permutation null
(zero variance across the `B_perm` recomputed `|pdCor|` values) for a **selected** candidate is a
numerical HALT (§11).

The permutation permutes `X_i` **marginally**, but `X_i` is dependent on the conditioning set `Z`
(candidates share exogenous params). Marginal permutation therefore breaks `X_i`–`Z` dependence and
samples the **joint-independence** null, not the conditional null — the known gap between a
permutation test and a conditional randomization test (Candès et al. 2018). The resulting p-values
are not guaranteed valid for the conditional hypothesis, so the **per-target BH-FDR at `q = 0.05` is
an APPROXIMATE selection control, not a guaranteed FDR bound.** The as-run result doc MUST NOT claim
exact FDR control; `q` is a frozen selection knob, not a proven error rate.

### 7.2 Per-target BH-FDR

Within each target row `j`, the 14 candidate p-values are subjected to
**Benjamini–Hochberg FDR control at `q = 0.05`** (`m = 14` tests per target). A candidate edge is
**selected iff it passes BH** within its target's row. Multiplicity is controlled **per target,
never pooled** across targets.

- The persisted `binary_mask` is the row-wise stack of the BH decisions, shape `(6, 14)` of `{0, 1}`.
- **Per-target largest-gap is a reported diagnostic only** (§10) and is **never** the selector.
- No oracle/labelled sweep may replace this primary result; any such sweep appears only as a clearly
  labelled diagnostic and can never alter the persisted mask.

## 8. Persistence (before any ground truth is read)

The discovery artifact is written **atomically** and is the point of no return. **Ground truth only
after persistence.** Loading recomputes `content_hash` and refuses to load on any parent-hash,
shape, numeric, or `protocol_commit` mismatch (the Plan 003 fail-closed pattern). It records at
least:

- `schema_version`;
- parent hash `dataset_hash` (fail-closed binding to the §4 dataset);
- `n_rows` (`= N_rows_per_seed`) and `seed` (the seed `r` of this artifact);
- standardization moments (per-column mean and population std for the 14 candidate and 6 target
  columns);
- `signed_pdcor` `(6, 14)` — the signed partial distance correlations (the signed diagnostic);
- `scores` `(6, 14)` — `|signed_pdcor|`, the persisted edge scores (NaN where guarded);
- `perm_pvalues` `(6, 14)` — the per-candidate permutation p-values (1.0 where guarded);
- `guarded_mask` `(6, 14)` of `{0, 1}` — guard fires (§6.3);
- `score_method` (`"u_centered_partial_distance_correlation"`), `alpha` (`= 1`),
  `denominator_epsilon` (`= 1e-12`);
- `threshold_method` (`"per_candidate_permutation_per_target_bh_fdr"`), `B_perm` (`= 999`),
  `permutation_seed` (`= 0`), `q` (`= 0.05`);
- `binary_mask` `(6, 14)` of `{0, 1}` — the BH selection (§7.2);
- `candidate_shape` `[6, 14]`;
- `protocol_commit` — the git SHA of this frozen-protocol commit;
- `git_sha` / `git_dirty` — provenance of the producing checkout;
- `content_hash` — a canonical SHA-256 over all of the above (every field except `content_hash`).

Loading fails closed unless: `content_hash` recomputes; `scores` equals `|signed_pdcor|` elementwise
where not guarded and is `NaN` exactly where `guarded_mask == 1`; `perm_pvalues == 1.0` wherever
`guarded_mask == 1`; `binary_mask` equals the per-target BH-FDR decision (§7.2) recomputed from
`perm_pvalues` at the frozen `q`; `binary_mask == 0` wherever `guarded_mask == 1`; `protocol_commit`,
`score_method`, `threshold_method`, `alpha`, `denominator_epsilon`, `B_perm`, `permutation_seed`,
and `q` equal the frozen constants; shapes equal `candidate_shape`; and the parent `dataset_hash`
matches. These re-derive values already in the file, so a correctly generated artifact is
byte-unaffected while a silently retuned one is rejected.

## 9. Recovery scoring (post-persistence ONLY)

After the discovery mask is persisted and hashed — **and only then** — map the `(6, 14)` discovered
mask into the full `(14, 14)` graph (`P + K = 8 + 6`) expected by `recovery_by_edge_type`: only the
child KPI rows (`8..13`) carry predicted edges; the 8 param-child rows are empty. Ground truth is
`E2V2Env().true_adj_matrix()`, read here for the first time, into a **separate** record that can
never feed back into selection. Report:

- precision / recall / F1 **overall** and for **NCP->KPI**;
- the **KPI->KPI false-positive count (of 36)** and the **rejection rate = 1 - FP / 36**;
- per-target metrics;
- `protocol_commit` stamped into the recovery record.

The recovery record loads fail-closed (Plan 003 pattern): it refuses on any parent-hash / shape /
numeric / `protocol_commit` mismatch.

## 10. Reporting format (frozen)

Written into the persisted artifacts and the as-run result doc:

- **Per (target, candidate):** signed `pdCor`, `|pdCor|` (the edge score), permutation p-value, BH
  decision, and the selected bit.
- **Negative unbiased estimates** are reported as-is in the signed diagnostic and folded by `abs()`
  for the score — **never silently clipped** to 0.
- **Guarded candidates** (undefined `pdCor`, §6.3) are flagged and counted; they report `NaN` score
  and `p = 1`, and are not selected.
- **Per target:** the BH-FDR `q` used, the selected parents, and the per-target largest-gap
  **diagnostic** value (never the selector).
- **Aggregate:** precision / recall / F1 overall and NCP->KPI; **KPI->KPI FP count (of 36) and
  rejection rate**; per seed and across the 10-seed envelope; **no seed dropped**.

## 11. Numerical halt conditions (broken-run guards, NOT scientific nulls)

These mean "the run is broken; fix the code/geometry, do **not** report a result" — they are
distinct from the scientific STOPs (§15) and from the declared denominator guard (§6.3), which is a
normal, reported outcome:

- `NaN` / `inf` in the scores **outside** the declared denominator-guard path -> **HALT**.
- The denominator guard fires on **`>= ceil(max_guard_fraction * 84) = 5`** of the 84 candidates in
  a seed (`max_guard_fraction = 0.05`) -> **HALT that seed** and diagnose the distance geometry /
  standardization; still report the aggregate guarded counts. Do not silently drop edges.
- A **degenerate permutation null** (zero variance across the recomputed `|pdCor|`) for a
  **selected** candidate -> **HALT**.

## 12. Seed envelope (frozen)

- **10 seeds, `r = 0..9`.** For each seed, `env_seed = weight_seed = r` (the per-seed index; note
  there is no training/weight arm in this plan, so `weight_seed` is inherited nomenclature only).
- **`sampling_seed = 0`** (frozen base). The per-seed exogenous-parameter draw (§4) uses
  `numpy.random.default_rng(numpy.random.SeedSequence(entropy=sampling_seed, spawn_key=(r,)))` with
  `sampling_seed = 0` and seed index `r`. This is the frozen derivation (not an example); the
  generated dataset is content-hashed (§4), which binds the realized draw fail-closed. So the 10
  datasets differ across the envelope while the sampling procedure is fully frozen. The
  `permutation_seed = 0` (§7.1) is a separate frozen constant that does not vary with `r`.
- Per-seed recovery is reported (§9, §10). **No seed is dropped or reweighted.** Recovery that is
  inconsistent across seeds is reported as-run; dropping/reweighting seeds is a STOP (§15).

## 13. Stability-selection cross-check (reported diagnostic, NOT the selector)

As a reported heuristic only, compute each edge's **selection frequency** across the 10-seed
envelope (§12): the fraction of the `B = 10` seeds in which the per-target rule of §7 selects that
edge (Meinshausen–Bühlmann 2010; CPSS Shah–Samworth 2013). With `B = 10` the frequency CI is coarse,
so this is a **reported** confirmation of stability — it is **never** the selector and **never**
alters any persisted mask. Each seed's mask is produced and persisted by the identical frozen
§4–§8 pipeline.

## 14. Frozen constant table

| Constant                | Value                                          | Section |
|-------------------------|------------------------------------------------|---------|
| `decoy`                 | OFF (`decoy_omit_p0_k5 = False`)               | §4      |
| `N_rows_per_seed`       | `4000`                                         | §4      |
| `obs_noise_scale`       | `0.0` (noiseless)                              | §4      |
| candidate shape         | `(6, 14)` — 48 NCP->KPI + 36 KPI->KPI = 84     | §5      |
| standardization         | per-column z-score, `ddof = 0`, frozen dataset | §6.1    |
| distance exponent `alpha` | `1`                                          | §6.1    |
| conditioning dim        | `13` (all candidates except `i`)               | §6.1    |
| `denominator_epsilon`   | `1e-12` (relative guard)                        | §6.3    |
| `B_perm`                | `999`                                          | §7.1    |
| `permutation_seed`      | `0`                                            | §7.1    |
| BH-FDR `q`              | `0.05` (per target, `m = 14`)                  | §7.2    |
| seeds                   | `r = 0..9`; `env_seed = weight_seed = r`        | §12     |
| `sampling_seed`         | `0` (frozen base)                              | §12     |
| `max_guard_fraction`    | `0.05` -> HALT at `>= 5` of 84                 | §11     |

All constants are **truth-free** and were pinned before any E2 truth was examined. None may change
after any E2 result is inspected.

## 15. Caveats, interpretation limits, and STOP conditions

**Interpretation limits (stated in the artifacts and result doc):**

1. **Association, not a CI certificate.** `|pdCor|` is a nonlinear conditional-*association* score;
   `pdCor = 0` is **not** conditional independence (Székely–Rizzo 2014). Selection (§7) is a
   benchmark heuristic, not a conditional-independence test. The permutation null is marginal, so the
   per-target BH-FDR is an APPROXIMATE selection control, not a guaranteed FDR bound (§7.1).
2. **Full-conditioning caveat (causal-interpretation limit).** Conditioning each candidate on all 13
   others is an E1-compatible stress test, but it can **overcondition** or **open collider paths**
   when lagged-KPI candidates are descendants. The score certifies **neither** conditional
   independence **nor** causal direction; direction is supplied **only** by the frozen temporal
   ordering `t -> t+1` (§5).
3. **Noiseless-control caveat.** E2 is noiseless, like E1. Its difficulty comes from nonlinearity and
   the finite-sample cost of conditioning a distance-dependence measure on 13 candidates — not from
   noise. Do **not** add noise, alter the E2 SCM, add the decoy, or restrict the candidate graph to
   manufacture a cleaner recovery (§15 STOPs). Moreover, because each `Y_j` is a **deterministic**
   function of a subset of the conditioning set `Z`, the partial-dCor behavior of non-parent
   candidates is not guaranteed clean — the E2 analogue of E1's target-residual-zero lesson. The
   relative denominator guard (§6.3) catches only true numerical degeneracy (`B̃` numerically
   collinear with `C̃`), NOT the mere fact that `Y_j = f(Z)`, so this manifests as method behavior to
   be **reported as-run** and reinforces that E2 recovery may be partial or null.
4. **Firewall.** E2 discovery is method validation, **NOT** evidence for the E2 decision-value
   hypothesis. The recorded E2 decision null (`GATE_CONTRACT_E2.md`, commit `8d70a4f`) stands
   untouched. **E2 recovery need not be green** — a partial or null recovery is a valid recorded
   boundary result (§2, §13), not a defect to tune away.

**Honest difficulty stance.** Unlike E1, E2 recovery is **not** expected to be a clean 0/1 by
construction. A partial or non-green E2 recovery is a VALID recorded boundary result, provided it is
reported as-run and does not trigger threshold changes, seed dropping, retraining, or reopening the
decision gate.

**STOP conditions (scientific — halt and escalate rather than proceed):**

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The score or threshold is chosen or adjusted to hit specific E2 edges, or revised after its
  recovery number is inspected.
- Recovery is inconsistent across seeds and the response is to drop or reweight seeds.
- **Any attempt to present E2 discovery as evidence for the E2 decision-value hypothesis, or to
  reopen / re-tune / "improve" the recorded E2 decision null.** (Claim-scope firewall.)
- Adding noise, the decoy, or any SCM change to manufacture a cleaner recovery.
- A training/planning arm is added without the project intentionally registering a *new* E2
  benchmark first.

**Operational note.** `N = 4000 x 999 permutations x 84 candidates` is heavy. This protocol
**REQUIRES** a truth-free memory/runtime smoke test (tiny `N`, few permutations, no ground truth)
before the real dataset run, to confirm the pipeline runs within resource limits. **The smoke test
uses throwaway constants and its results are discarded; the frozen constants of §14 MUST NOT be
reduced after seeing any E2 result.**

- If a reviewer rejects this contract **before** its first execution, revise it and record why in
  the Revision history. **After** first execution, do not revise it to improve recovery.
