# E2 RCoT low true-edge recall — diagnosis (2026-09-06)

**Question.** The frozen RCoT E2 discovery method (protocol `eba381a`, primary null `block_perm`)
fixed the KPI→KPI over-selection but recovers only ~2–3 of the 16 true NCP→KPI edges per seed
(recall 0.163, F1 0.251). Is the low recall (i) fixable by a looser null, (ii) fixable by a reduced
conditioning set, (iii) fundamental to E2's noiselessness — or a mix?

**TL;DR.** The low recall is **NOT over-conditioning / "explaining away" and NOT low test power.**
The block_perm conditional test already rejects the true edge at the cell level for **128/160 (80%)**
of true edges (raw p < 0.05). The recall collapses at the **BH-FDR selection step**: block_perm with
`block_perm_reps = 99` has a **hard p-value floor of 1/(99+1) = 0.010**, while per-target BH-FDR at
q = 0.05 with m = 14 requires the leading candidate's p ≤ 0.05/14 = **0.0036**. That threshold is
**unreachable** at 99 permutations, so the strongest true edge can only be selected when ≥3 candidates
happen to tie at the 0.010 floor and lift the BH staircase. The cause is a **BH × permutation-resolution
interaction**, which is **fixable** — and fixable *without* reintroducing the KPI→KPI false positives.

Single-core throughout (`OMP/MKL/OPENBLAS_NUM_THREADS=1`). No frozen artifact modified; no commits.
All scratch in the session scratchpad.

---

## Part A — `analytic_hbe` sensitivity on the 10 frozen E2 seeds (pre-registered variant)

Loaded each frozen dataset `runs/e2slice-recovery/replicate-0r` and ran
`discover_graph_rcot(rows, RCoTDiscoveryConfig(null_method="analytic_hbe"))`, scoring recovery with the
shared evaluate helpers against `E2V2Env(env_seed=0).true_adj_matrix()` (16 true NCP→KPI edges).
**Validation:** re-scoring the frozen `block_perm` masks with this pipeline reproduced every persisted
`recovery_rcot.json` number exactly (`match_all_frozen = True`). Nothing was written to
`discovery_rcot.json` / `recovery_rcot.json`.

| null (Z_full, m=14, BH q=0.05) | NCP→KPI recall | NCP→KPI precision | NCP→KPI F1 | NCP true-pos /160 | KPI→KPI FP /360 |
|---|---|---|---|---|---|
| **block_perm** (frozen) | **0.163** [0.00, 0.44] | 0.674 | 0.251 | 26 | **6** (1.7%) |
| `analytic_hbe` (variant) | **0.769** [0.75, 0.81] | 0.970 | 0.857 | 123 | **30** (8.3%) |

The looser (continuous) null recovers **4.7× the recall** (0.163 → 0.769) at high precision (0.97), but
at **5× the KPI→KPI false-positive rate** (6 → 30 of 360) — i.e. it partly reintroduces the exact
over-selection `block_perm` was frozen to prevent.

**Raw-vs-BH decomposition (real E2, the decisive diagnostic):**

| per 160 true edges | raw p < 0.05 | raw p < 0.011 (≈ floor) | passes BH q=0.05 |
|---|---|---|---|
| block_perm | **128** | 122 | **26** |
| analytic | 129 | 124 | **123** |

Both nulls have **essentially identical raw per-cell power (~80%)**. block_perm's *minimum* true-edge
p-value is **exactly 0.010 on every one of the 10 seeds** (the 1/(B+1) permutation floor); analytic's
minimum is ~1e-16. So block_perm loses recall **entirely at the BH step**, purely because its p-value
resolution cannot clear the BH leading threshold — not because the conditional test fails to see the edge.

**Part A verdict:** the low recall is **almost entirely a null-resolution/selection artifact, not a
power or conditioning problem.** A looser null "fixes" recall but at a real FP cost; per discipline,
`block_perm` remains the frozen result and `analytic_hbe` is reported as a sensitivity variant only.

---

## Part B — over-conditioning study, TRUTH-FREE, on the byte-faithful test-bed

Test-bed = `runs/calib-study/calib_study4_correct.kpi_fn` + `draw_params` (byte-faithful to real E2,
decoy OFF; the test-bed's own construction knows the 16 param→KPI parents). RFF numerics imported
UNCHANGED from `discovery_rcot.py`. Candidate layout = 8 current params + 6 lagged KPIs; targets = 6
current KPIs; per-target BH-FDR at q=0.05, m=14 — the real pipeline. Both **raw** (per-cell rejection
@0.05) and **BH** (per-target selection) are reported. Nothing scored on E2.

**B1 — conditioning-set × null (n=1000, reps=12, all 6 targets):**

| condition | raw power | **BH power** | raw KPI→KPI FP | **BH KPI→KPI FP** |
|---|---|---|---|---|
| analytic  · Z_full        | 0.776 | 0.734 | 0.079 | 0.037 |
| analytic  · Z_params_only | 0.750 | 0.708 | 0.058 | 0.019 |
| analytic  · Z_none (marg.)| 0.755 | 0.719 | 0.067 | 0.021 |
| analytic  · Z_coparents   | 0.760 | 0.729 | n/a | n/a |
| **block_perm(99) · Z_full**        | 0.771 | **0.078** | 0.039 | 0.005 |
| **block_perm(99) · Z_params_only** | 0.760 | **0.078** | 0.035 | 0.007 |

Two facts jump out. (1) **Raw power is ~0.75–0.78 for every conditioning set and both nulls** —
dropping the 6 lagged KPIs (`Z_params_only`), dropping everything (`Z_none`), or keeping only the
target's other true parents (`Z_coparents`) each move power by < 0.03. **The conditioning set is not
the lever; "explaining away" is falsified.** (2) `block_perm(99)` reproduces the real-E2 artifact
truth-free: raw power 0.77 but **BH power collapses to 0.078**, and reducing Z does **not** rescue it.

**B2 — permutation-resolution sweep (n=1000, Z_full, reps=8):** floor = 1/(B+1) vs BH leading
threshold 0.05/14 = 0.00357.

| condition | floor | raw power | **BH power** | **BH KPI→KPI FP** |
|---|---|---|---|---|
| block_perm(99)  · Z_full | 0.0100 | 0.781 | **0.086** | 0.003 |
| block_perm(199) · Z_full | 0.0050 | 0.789 | **0.695** | 0.010 |
| block_perm(299) · Z_full | 0.0033 | 0.797 | **0.742** | 0.010 |
| block_perm(499) · Z_full | 0.0020 | 0.797 | **0.734** | 0.010 |

Raising the permutation count from 99 → ≥199 lifts BH power from **0.086 to ~0.70–0.74** — matching the
analytic recall — **while KPI→KPI FP stays at ~0.010** (block_perm-grade, an order of magnitude below
analytic's). `B=299` is the clean choice: its floor 0.0033 sits just under BH's 0.00357 leading
threshold, so the strongest true edge becomes selectable on its own instead of needing a lucky tie.

**B3 — why the looser null is not a safe fix (analytic FP vs n, Z_full):**

| condition | raw KPI→KPI FP | BH KPI→KPI FP |
|---|---|---|
| analytic · n=1000 | 0.079 | 0.037 |
| analytic · n=4000 | 0.171 | **0.120** |

analytic's KPI→KPI false-positive rate **grows with n** (0.037 → 0.120 as n: 1000 → 4000; the real-E2
run at n=4000 gave 30/360 = 0.083, same regime) — it re-opens the over-selection `block_perm` was
frozen to close. block_perm keeps FP ~0.01 at every permutation count.

**Part B verdict:** a smaller conditioning set does **not** change the picture (power is already there at
any Z). The fix that gives **high true-edge power AND low KPI→KPI FP is more permutations**
(`block_perm_reps ≥ 299`): recall ~0.7 at FP ~0.01, keeping the exact-nominal conditional permutation
null. RCoT is fixable — but via p-value **resolution**, not via reduced Z.

---

## Overall diagnosis

- **(iii) Fundamental to E2's noiselessness — NO.** The true edges are plainly identifiable: the
  conditional test rejects them raw at ~80% (128/160 on real E2; raw power ~0.77 on the test-bed).
  Noiselessness does not hide the edges. **No benchmark-noise change is needed for this.**
- **(ii) Reduced conditioning — NO (not the lever).** `Z_full`, `Z_params_only`, `Z_none`, and
  `Z_coparents` all give the same power; reduced Z does not restore block_perm's BH power. The
  "condition on all 13 other candidates explains away the true parent" hypothesis is **falsified** —
  in this deterministic system the candidate parent is not in Z, so its target-variation survives
  residualization.
- **(i) Looser null — PARTIALLY, but not adoptable.** `analytic_hbe` restores recall (0.163 → 0.769
  on E2) because its p-values are continuous, but its KPI→KPI FP is ~5× higher and **grows with n**
  (Part B/B3), reintroducing the over-selection. Stays a sensitivity variant, per discipline.
- **Actual root cause + fix — BH × permutation resolution.** With `block_perm_reps = 99` the minimum
  p-value is pinned at 1/(99+1) = 0.010, which cannot clear the per-target BH leading threshold
  0.05/14 = 0.0036; the strong-but-quantized true-edge p is discarded at selection. **The fix is to
  raise `block_perm_reps` to ≥ 299** (floor 0.0033 < 0.0036): Part B shows this recovers ~0.7 recall at
  ~0.01 KPI→KPI FP, keeping the exact-nominal conditional permutation null and its FP control intact.
  This is a clean, well-motivated **v2 worth a fresh pre-registration** (it changes only the permutation
  count, not the test or the conditioning). Practical cost: ~3× the block_perm compute per seed
  (≈ 3.2 min → ≈ 10 min/seed at n=4000, single-core).

**One-line answers.**
1. *Part A:* the analytic envelope lifts recall 0.163 → **0.769** at F1 0.857, but KPI→KPI FP rises
   6 → 30 of 360 (1.7% → 8.3%). The low recall is **mostly null-resolution/selection strictness, not
   power** (block_perm raw power is 80%).
2. *Part B:* a smaller conditioning set does **not** restore power (power is already present at any Z);
   the conditioning set is not the lever. The fix that restores power while keeping FP low is **more
   permutations** (`block_perm_reps ≥ 299`): BH power 0.086 → ~0.74 at KPI→KPI FP ~0.01.
3. *Overall:* **fixable — resolution, not noise, not conditioning.** Raise `block_perm_reps` to ≥ 299
   (fresh pre-registration); `analytic_hbe` is not adopted (n-growing FP).

---

## Reproduction

- Part A envelope + validation: `scratchpad/partA_analytic.py`
- Part A raw-vs-BH p-diagnostic: `scratchpad/partA_pdiag.py`
- Part B conditioning × null + resolution sweep: `scratchpad/partB_conditioning.py`

Run with `PYTHONPATH=<repo> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/Scripts/python.exe -u <script>`.
