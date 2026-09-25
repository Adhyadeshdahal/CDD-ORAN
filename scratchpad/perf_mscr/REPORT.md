# MSCR performance and logic audit — branch `perf/mscr-opt` (commit e1d7bcf on feat/v2 2918ecb)

Written by the orchestrator from the perf agent's hand-back (subagents cannot write report files).
Everything the agent executed was small (n ≤ 2000, B ≤ 299, plus one B=2999 identity case at n=300)
because two confirmatory experiments were running; that was a scheduling choice. The code targets the
fastest hardware available at run time. Speed-ups are projections.

**Orchestrator verification (2026-09-25):**
- `equivalence_check.py` re-run: ALL IDENTICAL (17 cases × 7 CPU/torch paths). The CUDA log is taken from
  the agent.
- `mscr_v2_reference.py` is AST-identical, function for function, to committed v2 (`2918ecb`
  `cdd_oran/discovery/mscr.py`).
- `audit_cases.py` re-run: BUG-1 and BUG-2 reproduce exactly.

## 1. Profile (E2, n=2000, B=299, one thread, loaded machine)

| Phase | Time | Share |
|---|---|---|
| Bank | 1.72 s | 90% (~99% at B=2999) |
| argsort (inside the bank) | — | ~70% of the total |
| Observed S* | 0.17 s | 9% at B=299, ~1% at B=2999 |
| p-values and BY | ~0 | ~0 |

- Per element: rng.random ~5 ns, argsort ~24–28 ns, gather ~3 ns, reduceat ~4 ns.
- Work W = targets × cols × B × n:

| Corpus | W | One core, unloaded |
|---|---|---|
| E2 n=4000 | 1.0e9 | ~20 s |
| E2 n=24000 | 6.1e9 | ~120 s |
| E5 n=24000 | 1.15e9 | ~23 s |

- Memory peak per stratum in v2: 2·B·ns·8 bytes, i.e. 192 MB at n=24000.

## 2. Tier A (bit-identical)
Files: `cdd_oran/discovery/mscr.py` and the new `cdd_oran/discovery/mscr_torch.py`.
`discover_mscr(..., n_jobs=None, device=None)`; the private helpers keep their signatures.

- **A1 Blocked bank:** row blocks are contiguous pieces of the same C-order RNG stream.
- **A2 PCG64 jump-ahead:** each task clones the state and calls `advance(first_draw)`. The caller's
  generator is left in v2's end state; generators without jump-ahead fall back to sequential draws.
- **A3** One thread pool over all tasks; sums are still added in stratum order.
- **A4** The strata plan is computed once and shared across targets.
- **A5** The observed S* is threaded.
- **A6 torch/CUDA bank, why it is exact:**
  - it uses the same host uniforms;
  - sorting distinct keys gives the same order as `np.argsort`, and exact-tie rows are re-sorted on the host;
  - numpy's reduceat order (`a[lo] + pairwise_sum(rest)`) is replayed in float64;
  - on device out-of-memory it retries on row halves;
  - auto-selection is gated by a self-check.
- **A7 Runtime resolution:**
  - `n_jobs`: explicit, then `$MSCR_NUM_THREADS`, then `$OMP_NUM_THREADS`, then the CPU count;
  - `device`: explicit, then `$MSCR_DEVICE`, then CUDA if it passes the self-check, else numpy on CPU.

Evidence: `equivalence_check.py` (17 cases incl. ties, constants, dropped strata, B=2999, real E2/E5; 7
paths; outputs + raw bank + RNG end state), `check_tie_fallback.py`, and `tests/test_discovery_mscr.py`
passing.

Serial single-core speed is unchanged. The gains come from memory, cores and the GPU. The exact bits depend
on numpy's argsort and reduceat behaviour (numpy 2.4.2, pinned by `uv.lock`), so provenance must record
the numpy version.

## 3. Projections (NOT measured) and the benchmark plan
- This laptop (i5-1340P): about 4–8x on CPU. A server with P cores: about 0.8×P.
- RTX 2050 (Tier A): about 6–10x, limited by host RNG and PCIe. Device-side RNG (Tier B2) would give about
  20x.
- `bench_mscr.py` compares the v2 reference, CPU at n_jobs 1…N, and CUDA. It records wall time, peak RSS
  and an output hash; a hash mismatch is a FAIL. Output goes to `runs/perf-mscr/bench.json`.

## 4. Tier B (a new version `mscr-v2.1`, kept alongside v2)
- **B1 One bank shared across targets:** each target's p-vector keeps the same law; only the dependence
  between targets changes. About 3x.
- **B2 GPU (Philox) random numbers:** about 20x. The torch version and device class must be recorded.
- **B3 Hardening** (fixes BUG-1/2 and RISK-3/4, NIT-6):
  - centre y per stratum;
  - treat a near-zero total sum of squares as zero;
  - break candidate ties with a seeded random key;
  - compute the observed statistic with the bank's own summation kernel;
  - reject non-finite input.

Each Tier-B change requires re-validation (calibration plus flip rate; fresh confirmatory seeds are the
user's call).

## 5. Tier C (design only)
- **Sequential Monte Carlo (Besag–Clifford):** the p-value is valid, but it saves about nothing here,
  because the bank is shared and each E2 KPI has 2–3 true edges needing the full B.
- **Analytic Beta null:** fails in the ~2e-4 tail needed, and the max over conditioners is dependent.

Not recommended.

## 6. Audit

| id | Label | Finding | Input | Shipped results? |
|---|---|---|---|---|
| 1 | BUG | A constant target yields edges (rounding-noise TSS > 0) | y ≡ 1.1, n=1200: S* = 1.111, all declared | No (min KPI sd ≥ 4.4) |
| 2 | BUG | Tied or constant candidates are binned by row order, so time acts as signal | constant x0 vs drifting y: p = 0.0033, declared | No for E2/E3/E5 (continuous params). **Hypothesis: may drive E1's FDR 0.81** (single-param actuation, so the other params are constant within stretches) |
| 3 | RISK | Cancellation at a large mean | offset 1e6: p 0.82 → 0.037 | No (\|mean\|/sd ≤ 4) |
| 4 | RISK | The `>=` tie convention is decided by rounding | discrete y: bank < obs in 1818/2000 | Practically no |
| 5 | RISK (known) | The joint max-null is approximate | — | Calibrate per regime |
| 6 | NIT | A NaN in y silently gives p = 1 | — | — |
| 7 | NIT | All conditioners degenerate gives s* = −inf silently | — | — |
| 10 | NIT | Bit reproducibility depends on the numpy version | — | record in provenance |

Verified correct: BY step-up, p = (1+G)/(B+1), self-conditioner exclusion, bank bin sizes, strata dropping,
per-target seeding.

## Decisions for the user
- Merge Tier A after the running experiments finish.
- The defaults for device and thread resolution.
- The v2.1 re-validation bar.
- Whether BUG-1/2 are fixed in v2.1 or guarded in callers.
