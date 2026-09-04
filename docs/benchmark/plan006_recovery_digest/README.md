# Plan 006 — E1 v2 discovery RECOVERY envelope (committed digest)

This is the small, committed digest of the **10-seed v2 recovery envelope**, regenerated through a
committed, tested driver (`scripts/e1_slice_recovery_v2_sweep.py`) rather than an ad-hoc harness.
It closes the provenance gap the adversarial review flagged in phase 2. Heavy artifacts
(`rows.npz`, per-seed run dirs) stay gitignored under `runs/` and are byte-regenerable by the
command below.

**Headline:** v1's label-free discovery missed **both** KPI→KPI edges (KPI→KPI recall `0.000`,
overall recall `0.667` — see `../E1_DISCOVERY_RESULT.md`). The v2 per-target partial-correlation
method recovers **all 6** E1 edges, including both KPI→KPI edges, with **KPI→KPI recall `1.000`
across all 10 seeds**, zero false positives, no seed dropped.

## Method under test

Frozen contract: `../E1_DISCOVERY_PROTOCOL_V2.md`, **`protocol_commit = c66b81d3244a0395266d4410640a0065a0816102`**.
Edge score = magnitude of the partial correlation of each input with each target, controlling for
all other candidate inputs (§3); per-target `largest_gap` threshold with `FROZEN_FLOOR = 0.0` (§4);
recovery scored post-persistence vs `E1V2Env().true_adj_matrix()` (§7). No MLP training is involved
in this recovery claim.

## Exact regeneration command

```bash
# from the repo root (worktree agent/006-e1-kpi-kpi-discovery)
uv run python -m scripts.e1_slice_recovery_v2_sweep --out runs/e1slice-v2-recovery
# re-aggregate only (determinism check; byte-identical summary.json):
uv run python -m scripts.e1_slice_recovery_v2_sweep --out runs/e1slice-v2-recovery --aggregate-only
```

Frozen matrix (identical seed/dataset/split policy to Plan 004): replicates `r = 0..9`,
`env_seed = weight_seed = r`, `split_seed = 0`; dataset 48 episodes × 16 steps, warmup 2, noise 0;
split test_fraction 0.25. Per replicate: `generate → split → discover-v2 → recover-v2`.

## Per-seed recovery (all seeds identical: perfect)

| seed | dataset_hash | split_hash | discovery_v2 hash | recovery_v2 hash | overall P/R/F1 | NCP→KPI R | KPI→KPI R | missed |
|-----:|--------------|------------|-------------------|------------------|----------------|-----------|-----------|:------:|
| 0 | cfce4c10c7a7 | ec6658d0c7a4 | 6c2e0adfa950 | 508bc7956f69 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 1 | 254126c60cd5 | e244f626259a | 83ada25464b5 | 1f644db44d89 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 2 | cc60cd9ca292 | 0da14f93df61 | 0cc536052d91 | ead5308d65cb | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 3 | 51667a3d9a6e | 4c7d295a26a8 | e618e2016a37 | 5ad607845a1f | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 4 | 89bcdbcdb211 | 4a8575a7e25f | c3196dda1507 | 90e639cee212 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 5 | 236cec267482 | 079fd288b312 | a4f995271e7c | 7ccdfcda2cd6 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 6 | 8b93271655b0 | aa3be8e6e736 | 5e790aec96ed | f69999999a61 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 7 | 0e4719fce662 | c0bad2d906c6 | 235fd06f01cc | 6c563f062a8d | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 8 | 7f023ff2c511 | c0070ccd5adc | da6342342b4e | a7e316a4bdad | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |
| 9 | 38e89c31053c | 2f00e39b0fc8 | ed33a0f34453 | 00e3e5c24f11 | 1.00/1.00/1.00 | 1.000 | 1.000 | 0 |

Recall envelope (all 10 seeds): overall / NCP→KPI / KPI→KPI mean = min = max = **1.000**.
Seed 0 equals the frozen Plan 003 config; its `split_hash`/`discovery_v2`/`recovery_v2` hashes match
the standalone `discover-v2`/`recover-v2` run on `runs/e1slice/plan003_frozen`.

## Stability-selection frequency (§8: reported diagnostic only, B=10 — never the selector)

Per-edge selection frequency across the 10 seeds (child row, parent column):

|      | P0 | P1 | P2 | P3 | K0 | K1 | K2 | K3 |
|------|----|----|----|----|----|----|----|----|
| K0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| K1 | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 |
| K2 | 0 | 0 | 1.00 | 0 | **1.00** | 0 | 0 | 0 |
| K3 | 0 | 0 | 0 | 1.00 | 0 | **1.00** | 0 | 0 |

Exactly the 6 true edges selected at frequency 1.0 (both KPI→KPI in bold); every non-edge 0.0; no
false positives anywhere in the envelope.

## Provenance and determinism

`run_provenance.json` records the launch git SHA, dirtiness, `uv.lock` digest, and
numpy/torch/python versions; `summary.json` embeds it, so re-aggregation is **byte-identical**
(verified: original run + two `--aggregate-only` passes produce identical `summary.json`).

- `dataset_hash` / `scm_hash` / `split_hash` are git-independent — they match the phase-2 ad-hoc run
  and (seed 0) the frozen 003 artifacts exactly.
- The `discovery_v2` / `recovery_v2` `content_hash`es embed the producing checkout's git provenance
  (per the v1/003 convention: `git_sha` + `git_dirty` are hashed fields). This run was produced at
  `git_sha b5d2684` with `git_dirty = true` (the driver itself is uncommitted pending orchestrator
  commit). After the orchestrator commits, a re-run reproduces **identical masks, recovery, and
  stability**, but different `discovery_v2`/`recovery_v2` content_hashes (git provenance differs).
  The mask/scores/threshold — and therefore every recovery number — do not depend on git state.

## Files

- `summary.json` — canonical aggregate (per-seed recovery, recall envelopes, stability, provenance).
- `replicates.jsonl` — one line per replicate (recovery blocks, missed edges, mask, child hashes).
- `run_provenance.json` — launch provenance (git, versions, uv.lock, frozen settings).
