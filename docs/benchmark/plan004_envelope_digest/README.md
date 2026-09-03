# E1 multi-seed envelope — committed digest (Plan 004)

Verifiable digest of the 10-replicate frozen-matrix sweep. The bulk run outputs
(`rows.npz`, `model.pt`, per-stage logs under `runs/e1slice-v2-envelope/`) stay gitignored; this
directory holds the small artifacts that let the envelope be checked and reproduced from the repo:

- `summary.json` — aggregated envelope stats (mean/median/std/min/max + deterministic bootstrap 95%
  CI), every replicate's child metric hashes, paired MSE differences, settings, and provenance.
- `run_provenance.json` — environment (Python / NumPy / Torch, `uv.lock` sha256, code SHA, dirty flag).
- `replicates.jsonl` — one record per replicate (seeds, per-stage artifact hashes, status).

## Provenance
- Code SHA: `5152c6d` (`feat/v2`, hardened runner after the post-merge review remediation), `git_dirty=false`.
- Regenerate (deterministic — reproduces these hashes bit-for-bit):
  `uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope`
- Frozen matrix: replicates 0..9 with `env_seed=r`, `weight_seed=r`, `split_seed=0`; dataset 48 ep /
  16 steps / warmup 2 / noise 0; test_fraction 0.25; model hidden (16,) / lr 1e-2 / 300 epochs / batch 64.

## Headline (source: `summary.json`; 10 replicates; no significance claim — envelope only)

Graph recovery (mean; std in parentheses):
| block | precision | recall | F1 |
|-------|-----------|--------|----|
| overall | 1.000 (0) | 0.667 (0) | 0.800 (0) |
| NCP→KPI | 1.000 (0) | 1.000 (0) | 1.000 (0) |
| KPI→KPI | 0.000 (0) | 0.000 (0) | 0.000 (0) |

Per-arm one-step prediction MSE (mean / min / max):
| arm | mean | min | max |
|-----|------|-----|-----|
| oracle | 2.702e-07 | 2.818e-09 | 1.075e-06 |
| dense | 1.428e-05 | 1.111e-06 | 6.144e-05 |
| discovered | 9.902e-03 | 8.578e-03 | 1.159e-02 |

Ordering oracle < dense < discovered holds in all 10 replicates; `failed_replicates` is empty.

## Interpretation
Zero variance on all nine graph metrics: the KPI→KPI recall floor is a deterministic property of this
frozen label-free method, not a seed artifact. This **reproduces** the single-seed 2026-09-02 result
(replicate-00, bit-identical hashes) and the earlier uncommitted 2026-09-03 envelope — now backed by
the checkout. It does **not** clear the E1 recovery gate (see `../E1_DISCOVERY_RESULT.md`, Gate status):
the two KPI→KPI edges remain unrecovered, so downstream E2–E5 stay uninterpretable until a method
recovers them.
