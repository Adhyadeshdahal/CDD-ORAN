# E1 learned-discovery result (frozen protocol, single run)

This is the **as-run** result of the frozen protocol in `E1_DISCOVERY_PROTOCOL.md`, executed
once and reported **as-is**. The recovery is **partial** (all NCP->KPI edges recovered, both
KPI->KPI edges missed). Per the frozen discipline this is **not tuned**: it is a valid scientific
outcome, and the KPI->KPI floor is a real property of this randomized, temporally ordered, linear
noiseless E1 control under a label-free OLS + `largest_gap` rule.

## Provenance (the freeze predates the result)

- Protocol commit (frozen BEFORE any recovery): `e5312a696a0ed5f9d590e9ad766bdbfdad53a16c`
  (`docs: freeze E1 discovery protocol`). Recorded as `protocol_commit` in `discovery.json`.
- Implementation commit: `9f6fa5b697835e858bd4e95f2a99f86986ea5798`
  (`e1slice: add learned graph and discovered arm`).
- `dataset_hash` `cfce4c10c7a7…`, `split_hash` `ec6658d0c7a4…`,
  `discovery.json content_hash` `c67fd92ef7be…`, `metrics_hash` `c9ff3ed28f36…`,
  `recovery.json content_hash` `7914ad050826…`.
- Archived artifacts: `docs/benchmark/plan003_frozen_artifacts/{discovery,metrics,recovery,split}.json`
  and `run.log` (heavy `rows.npz`/`model.pt` live in the gitignored `runs/` dir).

## Configuration and command lines

48 episodes, 16 recorded steps, warmup 2, env_seed 0; test_fraction 0.25, split_seed 0
(36 train / 12 test episodes; 576 train rows, 192 test rows); hidden 16, lr 0.01, 300 epochs,
batch 64, weight_seed 0. Run in order:

```
uv run python -m scripts.e1_slice generate --episodes 48 --steps 16 --warmup 2 --seed 0 --out runs/e1slice/plan003_frozen
uv run python -m scripts.e1_slice split    --dataset runs/e1slice/plan003_frozen --test-fraction 0.25 --split-seed 0
uv run python -m scripts.e1_slice discover --dataset runs/e1slice/plan003_frozen
uv run python -m scripts.e1_slice train    --dataset runs/e1slice/plan003_frozen --hidden 16 --lr 0.01 --epochs 300 --batch-size 64 --weight-seed 0
uv run python -m scripts.e1_slice eval     --dataset runs/e1slice/plan003_frozen
uv run python -m scripts.e1_slice verify   --dataset runs/e1slice/plan003_frozen
uv run python -m scripts.e1_slice recover  --dataset runs/e1slice/plan003_frozen
```

The CLI exited 0 through verification (all three arms `PASS`, `max_abs_diff 0.00e+00`), and the
recovery scoring reads E1 truth only after `discovery.json` is persisted and hashed.

## Discovered graph vs E1 true parents

Label-free threshold `largest_gap(scores.ravel(), floor=1e-3)` = **0.6742** (tie rule
`score >= threshold`). Absolute standardized coefficients (edge scores):

| child | P0 | P1 | P2 | P3 | K0 | K1 | K2 | K3 |
|-------|----|----|----|----|----|----|----|----|
| K0 | **1.00** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| K1 | 0 | **1.00** | 0 | 0 | 0 | 0 | 0 | 0 |
| K2 | 0 | 0 | **0.898** | 0 | 0.439 | 0 | 0 | 0 |
| K3 | 0 | 0 | 0 | **0.892** | 0 | 0.456 | 0 | 0 |

- **True E1 parents (6 edges)**: K0<-P0, K1<-P1, K2<-{P2, K0}, K3<-{P3, K1}.
- **Discovered mask (4 edges)**: K0<-P0, K1<-P1, K2<-P2, K3<-P3.
- **Missed (2)**: K2<-K0 and K3<-K1 — their standardized coefficients (0.439, 0.456) fall in the
  gap **below** the selected threshold; the largest gap in the score distribution sits between the
  KPI->KPI band (~0.44) and the NCP->KPI band (~0.9), so `largest_gap` cuts there.
- **False positives**: none.

## Recovery metrics (post-freeze, vs `E1V2Env.true_adj_matrix()`)

| block | precision | recall | F1 | tp | fp | fn |
|-------|-----------|--------|----|----|----|----|
| overall | 1.000 | 0.667 | 0.800 | 4 | 0 | 2 |
| NCP->KPI | 1.000 | 1.000 | 1.000 | 4 | 0 | 0 |
| KPI->KPI | 0.000 | 0.000 | 0.000 | 0 | 0 | 2 |

## Three-arm one-step prediction (held-out test, n=192)

| arm | test MSE | test MAE | train MSE |
|-----|----------|----------|-----------|
| oracle | 4.253e-07 | 1.435e-04 | 3.838e-08 |
| dense | 1.433e-06 | 5.351e-04 | 4.731e-07 |
| discovered | 1.036e-02 | 6.038e-02 | 9.817e-03 |

All three arms share the identical 644-parameter per-output MLP; only the fixed mask differs.
The discovered arm's higher error is the direct consequence of the two missing KPI->KPI parents:
its K2/K3 heads cannot see K0/K1, so they cannot represent `K2 = P2 + 0.5·K0` or
`K3 = P3 + 0.5·K1`.

## Interpretation

The label-free rule recovers **exactly the NCP->KPI (interventionally reachable) structure** and
leaves a **KPI->KPI recall floor**. This is consistent with the split-by-edge-type diagnostic:
the missing mass is entirely KPI->KPI, which random NCP interventions in this control cannot help
resolve. The result is reported as the frozen method produced it; it was not adjusted after E1
truth was inspected.
