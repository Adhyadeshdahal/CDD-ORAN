# Brief: xm-harness (worlds, regimes, runner, scorer)
Read first: scratchpad/xmethod/CONTRACT.md (rules), cdd_oran/xmethod/api.py (interface), docs/ARCHITECTURE.md sec 0-1,
cdd_oran/envs/README.md, docs/benchmark/SEMANTICS.md, docs/benchmark/STAGE0_RESULT.md (F-dither design),
docs/benchmark/GATE_CONTRACT_E4.md.
Deliver (package cdd_oran/xmethod/, tests in tests/test_xmethod_harness.py):
1. worlds/: generate_dataset(world, regime, n, seed) -> (api.Dataset, api.Truth) for E1, E2, E3, E5 x {R1, R2} and
   E4 x {R1, R2, R3, R4} x lambda {0, .5, 1, 1.5}. Reuse the existing SCMs read-only (E2 via e2slice.dataset logic,
   E1 via e1slice, E3/E5 via envs/v2). R2 = Stage 0 F-dither (bounded setpoint + random dither; fill
   Design.random_part / fixed_part). R3 = NEW E4 variant with an OBSERVED confounder and logged propensities (new
   file; the frozen E4 SCM untouched). Every dataset gets the placebo action column P_placebo (CONTRACT sec 2).
   Truth: true edges, signs, exact-null edges, derived from the SCM definitions (not estimated).
2. score.py: score(result, truth) -> precision, recall, F1, sign accuracy, FDP, null-edge FPR, placebo declarations.
3. runner.py: shard (world, regime, n, seed, method) jobs; writes JSONL records (one per job: Result + scores +
   cpu_s + peak RAM); a launcher for Kaggle (scratchpad/e6_dev/kaggle_job.py pattern) and one for Colab or
   Lightning; a merge step. A dummy method (random scores) for end-to-end tests.
4. Sanity tests: E4 R1 association sign at lambda 1.5 is wrong (as the frozen contract states); R2 rows show the
   intended serial dependence; P_placebo independent of everything; byte-identical regeneration per seed.
Do not implement any discovery method. Report in the hand-back: dataset shapes per world, candidate counts, true /
null edge counts, generation CPU-s at n 24000, and every judgement call you made on R2 / R3 parameters (these go
into the protocol for the user's review).
