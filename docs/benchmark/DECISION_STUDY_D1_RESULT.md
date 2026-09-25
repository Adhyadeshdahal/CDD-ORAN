# Decision study D1: RESULT = PASS (2026-09-25)

Pre-declaration: `docs/benchmark/DECISION_STUDY_D1.md`, locked at `22cc352`. It is a **prospective replication
with DEV-informed thresholds**, not a strict confirmatory test. Harness `scripts/d1_decision_study.py`.
Provenance was written before scoring (git head `22cc352`, no uncommitted changes in the listed files). Raw
records: `runs/d1-decision-study/records.jsonl` (gitignored). Fresh corpus seeds 710000–710009 (N = 10).
The run completed without error (about 40,000 s wall time). The harness hardcoded 4 torch threads; that is
recorded here and removed in a later commit.

## Primary gates (N = 10 corpus seeds; small-sample benchmark gates)

| Gate | Result | Rule | Verdict |
|---|---|---|---|
| D1: E2 chain avoids the trap (n = 4000) | learned-MSCR mean loss **0.795**, one-sided 95% seed-bootstrap UB **1.017** (missing-edge 15.15) | mean ≤ 2.0 and UB ≤ 4.0 | PASS |
| D2: graph beats no graph (E2, n = 4000) | learned-MSCR < learned-none in **10/10** seeds; mean paired gap **11.24** | ≥ 8/10 (directional) | PASS |
| D3: E3 look-ahead on the learned model | CEM mean regret **0.0023** (optimal in 96.3% of states) | ≤ 0.01 | PASS |
| D4: myopia costs on the learned model | QACM-v2 mean regret **0.125** (optimal in 0% of states) | ≥ 0.10 | PASS |

## Mandatory reporting

**E2, mean loss / share of states with loss ≥ 1 / worst state:**

| Arm | n = 4000 (primary) | n = 24000 (secondary) |
|---|---|---|
| masked (true equations on the MSCR graph) | 0.000 / 0% / 0.00 | 0.000 / 0% / 0.00 |
| **learned-MSCR** | **0.795 / 20.9% / 18.70** | **0.771 / 17.8% / 13.08** |
| learned-true (ceiling) | 1.145 / 20.3% / 24.35 | 0.816 / 18.8% / 13.08 |
| learned-none | 12.04 / 69.4% / 53.38 | 1.155 / 19.1% / 25.05 |
| missing-edge (true equations minus P0→K5) | 15.15 | 15.15 |

**n = 4000 per-seed means, learned-MSCR:** 1.71, 0.65, 0.91, 0.52, 0.34, 1.11, 1.11, 0.54, 0.42, 0.64.

**E3** (mean regret; share of states optimal):

| Arm | QACM-v2 | CEM | MPPI | MSCR graph exact |
|---|---|---|---|---|
| true-model (diagnostic) | 0.118 | 0.000 | 0.000 | — |
| learned-MSCR | 0.125 (0%) | 0.0023 (96.3%) | 0.0023 (96.3%) | 9/10 seeds |
| learned-true | 0.125 (0%) | 0.0021 (96.6%) | 0.0021 (96.6%) | — |
| learned-none | 0.218 (0%) | 0.019 (65%) | 0.017 (68%) | — |

## What this supports (pre-declared wording)
On these synthetic benchmarks, with this training recipe, the fixed banks and these planner budgets, a world
model fitted on MSCR-discovered structure let the planner avoid the E2 trap and plan near-optimally on E3. The
discovered graph also beat an all-input model.

## Honest notes
- **Not uniformly safe.** learned-MSCR still loses ≥ 1 in about 18–21% of E2 states, and its worst state
  (18.7) is worse than the missing-edge average. Good on average is not safe in every state. This is the gap
  a cautious decision-maker (the guard) would have to close.
- **The graph matters most when data is scarce.** At n = 4000, no graph costs 12.0 against 0.8. At
  n = 24000, it costs 1.16 against 0.77. Whether the benefit is causal structure or just fewer inputs is
  tested separately by Stage 0 F-graph (group-L1 comparator).
- The learned-true ceiling is slightly WORSE than learned-MSCR at n = 4000 (1.15 vs 0.80). MSCR's sparser
  graph, which usually drops the weak P6→K5 edge, generalizes better at this sample size.
- QACM-v2 here is the V2 definition (`greedy_fm`), not the legacy `qacm.py`. CEM/MPPI use 512 rollouts against
  QACM's one-step sweeps: this is a myopic-vs-look-ahead contrast, not an efficiency comparison. E3's search
  is easy, so CEM and MPPI are not separated.
- E3's KPI→KPI discovery is exploratory (MSCR's FDR evidence is param-only). D3/D4 rest on it; the graph
  matched the truth in 9/10 seeds.
- The banks are fixed and shared across seeds. Only the corpora vary.
