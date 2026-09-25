# Decision study D1: discovery → learned world model → planner (pre-declaration)

**Status:** locked at the commit that adds this file, before any data on seeds 710000–710009. User go given
2026-09-25. Revised per sol review (`scratchpad/sol_review_predeclare_d1.md`: 4 blocking points addressed).

**What kind of study this is.** A **prospective replication with DEV-informed thresholds**, not a strict
confirmatory test. The design, arms and cutoffs were written after DEV probes on other seeds
(`scratchpad/decision_phase/`), so the cutoffs are not independent of those results. What the fresh seeds
test is whether the DEV picture replicates. DEV evidence does **not** establish that D1–D4 will pass; the DEV
fully-learned E2 runs at n=4000 were 2 seeds on the same bank. Every gate can fail, and failures are
recorded as-is.

## Question
With MSCR-discovered structure and per-KPI world models **fitted from data**, does a planner avoid the E2
conflict trap and plan well over E3's 3-step horizon? And does restricting the learned model to the discovered
graph beat giving it every input, under this fixed training recipe?

## Fixed components
- **Discovery:** `cdd_oran.discovery.discover_mscr`, frozen config. E2: params tested (n_params = 8), lagged
  KPIs as conditioners. E3: params AND lagged KPIs tested (n_params = 9); KPI→KPI selection is exploratory
  (MSCR's FDR evidence is param-only), and D3/D4 are conditional on it.
- **Learned world model:** `cdd_oran/benchmark/learned_world_model.py`: per KPI, the mean of 3 MLPs (2×64
  SiLU, standardized inputs/target, Adam lr 3e-3, cosine decay, 1500 full-batch epochs, member seeds 0–2) on the
  KPI's parents under the arm's graph, fitted on the same randomized corpus as discovery. Input layout =
  [lagged params | lagged KPIs]. E2's true KPIs depend on current params only, so E2 tests learned response
  surfaces; E3 has lagged-KPI parents and tests learned transitions.
- **Planners** (`cdd_oran/planners/sequence.py`): E2 (one knob, H=1): the frozen brute-force pick over the
  101-point grid. E3 (H=3): **QACM-v2** (`qacm_v2` = `greedy_fm`: exhaustive one-step conflict-knob choice
  under the locked cost, re-decided each step; the V2 definition, not the legacy `qacm.py` implementation),
  CEM and MPPI (64 samples × 8 iterations, seed 0). **Compute differs by design:** QACM does 3 one-step sweeps
  of 101 values; CEM/MPPI do 512 sequence rollouts. This is a myopic-vs-look-ahead contrast, not an efficiency
  comparison.
- **Scoring:** plans realized on the TRUE env with the locked standardized reward, on the fixed frozen banks
  (E2: the 32 decision-gate positives; E3: the 32-state bank, exact optimum over 101³ sequences). The banks
  are the same for every seed; seeds vary the corpus. E2 loss = true best − true reward of the chosen action;
  E3 regret = (G* − G(plan)) / D3.

## Arms
| Arm | World model | Role |
|---|---|---|
| **learned-MSCR** | fitted models on the MSCR graph | **the method** |
| learned-true | fitted models on the TRUE parents | ceiling for the learned model (required) |
| learned-none | fitted models on all inputs | no-graph comparator |
| masked (E2) | true equations on the MSCR graph | diagnostic: discovery error alone |
| missing-edge (E2) | true equations minus P0→K5 | diagnostic: the trap's cost |
| true-model (E3) | true simulator | diagnostic: planner quality alone |

"No true mechanism" applies to the three learned arms only; the diagnostic arms use true equations by design.

## Data and seeds
- E2 corpora: `generate_rows(E2DatasetConfig(n, seed))`, seeds **710000–710009**; **n = 4000 is primary**,
  n = 24000 secondary. A secondary result cannot rescue a failed primary gate.
- E3 corpora: randomized transitions (params i.i.d. U[0,1], true mechanism rolled forward, `k_t = f(p_{t−1},
  k_{t−1})`), n = 8000, seeds **710000–710009**.
- The harness asserts disjointness from previously used seeds (E2: 0–19, 100000–100003, 500000–500039,
  600000–616383, 800000–800099; E3: 0).

## Gates (small-sample benchmark gates; unit = corpus seed, N = 10)
| # | Claim | PASS |
|---|---|---|
| D1 | E2 chain avoids the trap (n = 4000) | learned-MSCR mean of the 10 per-seed mean losses ≤ 2.0 (missing-edge ≈ 15.1) AND its one-sided 95% percentile-bootstrap upper bound ≤ 4.0 (10,000 resamples of the 10 seed means, RNG seed 0) |
| D2 | Graph beats no graph on E2 (n = 4000) | learned-MSCR < learned-none in ≥ 8/10 paired seeds. **Directional criterion**; the paired differences and their mean are reported, not gated |
| D3 | E3 look-ahead on the learned model | CEM on learned-MSCR: mean regret ≤ 0.01 over 10 seeds × 32 states |
| D4 | Myopia costs on the learned model | QACM-v2 on learned-MSCR: mean regret ≥ 0.10 |

**Mandatory reporting (not gated):** all 10 per-seed means per arm; fraction of states with loss ≥ 1 and the
worst state (E2); max regret and fraction optimal (E3); learned-true and learned-none on both envs; the
true-model planner diagnostic on E3; the masked and missing-edge diagnostics on E2; n = 24000 results; the
MSCR E3 graph exact-match rate.

## Claim wording
If D1–D4 pass: "On these synthetic benchmarks, with this training recipe, fixed banks and planner budgets, a
world model fitted on MSCR-discovered structure let the planner avoid the E2 trap and plan near-optimally on
E3, and the discovered graph beat an all-input model." Not claimed: general causal-discovery value, planner
efficiency, CEM vs MPPI differences (E3's search is easy), noise/observational robustness, or O-RAN deployment.
