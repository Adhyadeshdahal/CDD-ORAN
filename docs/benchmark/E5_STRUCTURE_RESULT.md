# E5 structure gate: confirmatory RESULT = PASS (2026-09-24)

Contract: `docs/benchmark/GATE_CONTRACT_E5.md`, frozen at `7ab3899`. Harness `scripts/e5_structure_gate.py
confirm`; provenance was written before scoring (git head `7ab3899`, no uncommitted changes in the
contract, harness, arms, MSCR or env files). Raw output: `runs/e5-structure-gate/confirm.json`
(gitignored). Seeds 900000–900099 (N = 100), disjoint from all previously used E5 seeds (0–5999).
Corpus: CORE layer, randomized, noiseless, n = 24,000 per seed.

## Gates

| Gate | Result | PASS rule | Verdict |
|---|---|---|---|
| S1 MSCR declares P0→K_harm | 100/100 | ≥ 95/100 | PASS |
| S2 MSCR family-level FDR (400 seed×KPI families) | 0.0223, seed-cluster one-sided 95% UB 0.0323 | UB ≤ 0.05 | PASS |
| S3 MSCR total param-edge recall | 1.00 (600/600) | ≥ 0.90 | PASS |
| S4 SHAP-GBDT proxy misses P0→K_harm (τ = 0.10) | 100/100 | ≥ 95/100 | PASS |
| S5 Two-tower reconstruction misses P0→K_harm (τ = 0.10) | 100/100 | ≥ 95/100 | PASS |

## Reported (not gated)

- MSCR pooled FDP 0.0307 (95% CI 0.0164–0.0461); false positives 0.19 per seed.
- SHAP importance ratio of P0 for K_harm: mean 0.0246, max 0.0258 (cutoff 0.10).
- Per-edge recall over 100 seeds:

| Edge | MSCR | SHAP-GBDT proxy | Two-tower | Pooled \|corr\| |
|---|---|---|---|---|
| P0→K_ben | 1.00 | 1.00 | 1.00 | 1.00 |
| **P0→K_harm (harmful, gated)** | **1.00** | **0.00** | **0.00** | 0.99 |
| G1→K_harm | 1.00 | 0.00 | 0.00 | 1.00 |
| G2→K_harm | 1.00 | 1.00 | 1.00 | 0.20 |
| P0→K_mid | 1.00 | 1.00 | 1.00 | 1.00 |
| P3→K_dist | 1.00 | 1.00 | 1.00 | 0.87 |

- The SHAP and two-tower proxies also miss G1→K_harm (a small linear term next to the dominant
  base) in every seed. Their blind spot is weak parents beside a dominant one generally, not only the
  gated edge.
- Pooled |corr| recovers the harmful edge (99/100), as the contract anticipated: correlation is not
  what fails in CORE. It misses the symmetric-bump parent G2→K_harm in 80% of seeds.
- Calibration diagnostic on 20 fresh seeds (D1 pattern): seed-cluster UB95 / α = 1.11–1.16 at every
  gated α (limit 1.5).

## What this supports (contract wording rule)

On the randomized, noiseless E5 CORE corpus (n = 24,000), MSCR recovered the amplitude-subdominant,
co-parent-gated harmful edge in 100/100 fresh seeds and the full parameter graph, with empirical
evidence consistent with family-level FDR control (0.022, upper bound 0.032). The fixed SHAP-GBDT and
two-tower reconstruction proxies at τ = 0.10 missed that edge in 100/100.

Not supported: any decision-regret or decision-critical claim for E5 (deferred to the uncertainty-aware
planner study); threshold-robust or general SHAP/GNN failure; RCoT/pdCor behaviour on E5 (not run);
KPI→KPI discovery; noisy, confounded or observational data. Structure recovery is not calibrated effect
magnitude (see the decision-layer DEV record in the contract).
