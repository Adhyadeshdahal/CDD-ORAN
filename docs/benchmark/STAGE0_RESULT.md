# Stage 0 falsifiers: RESULT (2026-09-26)

Pre-declaration: `docs/benchmark/STAGE0_FALSIFIERS.md`, locked at `2918ecb`. Harness
`scripts/stage0_falsifiers.py`. Provenance was written before scoring (git head `2918ecb`, no uncommitted
changes in the listed files; hashes in `runs/stage0/provenance.json`). Raw records `runs/stage0/records.jsonl`
and scores `runs/stage0/score.json` (gitignored). The run completed in one pass without error. The harness
hardcoded 4 torch threads; that is recorded here and removed in a later commit.

These are **screening falsifiers**. A check that does not fire does not validate MSCR; it only fails to
weaken the named claim.

## Summary

| Check | Claim at risk | Verdict |
|---|---|---|
| F-detect, E2 | MSCR's detection novelty on E2 | **Not falsified** |
| F-detect, E5 (noiseless CORE) | MSCR's detection novelty on E5 | **FALSIFIED by SHAP-GBDT** (τ = 0.005) |
| F-graph, E2 | "the graph's decision value is more than fewer inputs" | **Not falsified** |
| F-dither, δ = 0.10, n = 4000 | MSCR finds the gated edge on bounded dither | **Not falsified** (FDR caveat below) |

## F-detect (test seeds E2 722000–722019, E5 922000–922019; N = 20 each)

**E2 (n = 4000).** MSCR: gated-edge recall **1.00**, family-FDR UB **0.024**.
- SHAP-GBDT: no FDR-valid operating point on DEV. On test it finds the edge at every τ (recall 1.00), but
  its family FDR is 0.10 or higher everywhere (0.63 at τ = 0.005, 0.104 at τ ≥ 0.05).
- Pooled |corr|: no FDR-valid operating point (test FDR 0.11–0.66; recall falls to 0.65 at τ = 0.20).
- RCoT-v2: recall **0.10** (FDR UB 0.025), far below MSCR.
- **Verdict:** not falsified. On E2, MSCR is the only detector here that finds the gated edge with family FDR
  under 0.05.

**E5 CORE, noiseless (n = 24000).** MSCR: recall **1.00**, family-FDR UB **0.000**.
- SHAP-GBDT: DEV selected τ = 0.005. On test: recall **1.00**, paired LB95 (SHAP − MSCR) = 0.00, FDR UB
  **0.000**. The falsifier fires. SHAP also finds it at τ = 0.01 and 0.02, but recall is 0 at τ ≥ 0.05.
- Pooled |corr|: no FDR-valid operating point (recall 1.00, test FDR 0.19–0.49).
- **Verdict: FALSIFIED by SHAP.** As pre-declared, MSCR's detection novelty is **dropped for E5** and
  reported as "matched by SHAP-GBDT at a low threshold". This does not by itself stop the decision claims.
- **Scope:** this E5 corpus is noiseless, although the agreed E5 design has noise on (see
  `E5_STRUCTURE_RESULT.md`). A plausible reason SHAP is clean at τ = 0.005 is that irrelevant knobs get
  essentially zero importance without noise. That is a hypothesis only. The noisy comparison is
  pre-declared separately (`E5_NOISE_DETECT.md`) and does not change this verdict.

## F-graph (E2, n = 4000, test seeds 723000–723009, N = 10, paired)

Mean decision loss per arm (share of states with loss ≥ 1):

| Arm | Mean loss | Share ≥ 1 |
|---|---|---|
| **mscr** (MSCR graph) | **0.544** | 18.8% |
| l1 (all inputs, group-L1, λ by DEV prediction error) | 5.750 | 48.4% |
| shap (τ = 0.10, descriptive) | 11.54 | 69.1% |
| all (no penalty, descriptive) | 11.66 | 68.1% |
| true parents (ceiling, descriptive) | 0.962 | 18.8% |

Per-seed values:
- mscr: 0.59, 0.66, 0.45, 0.56, 0.60, 0.46, 0.27, 0.36, 0.66, 0.83;
- l1: 4.34, 2.43, 4.13, 12.63, 7.68, 4.13, 6.52, 4.10, 3.94, 7.59.

Selected λ per KPI (DEV E2 0–4, validation MSE): K0–K4 = 1e-3, K5 = 3e-3.

- **Paired d = loss(l1) − loss(mscr):** mean **5.21**, one-sided 95% UB **6.77**.
- **Margin:** max(0.20, 0.25 × 0.544 = 0.136) = **0.20**.
- The UB exceeds the margin, so the bound does not rule out L1 being meaningfully worse. **Not falsified.** In
  fact, L1 was worse than MSCR on 10/10 seeds.
- **What this does and does not show:** group-L1 with λ chosen by prediction error, as pre-declared, did not
  prune well enough to decide on. It does not rule out other pruning schemes, or a λ chosen by decision loss
  (that would use the decision truth and was excluded by design).
- As in D1, the true-parent ceiling is worse than the MSCR graph at n = 4000 (0.96 vs 0.54). MSCR's sparser
  graph usually drops the weak P6→K5 edge.

## F-dither (test seeds 724000–724019, N = 20 per δ)

| δ, n | Corpora visiting gate (≥ 200 rows) | Median gate rows | P0→K5 recall (all / visited) | Family FDR (descriptive) | Clipped |
|---|---|---|---|---|---|
| 0.05, 4000 | 85% | 340 | 1.00 / 1.00 | 0.67 | 0 |
| **0.10, 4000 (primary)** | **95%** | 357 | **1.00 / 1.00** (CI90 [1, 1]) | 0.67 | 0 |
| 0.20, 4000 | 100% | 290 | 1.00 / 1.00 | 0.54 | 0 |
| 0.10, 12000 | 100% | 1019 | 1.00 / 1.00 | 0.67 | 0 |

- **Method:** recall among gate-visiting corpora is 1.00, far above the 0.80 screen. **Not falsified** under
  the pre-registered rule. **Correction (2026-09-26, rules and numbers unchanged):** this pass is
  uninformative. At δ = 0.05 and 0.10 (both n), MSCR v2 declared **all 48 knob→KPI pairs in all 20
  corpora**: FDR 0.67 is exactly the complete graph's FDR. At δ = 0.20 the graph is complete except K5. So
  "recall 1.00" means MSCR declares everything on this design, not that it detects the gated edge. The
  cause is that the setpoint blocks confound every knob with every KPI, and v2's null ignores blocks
  (diagnosis: `scratchpad/block_mscr/DESIGN.md`).
- **Data regime:** 95% of corpora visit the gate at the primary level, far above 25%. So the bounded regime
  does not rarely visit the gate.
- **FDR caveat (important):** family FDR on dither data is **0.54–0.67**. As pre-declared, this is
  descriptive only, because MSCR's permutation null ignores the block and time structure of setpoint +
  dither data. In practice, MSCR on bounded dither declares many false edges. **Any deployment claim on
  such data needs a block-aware MSCR**, which is a new method and still to be built and pre-registered.

## Consequences (as pre-declared)
- E2: MSCR's detection novelty stands. On E5, it is dropped in favour of "matched by SHAP-GBDT at a low
  threshold" (noiseless corpus).
- The claim that the graph adds decision value beyond fewer inputs survives this screen, against a group-L1
  comparator only.
- On bounded dither, MSCR finds the harmful edge, but its false-alarm control does not carry over. Next up:
  a block-aware MSCR.
