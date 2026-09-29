# Forward plan after the E6-P screens (agreed with user 2026-09-29)

**Main goal (never drift from it):** causal learning for O-RAN xApp conflict mitigation, i.e. a WG3 arbiter that
uses a discovered cause-effect map to beat QACM, PACIFISTA, xTRUCE and the other published methods on SLA at matched
energy. All environment and gate work exists only to give that arbiter a fair test world.

## Where we stand
- E6 (handover): dead. The causal effect model ranked at ρ≈0; there was nothing to win.
- E6-P v1: DEAD. Fresh-seed v2 with a guardrail-constrained oracle (O_tape):
  - Only **P3 surge-L40** (ES × PowerES × SliceGuarantee under a surge) shows a real loss (Λ 0.37, LB90 32).
  - There the constrained oracle recovers **R 1.35**, CI [0.95, 2.03], and every guardrail improves.
  - Its redrawn advantage is positive on 8/8 seeds.
  - Per-decision ρ_sign is **0.676 < 0.70**.
  - Stage 3: the strongest fixed rule (per-region hindsight static) reaches R 0.76. The oracle headroom over it is
    0.595 ≥ 0.10, so c3 PASSES.
  - **Final v2 verdict for P3 surge-L40: NOISE STOP.** c1, c2 and c3 pass; only c4 fails (0.676 vs 0.70).
    P1 is DEAD (no loss on fresh seeds).
  - Best deployable static rule: R 0.26 (the hindsight static R 0.76 above is privileged, not deployable). QACM and priority do nothing, because the conflict is indirect: ES sleep → neighbour
    load → protected-slice violations.
- The lesson: the edge is real on average but noisy per decision. So act on averaged evidence (regime or window
  policies), not on single decisions.

## Next steps (in order)
1. **MSCR discovery on P3 surge-L40.**
   - Collect randomized logs (the committed L0 infrastructure: units_p / collect_p).
   - Run MSCR, with the CRT wrapper `cdd_oran/decision/crt.py` for dependent data, to recover the chain:
     ES sleep on c → load on neighbour(c) → protected violations on neighbour(c), and PowerES → nothing material.
   - Score it against the simulator's ground truth (candidate-cell relation `ESPico.cand`, and the capacity
     mechanics).
   - Compare to SHAP→DAG, GNN and PACIFISTA-style profiling on the same logs.
2. **Causal world model plus a regime/window arbiter.**
   - The model uses only the causes MSCR found, plus known physics (ARBITER_DESIGN.md §2, SWM).
   - The arbiter chooses per-regime policies: accept / soften / delay ES requests near protected users, under the
     energy target and guardrails. It is judged over windows, not single decisions.
3. **The decisive ablation.** Run the same arbiter with the MSCR map, with no map (all features), and with a
   SHAP-derived map, against QACM, the best static rule and hindsight static. The MSCR arm must win for the causal
   claim.
4. **Publish.** Discuss with the user. Options:
   - a benchmark/negative-results paper (E6, E6-P, Gate A v2);
   - a positive paper if steps 1-3 work;
   - reconciling the Discover journal draft.

## Parked or awaiting user
- xTRUCE XTS-v2: evaluate M4 on h0 strata only (screens X1/X2), < 5 CPU-h.
- The per-decision learned arbiter (L0 infra committed at c30f893): parked. Reuse its collection code for step 1.
- Confirmation seeds 150200-150399 and 155200-155399 stay untouched until a preregistered confirmatory run.

## Compute notes
- Kaggle: 5 sessions × 4 CPU, unattended up to 11 h. It is the default for long or unattended runs.
- Colab free TPU VM: capped at 4 CPUs, ~2.9× faster per core. The session dies without a keep-alive; keep-alive and
  resume support is being added to colab_run.py.
- Never pool results with local Windows (libm differs). Kaggle and Colab can be pooled.
