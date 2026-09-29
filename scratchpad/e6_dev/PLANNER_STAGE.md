# E6 planner/action stage: design draft (2026-09-27, DEV; revise after the headroom grid)

## Problem size
282 knobs (186 CIO pairs, 24 hys, 24 TTT, 24 ll_ratio, 21 macro carriers, 3 pico sleep), 24 cells. KPM per cell:
fast 1 s (PRB util total/slice, idle reserved PRBs, active UEs, LL delay p95, carriers), thp 5 s (slice throughput,
eMBB p5), mob 30 s (HO/too-late/too-early/ping-pong/RLF per cell pair), energy 60 s.
Conflicts are cross-knob and implicit (ES carrier ↔ TS CIO ↔ MRO; SLICE reservation ↔ TS/ES util readings), so
priority/knob-lock arbiters equal no arbitration (seed 11).

## Pipeline (ours)
1. **Logs:** run noarb (and, if allowed, bounded dither) episodes on DEV seeds → per-cell, per-second panel of applied
   knobs (own + neighbour aggregates) and KPM KPIs.
2. **Discovery (MSCR):** cell-exchangeable template. Pool cells; candidate parents of cell-c KPI k = own-cell knobs,
   in-neighbour CIO aggregates, neighbour carriers/util, lagged own KPIs. Output = template graph + per-edge p-values.
   Discovery answers which xApp actions touch which other xApps' KPIs, i.e. the implicit conflict map.
3. **World model:** per-KPI learned model on discovered parents, trained on within-state DIFFERENCES / local responses
   (D1 lesson: level-accurate passive models rank candidates worse than accept-all). Must report uncertainty.
4. **Arbiter/planner:** every decision, candidates = accept-all, per-xApp veto of the conflicting requests, and a
   bounded MODIFY (partial step) class + a few own writes (D0 lesson: joint bounded nudges ≫ veto-only).
   Choose the candidate only if the model's predicted gain beats accept-all by more than its uncertainty
   (confidence-gated deviation), else accept-all. Guardrails (energy, goal retention) enter as constraints.
5. **Evaluate** on E6_METRIC.md vs noarb/freeze/priority/lock/best-static-subset and the published-method wrappers.

## Decisions pending on headroom results
- If the AR-mask oracle already shows large headroom under guardrails → start with veto+modify arbiter.
- If AR headroom is small (as in D0 on E2-CL) → build a MODIFY/own-write oracle before the learned arbiter.
- Whether the graph earns its keep: ablate MSCR graph vs all-inputs vs SHAP-parents world models (same as D1/K2).
