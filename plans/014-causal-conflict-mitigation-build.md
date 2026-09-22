# Plan 014 — Causal-discovery-driven xApp conflict mitigation: the build

**Status: DECISION COMMITTED 2026-09-22 (user).** Chosen from a 5-way build ideation. Read `docs/ARCHITECTURE.md` first. Supersedes the "what next" question after Thread B closed (`reports/2026-09-19-thread-b-worldmodel-offgate/` §8).

Strategy: **backbone + per-edge structural mechanisms + cheap moonshot probe.** A proper, SOTA solution — not a hack, not half-work relabeled. Timeline flexible (~weeks, past the original 2).

## The ownable SOTA gap (why this wins)
Literature recon (5-agent ideation): closest competitor **Sharma et al. 2025** (arXiv 2510.13031) uses a **SHAP-guessed DAG + single-pair ATE**, descriptive dashboard only — **never tests whether the analysis changes an action**. Others: priority/scheduler (Adamczyk; Cinemre A2C), game-theoretic **QACM** (Wadud), correlational graph-learning (GNN/two-tower/GCN), digital-twin profiling (PACIFISTA). **Nobody does real structure DISCOVERY (not an assumed DAG) → multi-hop do-propagation → ACTION SELECTION validated against a causal ORACLE.** That gap = our E2/E5 frame.

**Honest scope of the claim (load-bearing — do not overclaim):** "given accurate structure discovery, causal do-propagation catches locally-good/globally-harmful shared-knob conflicts that heuristic-DAG / correlational / priority methods structurally cannot" — NOT "deployable on a live RAN today."

## Backbone (the paper spine — do first; agents 3+5)
1. **Wire discovery → do-propagation → decision into ONE live path** (today disconnected — `ARCHITECTURE.md` gap #1). The discovery mask selects which edges propagation may use.
2. **Build E5** (`e5.py`): composed shared-knob scenario (fan-out + chain + confound + noise) — the actual conflict-mitigation stress test.
3. **Named baselines in our harness**, identical seeds: SHAP-DAG+ATE (Sharma), QACM consensus, correlational GNN/two-tower; do-nothing floor; true-SCM oracle ceiling; partial-corr/pdCor ablation.
4. **Metrics:** regret = G_oracle − G_planner (`GATES.md`, unchanged); harmful-edge discovery P/R/F1 (e.g. P0→K5); decision accuracy (fraction of seeds rejecting the harmful action, matching oracle); ATE table in Sharma's units for legibility.
5. **Grounding bridge:** adopt Sharma's RCP/KPI naming (bandwidth/PRB/TxPower/#antennas → throughput/SE/BLER) as a labeled scenario. (ns-O-RAN = future, not gating.)

## Mechanism substrate (turns the true-sim cheat into a real method; agent 3)
Replace true-sim propagation with **per-edge structural mechanism estimates** (a local structural estimator per discovered edge, fit from logged randomized actuation; uncertainty via bootstrap/GP), composed via do-propagation, with **uncertainty-gated abstention** (wide posterior → planner abstains to safe default). Rationale: per-edge local estimation is strictly easier / more identifiable than Thread B's failed **global** surrogate; degrades edge-by-edge; auditable.

## Moonshot probe (cheap go/no-go on the deployability grand prize; agent 1)
**Phase 0, cheap, NO new runs:** fit an **explicit gate+expert** model (bump family matched to E2's Gaussian gate `K5 = -35·exp(-((P0+P7-25)²)/(2·safe_exp(P6)²))`; NLS/EM) truth-free on the **already-collected E2-ID B-bank**; compare off-gate collateral vs CL64/ANISO/honest-tree at low occupancy. **Clears the 0.02 bar where Thread B failed → escalate** to the gated-mechanism learned WM (amortized pretrain + active experimental design). **Doesn't → stay per-edge + abstention, scope honestly.**

## Prerequisites / cleanup
- **Resolve M3-vs-RCoT-v2 canonicalization** (`ARCHITECTURE.md` contradiction #1): pick ONE canonical discovery front-end.
- **Collapse to one pipeline:** quarantine/retire legacy Env I–IV (`plans/012`); port kept baselines to `V2Env` or abandon.
- Commit `docs/ARCHITECTURE.md` + this plan.

## Roadmap / future (next paper, not gating this one)
- ns-O-RAN real grounding (agent 4). Online/interventional causal-bandit deployment (agent 2). Full gated-mechanism learned WM (agent 1) if Phase-0 warrants.

## Parked scratchpad prototypes (real uncited results — do NOT re-derive)
- `scratchpad/e3_multistep_wm/` (E3 multi-step world-model prototype) and `scratchpad/p0k5_downstream/`
  (finding: M3's FP overshoot does not change the E2-spine decision vs oracle — `m3_arm.per_seed_mean_pos` all 0.0)
  hold real truth-free results not cited by any committed report. Reuse when the spine/E3 work needs them; do not
  re-run from scratch. (KEEP: `design_adaptive_m3` = Thread B evidence + B-bank; `p0k5_fp_calibration`/`m3_superset` = M3 productize source; `noise_robustness` = cited by the 2026-09-07 report.)

## Risks / honesty
- Per-edge mechanism identifiability needs randomized actuation (real O-RAN rarely logs it) → abstention gate is the safety valve; scope claims to where randomized/quasi-experimental data exists.
- The "beats SOTA" claim rests on E5 (unbuilt) — build it early.
- Baselines must be implemented faithfully, not straw-manned.
