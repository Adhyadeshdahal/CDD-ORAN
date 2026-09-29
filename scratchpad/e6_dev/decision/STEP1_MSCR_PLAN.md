# Step 1: MSCR causal discovery on E6-P P3 surge-L40 (plan, 2026-09-29, not frozen)

This condenses the planning agent's report; its file:line references were checked at 662a8e7. Goal: does MSCR recover
the true cause→effect structure of the indirect conflict from randomized logs, and does it beat the baselines?

## Premise corrections
- **PowerES is not minor.** Removing ES gives R 0.56; removing PowerES gives R 0.36. Do not preregister any null
  edge; the knockouts decide.
- **Macro carrier-off does not move load.** It only changes capacity (sim.py:254-256) and energy (:330). Load moves
  only through:
  - **pico sleep**, via `sleep_handover` to each UE's best remaining cell (ric.py:110-115, sim.py:431-446);
  - **ptx**, via the gain column, which shifts the A3 border and interference (sim.py:145,225,241).
- **MSCR history.** MSCR has never been run on E6-P. On E6, row-permutation MSCR declared every knob (spurious
  regression; decision_stack/REPORT_H.md), so the CRT is needed. `crt.py` needs a unit-level assignment model
  (crt_units.py below).
- **The cloud bundle excludes `cdd_oran/discovery`,** so collection runs in the cloud and MSCR analysis runs locally.

## Data
- **Config:** `make_cfg("P3", 3, seed, lf=1.75547)`, 720 s episodes.
- **Policy:** `RandomizedUnitPolicy` with HIGH_NO_RB tables: accept 0.5 / half 0.2 / reject 0.3 for every xApp.
  `open_rule="feasible"`.
- **Per-episode JSONL:**
  - units (c, x, knob, t0, mode, p, ctx);
  - a privileged per-second per-cell `lab_series` (pv, v, e, rlf, prb_used, prb_cap, UE-s, protected UE-s);
  - the obs-only E6-P panel (`build_panel_p`);
  - `gt_static` (`ESPico.cand`, cell_site, neighbours), which discovery must never read (enforced by a test).
- **Episodes:**

  | Use | Episodes |
  |---|---|
  | DEV | 20 |
  | EVAL | 60 (3 folds × 20) |
  | PLACEBO (modes drawn and logged, but accept applied: a sharp null) | 20 |
  | GT knockout | 20 |
  | PACIFISTA profiles | 3 × 8 |

- **Hypotheses:** knob families {carrier, sleep, ptx, prot_min} × relations {own, nbr = exposure set minus own,
  far = negative control} × KPIs {pv, v, e, rlf, load} = 60.
- **Unit target:** the KPI over [t0, t0+90 s] minus the pre-window, summed over the relation's cells.

## Ground truth
The CRN knockout on the GT seeds is primary; the physics table is only a sanity check.
- Accept vs reject, k ∈ {1, 2, 3}, H = 90 s.
- **TRUE(sign)** if the episode-cluster bootstrap 95 % CI excludes 0 and |mean| ≥ 5 % of the KPI's largest |mean|
  (pv floor 0.5).
- **NULL** if the CI lies within ±δ.
- Otherwise **INDET**, excluded from scoring and counted.
- Expected: sleep → nbr load/pv/v (+), with cand dominant. Carrier-off → own only. ptx → own e (−) and nbr load (+).
  prot_min → own pv (−). far → all NULL.

## Methods
- **Primary: MSCR-CRT** (crt_units.py): redraw a family's modes from logged π0 per unit, `MSCRStat` + `signed_stat`,
  B = 999, BY at q = 0.05 over 60 hypotheses, lag-1 carryover audit.
- **Naive comparator:** MSCR-rowperm (`discover_template` on the panel).
- **Baselines**, same unit table, τ tuned on DEV only:
  - SHAP-GBDT;
  - |corr|;
  - Granger;
  - two-tower (parametrised sizes);
  - PACIFISTA-style ECDF INT distance;
  - QACM KPIPredictor (own cell only, so indirect recall is 0 by construction).
- **Metrics** over TRUE ∪ NULL:
  - precision, recall, F1 and sign accuracy, overall and **indirect (nbr)**;
  - the count of far false positives;
  - top-1 receiving cell vs `cand`.

## Seeds, criteria, verdicts
- **Seeds:** register **180000-183999** as `e6p_discovery_episodes`. P3 s3 uses 183300 + j:
  - j 0-19 DEV, 20-79 EVAL, 80-99 GT;
  - PLACEBO and profiles on 183400-183443.
- **Tags:** 6612-6615 (pending) plus 6616 (CRT) and 6617 (bootstrap).
- **Freeze** docs/benchmark/E6P_DISCOVERY_PROTOCOL.md before any EVAL or GT run. DEV may run before the freeze.
- **Criteria:**

  | Criterion | Rule |
  |---|---|
  | K0 validity (PLACEBO) | Rejection rate ≤ 0.08 at α 0.05, ≤ 1 BY declaration out of 60. Otherwise INVALID. |
  | K1 support | ≥ 60 sleep units with ≥ 15 rejects. Else one extension to 120 EVAL episodes, then UNDERPOWERED. |
  | G premise | GT shows sleep → nbr pv TRUE(+). Else NO-CHAIN (stop). |
  | P1 | Indirect recall ≥ 0.67 at indirect precision ≥ 0.80; overall F1 ≥ 0.60; sign accuracy ≥ 0.90; ≤ 1 far declaration. |
  | P2 | MSCR indirect F1 ≥ the best DEV-tuned baseline, pooled and in ≥ 2 of 3 folds. |

- **Verdicts:**
  - PASS = P1 and P2.
  - PARTIAL = P1 only; step 3 then must include the SHAP-map arm.
  - KILL = P1 fails; step 2 then uses physics only, as a declared privilege.

## Build
- **Modify:**
  - collect_p.py: tap fields, `series()`, PlaceboPolicy, HIGH_NO_RB;
  - labels_p.py: per-cell vectors;
  - features.py: `build_panel_p`;
  - discovery.py: `kpi_owner` override;
  - SEED_REGISTRY.
- **New:**
  - crt_units.py, gt_p.py, baselines_disc.py, edge_score.py;
  - driver scratchpad/e6_dev/e6p_discovery.py, with stages dev / eval / placebo / gt / prof and wrappers;
  - tests for each.
- **Compute:**
  - collection: ~3 CPU-h on Kaggle;
  - GT: ~25 CPU-h (measure it with 2 smoke GT episodes first);
  - analysis: local, < 1 h.

## Risks
- Units are sequentially dependent. Handled by PLACEBO calibration and the lag-1 audit.
- Sleep is rare. Handled by K1.
- pv is sparse and zero-inflated.
- Load confounds panel methods; randomization fixes this for unit methods.
- Sleep has a compositional own-cell effect.
- Layout changes per seed.
- Estimand mismatch: GT continues with accept-all, the CRT with π0.
- prot_below is not a standard KPM.
