# E6-P learned WG3 arbiter: design draft (Plan agent, 2026-09-29, NOT frozen)

Build only if Gate A v2 passes for P1 or P3. This is a condensed copy of the design agent's report; the file:line
references were checked by the agent at 2afd081.

## 0. Facts from the code that shape the design
- **Only P1 and P3 matter.** P2 is not screenable.
- **One writer per knob per cell.** ES owns carrier and sleep, PowerES owns ptx, SG owns prot_min. A lock only
  force-rejects requests on its own knob (env.py:231-239), so **lock ≡ reject** for P1 and P3.
- **Rollback does nothing for ES.** RB_WINDOW is 60 s (e6p_screen.py:321,363) but the carrier/sleep dwell is 120 s
  (ric.py:23-24), so the actuator NACKs the rollback. The oracle's `rb` draw on ES is inert, which makes the oracle
  conservative.
- **Energy decisions are few and long-lived.** In e6p-s0b, ES alone makes about 36-45 applied changes per episode;
  SG makes 127-264. This is unlike E6, where one policy was the sum of ~78 tiny effects.
- **Episode cost.** An episode is 720 s and costs 71-80 CPU-s.

## 1. Decision unit
- **Unit = (cell c, xApp x, t0).** The mode holds for T = 60 s.
- **Modes:** {accept, half, reject}, plus a rollback flag for SG and PowerES only.
- **Why 60 s and not 20 s.** ES re-proposes every 10 s, so a 20-s reject only delays a carrier-off.
- **Predeclared switch (G0b).** Fall back to the oracle's region × xApp unit with D = 20 if either:
  - cell-unit contrasts explain less than 50 % of the region contrast; or
  - more than 50 % of the headroom mass sits in unmodelled unit types.

## 2. Model
### Primary: structural world model (SWM)
- **Output.** Per unit and mode, the SWM predicts (ΔPV, ΔE, ΔV) versus accept over 90 s, on the exposure set N(c).
- **Known mechanisms, not learned:**
  - carrier 106→53 PRB;
  - sleep sets capacity to 0 and hands UEs over;
  - EARTH energy;
  - analytic ptx SINR shift.
  So ΔE is near-deterministic.
- **Learned part.** A hurdle-Poisson model of violations from about 10 obs-only mediators (ric.py:204-230):
  - Violations are monotone in capacity and prot_min.
  - Contrasts are paired and EB-shrunk.
- **Where causal discovery enters.**
  - A design-based CRT (crt.py:240) on the randomized modes gives the exposure map. It must recover `ESPico.cand`,
    which serves as a built-in ground truth.
  - MSCR (discovery.py:174) selects the mediator parents, and the edge signs set the monotone constraints.
- **Decision rule.**
  - Pick the mode minimising ΔPV + λΔE.
  - Deviate from accept only if the conformal lower bound is > 0 (gate.py:61) and the context is in support
    (gate.py:41).
  - Energy budget: forgone saving ≤ β(realised + forgone), with β = 0.07.
  - λ is set on calibration seeds only.

### Ablation and other options
- **Ablation: DR contextual bandit.** It uses cross-fitted AIPW on the raw context, with no graph or mechanisms.
  SWM − DR is the causal-structure claim.
- **Imitation of the oracle.** Descriptive only: its labels are tape-noisy, it ignores the budget, and there would be
  only ~600 of them.

### Why SWM should beat the baselines
- **vs QACM** (published.py:345-452):
  - Its detector is reactive: it fires only after a KPI is already violated on the same cell (:368-380).
  - Energy has QoS None, so matched energy is never controlled.
  - It predicts one report ahead on the acting cell only, so it misses pico sleep hurting the candidate macro and
    the 120-s persistence.
  - It is fitted on accept-all logs, so it is confounded.
- **vs PACIFISTA:** its output is a static subset, so it is at most R_static.

### E6 failure modes and counters
| E6 failure | Counter | Fails again if |
|---|---|---|
| ρ≈0 | G0c runs before anything is built | Violations depend on per-UE position or bursts that cell KPM cannot see |
| Overdispersion | Paired contrasts, ~10 features, monotone, shrinkage | The response is non-monotone (a HO helps a protected UE) |
| 82 % out of support | Every mode logged with p ≥ 0.1 | Joint co-site exposures were never logged |
| Oracle reliability 0.56 | 2 CRN reseeds per label, G0a | Split-half correlation < 0.7 |

**Extra risk: SUTVA.** The xApp 'hold' variant skips a cycle after any NACK (xapps.py:54-60).

## 3. Randomized collection
- **Logging policy π0.** Context-free, with two regimes that alternate by parity of j:

  | Regime | ES / PowerES | SG |
  |---|---|---|
  | high-ε | accept 0.5, half 0.2, reject 0.3 | accept 0.5, half 0.15, reject 0.25, accept+rb 0.10 |
  | low-ε | accept 0.8, other modes scaled | accept 0.8, other modes scaled |

- **Draws.** `default_rng([seed, 6612, c, x_idx, t0])`, so they reproduce inside copies. The minimum propensity is
  0.1.
- **Labels.**
  - Sampling rate: 0.3 for ES/PowerES and 0.05 for SG (tag 6613).
  - Each label is the per-mode `env.copy(reseed=k)` for k ∈ {1, 2}, with the mode held for T and the rest on AA,
    over H = 90 s.
  - Label reliability is the correlation between k = 1 and k = 2.
- **Off-policy evaluation.** Unit-level IPS / SNIPS / DR with an episode-cluster bootstrap. It assumes additivity
  across units, which is flagged.
- **Seeds (DEV).** The block 160000-179999 and tags 6612-6615 must be added to SEED_REGISTRY.json first.
  - Collection: `160000 + 1000·pair + 100·stratum + j`, j = 0-99.
    - j 0-59: fit (labels on j 0-19)
    - j 60-79: calibration
    - j 80-99: OPE check
  - On-policy evaluation: `170000 + ...`, j = 0-39.
- **Budget.**
  - About 25 CPU-h for each P1 (pair, stratum) and about 35 CPU-h for each P3 (pair, stratum), with a cap of
    130 CPU-h.
  - Stage L0 (10 episodes plus ~220 labels) costs about 4 CPU-h, and the kill decisions happen there.

## 4. Criteria, fixed before any outcome
**L0: kill unless all of these hold.**
- **G0a:** label reliability ≥ 0.70.
- **G0c:** CV-by-episode sign AUC on nonzero labels ≥ 0.70 (90 % lower bound ≥ 0.60), and contrast R² ≥ 0.05.
- **G0b:** fixes the decision unit.

**Main evaluation.**
- **S1:** R_SWM − R_static,fit ≥ 0.05, with the lower bound > 0 after Holm correction.
- **S2:** R_SWM ≥ R_static + 0.4·(R_or − R_static).
- **S3:** energy-matched, and every guardrail ≤ 1.10 × AA.
- **S4:** on held-out labels:
  - sign AUC ≥ 0.70;
  - SD ratio ≤ 2;
  - calibration slope in [0.5, 1.5].
- **S5:** SWM > DR, with lower bound > 0. This is the causal-structure claim.
- **S6:** the DR off-policy estimate falls inside the on-policy CI.
- **S7:** SWM > QACM.

**Verdicts.**
- PASS = S1-S4. The thesis claim also needs S5.
- KILL if the S1 lower bound ≤ 0, or S3 fails, or the arbiter abstains on more than 95 % of units.

## 5. Freeze checklist
**Needs user sign-off:**
- the seed and tag registration;
- the unit definition, T, and lock ≡ reject;
- π0;
- all thresholds;
- β;
- the cap;
- using the screen's R_or as reference;
- supplying the candidate relation from config if the CRT fails (a declared privilege).

**Files:**
- `docs/benchmark/E6P_ARBITER_PROTOCOL.md`
- `cdd_oran/decision/{units_p,collect_p,labels_p,mechanism_p,swm,bandit_dr,arbiter_p,ope}.py`
- `scratchpad/e6_dev/e6p_arbiter_pilot.py`
- tests: `test_e6p_{units,collect,labels,mechanism,arbiter}.py` and `test_decision_{swm,ope}.py`

**Risks a reviewer would attack:**
- Winner's curse from the screen's selection: this is DEV evidence only.
- Labels assume AA continuation, so they only guarantee a one-step improvement.
- SUTVA and interference.
- prot_below is not a standard KPM (TS 28.552).
- The EARTH energy model is extrapolated for ptx above 0.
- 40 evaluation seeds versus the screen's 8.
