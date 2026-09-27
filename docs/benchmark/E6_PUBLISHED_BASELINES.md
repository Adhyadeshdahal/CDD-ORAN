# E6 published baselines: decision rules, E6 mappings, fidelity

Status: DRAFT 2026-09-28. Code: `cdd_oran/envs/e6/published.py`. Tests: `tests/test_e6_published.py`.
Metric and comparator rules: `E6_METRIC.md` §3–4 (every published method runs behind a declared mitigation wrapper,
with settings in its favour and the same DEV tuning budget, and is also reported on its native metric).

Conventions used below.
- **[P]** means the paper states it. Quotes are verbatim, with the section, equation or algorithm number.
- **[I]** means we inferred it: the paper is silent, and E6 forces a choice.
- **[F]** means the choice was made in the method's favour.
- "Wrapper" or "resolver" means a decision rule the paper does not contain. It is **ours** and must be labelled as ours
  in every table.
- E6 arbiter API (`env.py` docstring): `arb(obs) -> {"decisions": [...], "writes": [], "rollback": [...]}`.
  - Each decision is one of `accept`, `reject`, `("modify", v)`, `defer` or `("lock", secs)`.
  - `obs` holds `t`, `new_reports`, `config`, `requests`, `locked`, `changes`, `churn_cap` and `static`.
  - Arbiters never see the plant.

## Summary

| method | what the paper decides | E6 status | why |
|---|---|---|---|
| QACM (Wadud et al., IEEE TGCN 2024) | the value of one conflicting parameter (Alg. 1 / Eq. 3) | **implemented** `QACM` + `KPIPredictor` | decision rule faithful; its KPI predictor has to be learned from DEV logs |
| CMF (Adamczyk et al., IEEE ComMag 2023) + Wadud'25 SBD / P-x | block or allow each control message (DCD / ICD + priority CR) | **implemented** `CMF(mode="priority"\|"sbd")` | fully rule-based |
| PACIFISTA (del Prever et al., IEEE TMC 2025) | which **apps** to deploy (severity greedy, §8) | **implemented** `Pacifista` → `subset(A_DPLY)` | needs a standalone profiling episode per xApp |
| Djidjev & Kaminski, arXiv:2606.06459 + 2606.06663 | detection / tracking only | **implemented** detector + **our** resolver `Djidjev` | the papers say explicitly that it is "not … an autonomous mitigation mechanism" |
| Sharma et al., arXiv:2510.13031 | detection + ATE/CATE only | **interface only** `Sharma` | xgboost, dowhy and econml are not installed; needs randomised dither logs |
| two-tower, arXiv:2601.13213 | detection only (supervised) | **interface only** `TwoTower` | needs a ground-truth label matrix Y and the rules of its ref. [8] |
| GRAPHICA GCN, arXiv:2503.03523 | per-timestamp conflict class + root-cause analysis | **interface only** `Graphica` | architecture details unreported; labels undefined for E6; no mitigation |

The xApp descriptors shared by all methods are in `published.MANIFEST`. They are operator-declared, as QACM and CMF
assume ("the MNO is expected to provide the xApp details, including their ICPs and KPIs", Wadud'25 §IV).

| xApp | ICPs | own KPI (per cell) | QoS q, δ (1 = minimise) | source |
|---|---|---|---|---|
| MRO | cio, ttt, hys | too-late HO ratio | 0.02, 1 | [I] MRO's nominal trigger; E6 has no numeric RLF target |
| TS | cio | eMBB p5 throughput | 2 Mb/s, 0 | E6 SLA target |
| ES | carrier | energy | none (always met) | [I][F] energy has no SLA target, so the method is favoured on SVR |
| SLICE | ll_ratio | LL p95 delay | 100 ms, 1 | E6 SLA target |

---

## 1. QACM: QoS-aware conflict mitigation (Wadud, Golpayegani, Thomas, Marchetti; IEEE TGCN 8(3), 2024)

**What it decides.** QACM is the Conflict Mitigation Controller (CMC). When the Conflict Detection Controller (CDC)
reports a conflict over a parameter p_l, the CMC picks one value for p_l. The goal is "to ensure that the majority, if
not all, xApps meet or exceed their specified QoS thresholds by identifying an optimal setting for the contentious ICP"
(§II).

**Rule [P].**
- Eq. (3a): minimise Σ_i w_i d_i ζ − (Σ_i s_i)².
- (3b) Σ_i w_i = 1.
- (3d)/(3e): d_i ≥ 0 is the shortfall (δ_i = 0, KPI maximised) or the excess (δ_i = 1, KPI minimised) of U_i(p_l)
  against q′_i.
- (3f)–(3h): s_i = 1 iff the threshold is met.
- (3i): p_l ∈ [p_l^min,opt, p_l^max,opt].
- (3j): |X′| ≥ 2.
- The dynamic heuristic, **Alg. 1**, is what we implement:
  - "for each p_l ∈ [p_l^min,opt, p_l^max,opt] … for each i … Obtain predicted U_i(p_l) from x′_i";
  - "cost[i] = w_i d_i × ζ";
  - "fCost = Σ(cost) − (Σ(s))²";
  - "if minCost > fCost then … Update p_l^opt = p_l".
- ζ = 10³ ("we used ζ = 10³", §III).
- The range is the union of the xApps' ranges: "{min(p^min,x′1, p^min,x′2), max(p^max,x′1, p^max,x′2)}" (§VI-C).
- Weights: equal weights give QACM; weights assigned by the Conflict Supervision (CS) xApp give QACM-P (§VI-C, §VII).
- Utilities: "z-score normalization technique for converting the associated KPI of an xApp to utility" (§VI-A). An xApp
  with several KPIs uses "a weighted-average method" (§VI-A).
- KPI prediction: "We presume that xApps are pre-trained with offline KPI prediction models capable of estimating KPI
  values based on provided ICPs" (§VI-A). The model is an ANN with "four hidden layers, each with 128 neurons … tanh …
  dropout … 0.2 … adam … mean squared error … 10 epochs with a batch size of 10". It was chosen over polynomial
  regression (PR) by EVS / R² / MSE (§VI-B, Table III).

**Detection (the CDC that triggers QACM).** QACM "assume[s] that all other components of the CMS framework perform their
tasks efficiently, and the CMC is notified by the CDC" (§VI). The same authors' runtime CDC (Wadud et al., INFOCOM'25
workshop, arXiv:2411.03326, §IV) works as follows [P]:
- "The CDC is triggered only when a KPI violation occurs".
- The KPI Degradation Occurrences (KDO) store "alerts the CDC" when "any SLA-sensitive KPI falls below the predefined
  threshold".
- The Recently Changed Parameter (RCP) store finds the parameter changed at t_clock.
- If "the instructing xApp … is the same as the xApp associated with the degraded KPI … 'no conflict'".
- If they share the parameter, the conflict is "direct". If the parameter is in the KPI's parameter group PkG (their
  Alg. 1), it is "indirect". Otherwise it is "implicit".

**E6 mapping (`QACM.__call__`).**
1. **CDC, run pre-action [I][F].** A pending request of xApp *a* on knob *k* conflicts with every other deployed xApp *b*
   whose SLA-sensitive KPI violates q_b on a cell that *k* acts on (CIO[s,n] acts on s and n; every other knob acts on
   its own cell). It also conflicts with every other xApp requesting the same knob in the same second (direct conflict).
   - The paper's CDC is post-action and uses the RCP. Screening the pending request instead lets QACM prevent the harm
     rather than repair it afterwards, which favours QACM.
   - `anticipate=True` [I] also involves xApps that are not violating. It is off by default and can be tuned on DEV.
2. **|X′| < 2** means no conflict, so the request is accepted (Eq. 3j).
3. **Candidates.** The one-step actuator grid of *k* (range, quantisation and max step from `ric.LIMITS`), restricted to
   [min, max] of {current value} ∪ {step-clipped proposals}.
   - [I] E6 xApps do not report an "optimal configuration range". The value in force stands in for the other party's
     range, and the proposals stand in for the requesters' ranges.
4. **U_i(p_l)** is the mean over the acted-on cells of the predicted z-scored KPI of x_i (the paper's weighted average,
   with equal weights). The thresholds are z-scored with the same statistics.
   - An xApp without a threshold (ES) or without a current KPI signal counts as met: d = 0, s = 1.
5. **Alg. 1** exactly: ζ = 10³, equal weights (or `weights=` for QACM-P), first strict minimum.
   - Iteration order [I][F]: Alg. 1 does not specify tie handling. `order="prop_first"` scans the proposals first, then
     the current value, so a blind predictor passes requests through (noarb behaviour). `order="ascending"` is the
     literal loop order.
6. **Decision.** p_opt = a proposal → accept that request and reject the others on the knob. p_opt = current value →
   reject all. Otherwise → `("modify", p_opt)` on the last request.

**KPI predictor (`KPIPredictor`, `transitions`).**
- One model per (knob type, KPI).
- Features: role one-hot (self/src/dst), Δ = p − current, current value, current cell KPI, current cell PRB utilisation.
  All are what the arbiter sees at the decision second.
- Target: the cell KPI in the first report of that granularity whose window starts at or after the change.
- Both the paper's ANN (exact spec, torch) and PR (degree 2) are fitted, and the one with the higher held-out R² is
  kept, per (type, KPI) [F]. The paper picked ANN globally.
- Null samples (unchanged knobs of the same type, Δ = 0) are included, so the model also predicts the status quo.
- Fewer than 30 samples → a persistence model: QACM then sees no effect of that knob [declared].

**Training data (contract).** `fit(episodes)`, where each episode is the `env.log` of a DEV run
(`published.run_logged`).
- Noarb logs only cover the knobs the xApps actually move. In a 120 s M4 noarb run (seed 12) the sample counts were:
  cio ~600–750 per KPI, ll_ratio ~40–60, carrier ~20, MRO knobs ~0.
- For coverage, add **dither episodes** (bounded random one-step changes on every knob type). Carrier changes are
  dwell-limited to one per 120 s per sector.
- Target: at least ~1000 samples per (type, KPI). The DEV budget must be fixed before any comparison. This is an
  estimate, not measured.

**Native metric.** Per-xApp KPI distance to its QoS threshold and the number of xApps meeting their thresholds
(§VII–VIII, Figs. 8–12). Report "share of conflicts in which all involved xApps meet q" next to SVR.

**Settings in its favour [F].**
- Pre-action CDC.
- Thresholds taken from E6's own SLA targets.
- ES has no energy threshold.
- Model choice per (type, KPI) by held-out R².
- Pass-through tie order.
- Dither training data.
- DEV-tuned `anticipate` and predictor data size.

**Fidelity risks.**
- (a) The predictor is the whole method in practice. The paper admits "KPI prediction is a complex process influenced
  not only by ICPs but also by the dynamic state of the network" (§IX). Our state features (current KPI, utilisation)
  go beyond the paper's ICP-only inputs [I][F]. Smoke R² (120 s of training data) was 0.07–0.75 on the main pairs.
- (b) The range inference in step 3.
- (c) The pre-action CDC is a deviation, though a favourable one.
- (d) A per-cell KPI averaged over the two CIO cells can hide a violation on one of them.
- (e) An older port for the legacy env is in `cdd_oran/planners/qacm.py` (world-model scoring with extra
  ensemble/utility terms). It is **not** this rule and must not be used as the E6 baseline.

## 2. CMF: conflict mitigation framework (Adamczyk et al., IEEE ComMag 2023), with Wadud'25 SBD / priority variants

**Rule [P].**
- DCD: "each new control message … is compared to all currently effective xApp control decisions; if any decisions
  share the control target and at least one of the modified parameters, data about the conflicting decisions is
  provided to the CR Agent" (§III-B).
- ICD: "first maps the target parameter onto predefined PGs … checks the entries in 'Recently changed Parameter Groups'"
  (§III-C).
- CR in the evaluation: "If a given xApp is prioritized, each of its decisions takes effect on the network regardless of
  conflicts" (§IV-A). "If no conflicts are detected, any decisions … take effect in the order they are provided".
- Wadud'25 §V adds "a set back to default (SBD) method where the TXP was reset to its default value" and P-ES / P-MRO
  priority.

**E6 mapping (`CMF`).**
- The Recently Changed Parameters (RCP) store holds the requests this arbiter accepted. Each stays "effective" for its
  xApp's control time span, which is the xApp's cadence [I]. `span_s` overrides it.
- A request conflicts if another xApp's effective decision is:
  - on the same knob (direct); or
  - on a knob in the same parameter group acting on a shared cell (indirect). The group definition is set by `groups`:
    - `"manifest"` = Wadud'25 Alg. 1 PkG (faithful);
    - `"cell"` = every knob on a shared cell [I], broader.
- Requests are processed in priority order. The prioritised xApp always wins. A lower-priority conflicting request is
  rejected (`mode="priority"`) or replaced by `("modify", initial value)` (`mode="sbd"`).
- Default order = `baselines.PRIORITY` (SLICE, MRO, TS, ES). [F] The order and `groups` are tuned on DEV under the
  common budget.

**Training data.** None.

**Native metric.** Counts of call blocks, RLF, HO and ping-pong under the three CMF modes (§IV-B).

**Fidelity risks.** The PG definitions are the operator's choice in the paper. With `"manifest"` groups and no MRO
activity (short episodes), CMF sees zero conflicts and equals noarb (smoke, seed 11).

## 3. PACIFISTA (del Prever et al., IEEE TMC 2025)

**What it decides [P].** Whether to deploy apps. The mitigation module decides "on the deployment of O-RAN applications
… avoiding the deployment of an application that would generate too large of a conflict" (§4.1). Profiling "happens
offline" in a sandbox (§4.2).

**Rule [P].**
- Profiles are one ECDF per KPM and per operating condition, with each app run alone (§5.1).
- Distance: INT = sqrt((1/L) ∫|F1(x) − F2(x)| dx), with L = max(x) − min(x) (Table 1). "INT distance is convenient for
  … measuring conflict severity".
- Severity: σ^K = H(D^f(K*, c)), with H a weighted average, median or maximum (§7).
- Mitigation (§8):
  1. Deploy every app with no conflict of any type.
  2. Take "a = argmax_{a∈A*\A_DPLY} I_a" and deploy it if "max_{a*∈A_DPLY} σ^K_{a,a*}(K*|c) ≤ δ_TOL".
  3. Repeat until A* is empty.
- δ_TOL ∈ [0, 1]. The paper evaluates 0.25 and 0.5.

**E6 mapping (`Pacifista`).**
- `fit(profiles)`: `profiles[x]` = logs of E6 episodes run with `subset((x,))`, i.e. that xApp alone. K* = PRB
  utilisation, LL p95, eMBB p5, too-late ratio, energy (per-cell values pooled).
- H = mean.
- Conflicts: MRO/TS share CIO (direct). Every knob moves the shared load/PRB KPMs, so every pair also has a KPM conflict
  [I]. Step 1 therefore deploys nobody when there are two or more apps.
- The greedy uses priority `order` and δ_TOL. The runtime arbiter is `subset(A_DPLY)` for the whole episode.
- modify / defer / lock are not in the paper and are not used.

**Training data.** At least 3,000 samples per KPM per app ("collected at least 3,000 samples for each xApp for each
slice"). With 24 cells:
- fast KPIs: ~125 s of standalone run;
- eMBB p5: ~625 s;
- MRO counters (30 s): ~3,750 s;
- energy (60 s): ~7,500 s.

These are per xApp and per condition.

**Native metric.** σ matrices, KPI change against severity, and the co-deployable sets at each δ_TOL (Tables 3–4, 7).

**Settings in its favour [F].**
- δ_TOL ∈ {0.25, 0.5} plus a DEV sweep.
- KPM weights and priority order tuned on DEV.
- A condition-aware re-run per load bin is allowed [I]: "δ_TOL … set dynamically each time PACIFISTA is run". This is
  **not implemented** (TODO).

**Fidelity risks.**
- σ measures how dissimilar the apps' standalone distributions are, not the harm of running them together, and it has
  no sign.
- Decisions are per app, not per request, so it cannot beat the best static subset (`E6_METRIC.md` comparator).
- Per-cell pooling is our choice.

## 4. Djidjev & Kaminski: dependency detection (arXiv:2606.06459) and runtime tracking (arXiv:2606.06663)

**What they decide [P].** Detection only. Paper A: "we will study how the recovered Boolean dependency matrix can
support downstream tasks such as conflict monitoring and mitigation" (§IV). Paper B: "intended as an explainable signal
for conflict diagnosis and slow-loop model refresh, not as an autonomous mitigation mechanism" (Abstract).

**Rule [P].**
- Paper A Eq. (4): B_P(i,t) = 1 if ΔP_i(t) ≠ 0. μ̂_j and σ̂_j = max(SD, 1e-8) of ΔK_j are computed over the null rows
  T_0 = {t : Σ_i|ΔP_i(t)| = 0}. The event is "B_K(j,t) = 1{|z_j(t)| > z_th}". z_th = 4.5–5 is "near-perfect" (Fig. 3).
- Paper B:
  - Eq. (1): b_K = L ⊗_B b_P, a Boolean OR over the parents.
  - Eq. (4): ℓ̂_j = argmin_ℓ ‖B_K,j − ℓ ⊗_B B_P‖₀ over the window, with "Ties are broken by minimum Hamming distance
    from the current row estimate". The search is exhaustive over 2^n_P.
  - Alg. 1: "if window is full and (L̂_cur is empty or (2) fails) then Recompute L̂_cur". W ∈ {4, 8, 16, 32}.

**E6 mapping (`Djidjev`).**
- Rows are (KPI, cell) for LL p95, eMBB p5 and too-late ratio.
- Columns are 7 local parameters: CIO out, CIO in, hys, ttt, ll_ratio, carrier, sleep. With n_P = 7, the exhaustive
  2^7 search is faithful [I]: the local aggregation keeps it tractable.
- B_P = a local parameter changed within [previous report t0, this report t1) [I]. This lag alignment is needed because
  the papers assume a same-step response.
- The null rows used for μ̂ and σ̂ are **local**, meaning no local change occurred [I]. E6 has almost no globally quiet
  seconds. They are fitted on DEV burn-in logs.
- z_th = 4.5 and W = 16 by default [F: the paper's best point, DEV-tunable].
- **Resolver (ours) [I][F].** Reject a request if another xApp's KPI row on an acted-on cell has this knob's parameter as
  a tracked parent **and** that KPI currently violates its QoS. Otherwise accept.
  - `modify` is not used: the papers have no magnitudes.
  - An optional `defer` on tracker change alarms is not implemented (TODO).

**Training data.** Burn-in logs, used for the null-row statistics only. 120 s gave 50 of 72 rows. Use ≥ 10 min of DEV
noarb or dither.

**Native metric.** Edge P/R/F1 and Hamming distance against the true support; change recall/precision/F1 with
τ = W; false-alarm fraction; delay. On E6, truth for the support is available from the plant design, as
evaluation-only data.

**Fidelity risks.**
- Lag alignment and local null rows are both deviations.
- The OR model cannot attribute concurrent changes or signs.
- The single-column trigger has no error control.
- B_P fires only on requests the arbiter accepted, so rejecting starves the evidence.
- The resolver is entirely ours and is labelled "Djidjev monitor + our resolver".

## 5. Sharma et al., arXiv:2510.13031 (NFV-SDN 2025): interface only

**Rule [P].**
- Regression: one model per KPI from the RAN control parameters (RCPs). XGBoost was chosen as "highest performance"
  (§V-B).
- Permutation importance and mean|SHAP| (Table III).
- DAG: "creating causal edges from the RCPs which are the most influential towards their associated KPIs" (§IV-C.1). No
  numeric cutoff is given. A per-KPI relative cutoff τ_rel ∈ (0.061, 0.077] reproduces their Table IV [I].
- ATE via DoWhy backdoor, with refuters passing at "high p-values (> 0.05)".
- CATE via "CausalForestDML … with XGBoost selected as the base regressor".

**No mitigation [P].** The ATEs "enable the MNOs to design a conflict mitigation framework that can notify the xApps …
and suggest parameter adjustments to remain within acceptable operational tolerances" (§V-C).

**E6 contract (`Sharma`).**
- `fit(episodes)` on **randomised dither** DEV logs. Their data: "1000 episodes" with RCPs randomly sampled.
- Resolver (ours) [I]:
  - accept if every predicted child-KPI change CATE·(prop − cur) stays within the QoS tolerance;
  - otherwise `("modify", largest safe step)`;
  - otherwise reject.
- [F] KPM state covariates may be given to the DML (their §IV-C permits "adjust for all known confounders").

**Blocked on.**
- xgboost, dowhy and econml are not installed.
- The dither-episode data contract.

**Risks.**
- The repo's earlier SHAP-DAG port (`scripts/e2_baseline_shap_dag.py`) uses HistGradientBoosting, OLS backdoor,
  τ = 0.10 and no CATE. That does not reproduce their DAG.
- A per-unit linear ATE is a poor fit for discrete steps such as carrier.

## 6. Two-tower, arXiv:2601.13213: interface only

**Rule [P].**
- Per-node L-sample vectors pass through Linear → ReLU → Linear with H = 16 and are L2-normalised.
- Scores: "S_pk = α|Zp||Zk|ᵀ".
- Binarisation: row sparsemax, then Â = 1(P > 0), i ≠ j. The result is augmented with A_known.
- Conflict identification uses the rules of their ref. [8].
- Training is **supervised** BCE against "the ground-truth label matrix Y" (Eq. 14).
- Detection only: it could "raise alarms … and/or … generate policies" (§II), but neither is specified.

**E6 contract (`TwoTower`).**
- `fit(episodes, Y)`. Y must come from a separately seeded E6 twin or sandbox. E6 truth gives an upper bound only.
- Resolver (ours): flag a request whose (xApp, knob) is in an identified conflict with an active xApp, then apply a
  fixed swept policy (defer / reject / priority).

**Blocked on.** The label source for Y, and the identification rules of [8], which are not in the paper.

**Risks.**
- The repo's earlier "TwoTower" (`scripts/e2_baseline_gnn.py`) is **not faithful**: it uses self-supervised MSE, scalar
  encoders and a τ_rel threshold instead of sparsemax.
- Symmetric scores carry no direction.
- Whether the diagonal is masked before sparsemax is unspecified.

## 7. GRAPHICA GCN classifier (Al Shami, Yan, Fapi; arXiv:2503.03523): interface only

**Rule [P].**
- Binary per-timestamp states: an xApp is "1" if active; a parameter or KPI is "1" "if the value … is changed from its
  previous timestamp value".
- Three subgraphs (G_PA, G_KP, G_P′P).
- 2-layer GCN, mean pool, FC layer, classes "0: normal, 1: direct, 2: implicit, 3: indirect".
- Focal loss (Eq. 1). Adam, lr 0.01, batch 128.
- RCA: "checks the source node with more than one incoming edge".
- No mitigation: "Future work will focus on developing … mitigation strategies" (§VI).

**E6 contract (`Graphica`).**
- `fit(episodes, labels)`.
- Resolver (ours): classify the hypothetical post-accept state. If it is not normal, accept the highest-priority
  root-cause xApp and reject or defer the rest.

**Blocked on.**
- Hidden sizes and node features are unreported.
- torch_geometric is not installed.
- With structural labels, almost every second with two or more active xApps is a "conflict" in E6, so the classifier
  would re-learn a deterministic rule.

---

## Open issues (for the env / benchmark owners)

1. **Requested knobs missing from `obs["config"]`.** TS requests the reverse CIO ("cio", n, s) where s ∉
   neighbours[n]. The env still applies it via `knob_set`, but that knob is not in `env.knobs`, so `obs["config"][k]`
   raises KeyError.
   - Seed 12, M4: 19 such knobs in 30 s.
   - `published.py` falls back to `request["cur"]`.
   - Other arbiters that index `obs["config"]` will crash.
2. **`obs["static"]["xapps"]` is not "declared knobs only".** It lists every knob type for every xApp (`env.static`).
   The published methods therefore use the operator manifest `published.MANIFEST` instead.
3. **`env.log` does not record requests or decisions** (only config and reports). Detectors that need the requesting
   xApp per change (GRAPHICA, the xApp-level PACIFISTA indicators) need it logged.
4. **MRO is silent in short episodes.** It has a 30 s cadence and needs at least 3 events, so the direct MRO/TS CIO
   conflict does not appear in 120 s runs. DEV comparisons need the full episode length.
5. **The DEV data budget for learned pieces must be frozen before any comparison.** Candidates:
   - QACM predictor: dither logs;
   - Djidjev: null-row burn-in;
   - PACIFISTA: standalone profiles.

## Smoke (2026-09-28; DEV seed 11, M4, 60 s warm-up + 60 s scored; predictor/burn-in = one 120 s noarb log of seed 12)

Every arbiter runs in about 3 s per episode. One seed with 60 s scored proves only that the code runs; these are not
results.
- **QACM:** 65 conflicts flagged, 20 rejects, 0 modifies.
- **CMF (priority and SBD):** 0 conflicts, so identical to noarb.
- **Djidjev:** 38 rows tracked, 4 parent edges, 0 rejects.
