# cdl fidelity (CONTRACT sec 4, F1-F6; ruling R-50)

Status: DRAFT 2026-10-03, worker cdl (branch xm/cdl). Code: `cdd_oran/xmethod/methods/cdl.py` (adapter, registry name
`cdl`), `cdd_oran/xmethod/methods/_cdl_model.py` (model port). Tests: `tests/test_xmethod_cdl.py`. Gate scripts:
`scratchpad/xmethod/cdl_fidelity.py` (F2 / F3 against the original code, run inside a worktree of
`refactor/codebase`), `scratchpad/xmethod/cdl_gpu_check.py` (GPU vs CPU), `scripts/xm_classic_fidelity.py f4 | cost
--method cdl`. Results: `scratchpad/xmethod/results/cdl/`.

## F1 source

The authors' NaNA 2026 conference method (CDD-ORAN, archive/xApp-Nana-Conference-2026 sec 3.2), an adaptation of
Causal Dynamics Learning (Wang et al. 2022). Code: branch `refactor/codebase` commit 65b56f3 (2026-08-20),
`cdd_oran/models/cdl.py` (model), `cdd_oran/experiments/train.py` (training loop), `configs/env_{i,ii}_cdl.yaml`.
torch 2.10.0 (uv.lock).

## F6 port and adapter: every deviation

Model port (`_cdl_model.py`, lines marked `PORT:`; with the defaults it is the original, see F2):

| # | change | reason |
|---|---|---|
| P1 | no `CausalModel` base class | the old `cdd_oran.models` package is not on feat/v2; the base only declared the abstract interface |
| P2 | `target_dim` (default = state_dim) | the conference predicted every state variable (NCPs and KPIs at t+1); a Study A row has only the KPIs at t+1 (Y) as targets |
| P3 | `informed_drop` (default True) | see A3 |
| P4 | `train_step` / `update_mask` delegate to `fit_step` / `cmi_step` (s_t, targets, a) | needed by P2; same arithmetic and torch RNG draws |

Offline adapter (`cdl.py`):

| # | conference | Study A adapter | reason |
|---|---|---|---|
| A1 | state nodes = NCPs (min-max to [0, 1]) + KPIs (z-scores); CDL action input = meta-action (param_id, bin, offset) of a single-parameter change | state nodes = every action column at t (min-max) + every non-NaN lagged KPI (z-score) + every context column (min-max); CDL action input = a constant 0 | the conference's causal edges were NCP_t -> KPI_t+1 with each NCP its own state node, which is what Study A scores per (action, KPI) candidate; the CDL action input is ONE node (one feature extractor over the whole vector), so feeding the actions there would give one CMI for all actions together. Study A rows have no meta-action (every action is drawn each row), so the action node carries a constant. (Q1, orchestrator-approved.) |
| A2 | targets = all state variables at t+1 | targets = Y (KPIs at t+1), z-scored | P2 |
| A3 | mask: 50 % "drop the changed param" (a[:, 0]), 50 % uniformly random node | uniformly random node only | no single changed param exists in a Study A row (R1: all actions redrawn; R2: all dithered). (Q1, orchestrator-approved.) |
| A4 | online: env steps 0..19 999, 80 % of transitions to the train buffer, training from step 4000, 1 Adam step / env step on 128 rows drawn from the growing buffer (16 000 steps on a ~16 000-row buffer = ~130 epochs); CMI batch every 10 steps; EMA every 10 CMI batches | offline, FIXED EPOCH BUDGET: gradient steps = min(16 000, ceil(130 n / 128)) (n 500 / 1000 / 4000 / 8000 / 24000: 508 / 1016 / 4063 / 8125 / 16 000); the whole dataset is the buffer; 128 rows drawn with replacement per step; CMI batch at step % 10 == 0, EMA (tau .99, from 0) every 10 CMI batches | ruling Q2 (orchestrator): the data-to-update ratio of the paper's regime. The brief's first rule (16 000 steps for every n, 4096 epochs at n 500) overfits: the CMI is measured on training rows, and on synthetic n 500 (b 1) null action CMIs averaged .49 nats (max .72), null KPI -> KPI .83 (max 1.11), planted .84 (results/cdl/f4_rule_a, 16 fits); at n 1000 nulls .19-.52. Not chosen: CMI on a held-out split (not the conference's estimator). |
| A4b | EMA from 0, 160+ updates (Env I 160, Env II 460) | EMA from 0, steps / 100 updates (5 at n 500): every CMI of a fit is shrunk by ~1 - .99^(steps / 100) (.05 / .10 / .33 / .80 at n 500 / 1000 / 4000 / cap) | consequence of A4 (same EMA code); a common factor within a fit and within an n, so the per-(cell, n) R-29 tau is unaffected; the fixed .16 secondary is not comparable across n |
| A5 | no 80 / 20 split for the CMI (CMI batches come from the train buffer; the test buffer only fed an MSE monitor) | all rows used for training and CMI | the test buffer played no role in the graph |
| A6 | declaration CMI >= .16 (configs/env_i_cdl.yaml; the paper text says .2) | primary: score > R-29 conformal placebo tau (R-50); CMI >= .16 = `native_declared` (secondary) | R-50 |
| A7 | unsigned | sign = R-4 pcorr given the native set | R-4 |
| A8 | CUDA by default, non-deterministic | CPU, `torch.use_deterministic_algorithms`, torch seed from tag 7804, 1 thread | reproducibility (R-35); GPU does not reproduce CPU and gives no speed-up (see cost) |
| A9 | - | R-37: primary fit without `P_placebo_conf`; its candidates from a second fit with every action | R-37 / R-38 |

Not ported: the planners, the MLP world model, `predict_next_state` use at deployment (discovery only).

## F2 port == original (identical inputs)

| check | result |
|---|---|
| golden fixtures `tests/golden/Environment{I,II}_CDL.json` (refactor/codebase): untrained CDL (seed 12345, golden kwargs) `predict_next_state` probe; the fixtures pin no learned graph | port and original reproduce both, max abs diff 1.2e-7 (fixture tolerance rtol 1e-5): PASS (`results/cdl/f2_golden.jsonl`; also `tests/test_xmethod_cdl.py`) |
| the ORIGINAL training loop `cdd_oran.experiments.train.main`, original CDL vs port (`train.get_model` patched), CPU, deterministic, Env I and Env II, 1500 env steps (1000 gradient steps) | final CMI and every weight BITWISE equal: PASS (`results/cdl/f2_equiv.jsonl`) |
| same, full length (F3 runs, Env I 20k / Env II 50k steps) | CMI matrices bitwise equal: PASS |
| GPU (T4, CUDA) vs CPU, same seed, same row batches (SYN n 1000) | NOT reproduced: max abs diff .076 (scores up to .73), Spearman .95 (torch's mask draws come from the CUDA generator); GPU wall 252 s vs CPU 272 s. GPU not used: cdl runs CPU-only (R-26) |

## F3 the paper's graph-recovery result (paper sec 5.1: seed 45, Env I 20k / Env II 50k steps, recall = F1 = 1.00)

Original loop, port and original model, CPU deterministic, 1 thread (Kaggle `cdl-f3-b`, image torch 2.11.0+cpu, numpy
2.1.3, gymnasium 1.2.0; results `results/cdl/f3/`):

| env | edges | threshold .16 (repo config) | threshold .2 (paper) | min true-edge CMI | max null CMI | verdict |
|---|---|---|---|---|---|---|
| II | 16 | TP 16, FP 0, recall 1.00, F1 1.00 | same | .864 | .090 | PASS |
| I | 10 | TP 9, FP 0, recall .90, F1 .947 | same | .082 (p3 -> K2) | .068 | PARTIAL |

Env I misses p3 -> K2: K2 = exp(-(p1 - 50)^2 / (2 p3)^2) with p3 in [0, 3] and p1 in [0, 300], so p3 moves K2 only
when p1 is near 50; the other nine true CMIs are 1.23-2.43. The paper's run used CUDA (non-deterministic).

Seed check, Env I, port in the original loop (Kaggle `cdl-f3-seeds`; `results/cdl/f3_seeds/`):

| seed | TP / FP / FN (.16 and .2) | recall | F1 | p3 -> K2 CMI | max null CMI |
|---|---|---|---|---|---|
| 45 | 9 / 0 / 1 | .90 | .947 | .087 | .092 |
| 46 | 10 / 0 / 0 | 1.00 | 1.00 | .278 | .064 |
| 47 | 10 / 0 / 0 | 1.00 | 1.00 | .927 | .080 |
| 48 | 10 / 0 / 0 | 1.00 | 1.00 | .319 | .071 |
| 49 | 10 / 0 / 0 | 1.00 | 1.00 | .238 | .100 |

F3 verdict: Env II PASS (the paper's result exactly). Env I reproduces the paper's recall = F1 = 1.00 at 4 of 5 seeds;
at the paper's seed 45 on CPU it misses the weak p3 -> K2 edge (9 / 10, F1 .947); no false positive in any run. Seed 45
gives p3 -> K2 .082 in `cdl-f3-b` and .087 here (same torch 2.11.0+cpu): CPU runs are bitwise reproducible on one host
(F2) but not across Kaggle hosts. Deviation: both F3 jobs ran the Kaggle image's torch 2.11.0+cpu, numpy 2.1.3,
gymnasium 1.2.0, not the uv.lock pins (the seed job's pinned install failed: gymnasium is not on the PyTorch wheel
index, so the combined pip command installed nothing). The F2 checks ran locally on the uv.lock stack.

## Cost (R-13; rule A4)

One `run()` per dataset, CPU, 1 thread, one process alone (Lightning CPU studio `cdl-c`, torch 2.10.0+cpu, numpy 2.4.2,
scipy 1.18.1; `scripts/xm_classic_fidelity.py cost`; `results/cdl/cost_lightning_cdl-c.jsonl`). E2 R2 is the largest
world (9 actions, 6 KPIs, 90 candidates):

| dataset | n 1000 | n 4000 | n 24000 |
|---|---|---|---|
| E2 R2: CPU-s (peak RSS MB) | 33.6 (297) | 128.5 (301) | 513.6 (310) |
| E4 R3 (two fits, R-37) | - | 43.0 (282) | - |

Cost is linear in the gradient steps (so in n up to the 16 000-step cap at n ~ 15 750, flat after); n 8000 ~ 260 s by
the same rate. Every n up to 24000 is far inside the 2 CPU-h budget: cdl SCALES ({8000, 24000} included).

## Integration (brief step 4)

`scripts/xm_classic_integration.py --methods cdl --worlds <W> --n 1000 --kappa 0.25` (Kaggle `cdl-int-a`, torch
2.10.0+cpu, numpy 2.4.2, scipy 1.18.1): per (world, regime) cell, tau = R-29 conformal cutoff on DEV seeds
3000001-3 (E4 at lambda 1.5), then `cdd_oran.xmethod.runner run` + `score` on DEV seed 3000000. Every cell runs
(`results/cdl/integration/`, table `results/cdl/integration_summary.txt`). ONE dataset per cell: plumbing check, not
an evaluation.

| cell | tau | primary (action): declared, TP / true, null FPR, placebo declared | secondary (KPI -> KPI): declared, TP / true, null FPR | P_placebo_conf declared | CPU-s |
|---|---|---|---|---|---|
| E1 R1 | .0018 | 10, 4/4, .500, 1 | 7, 2/2, .357 | - | 46.2 |
| E1 R2 | .0023 | 5, 4/4, .083, 0 | 9, 2/2, .500 | - | 42.4 |
| E2 R1 | .0034 | 13, 11/16, .062, 0 | 8, 0/0, .222 | - | 63.2 |
| E2 R2 | .0047 | 11, 10/16, .031, 1 | 7, 0/0, .194 | - | 63.2 |
| E3 R1 | .0024 | 4, 4/4, .000, 0 | 7, 3/3, .182 | - | 43.1 |
| E3 R2 | .0020 | 6, 4/4, .125, 0 | 12, 3/3, .409 | - | 37.0 |
| E4 R1 | .0007 | 0, 0/1, -, 0 | 0, 0/0, - | - | 15.5 |
| E4 R2 | .0004 | 1, 1/1, -, 1 | 1, 0/0, 1.000 | - | 13.9 |
| E4 R3 | .0010 | 1, 1/1, -, 0 | 0, 0/0, - | 1 | 31.8 |
| E4 R4 | .0006 | 1, 1/1, -, 1 | 1, 0/0, 1.000 | 1 | 29.0 |
| E5 R1 | .0074 | 5, 4/6, .100, 0 | 1, 0/1, .067 | - | 45.9 |
| E5 R2 | .0044 | 5, 5/6, .000, 0 | 4, 1/1, .200 | - | 45.0 |

Taus are small (.0004-.0074) because at n 1000 the EMA shrink factor is ~.10 (A4b). The KPI -> KPI family shares the
action-placebo tau (as the other score-only adapters, FIDELITY_CLASSIC Q-C2) and its null rates are high here (.18-.50 in
E1 / E2 / E3 / E5): lagged-KPI CMIs run larger than action CMIs, so a cut tuned on an action placebo is not calibrated
for that family. The fixed .16 rule (secondary) declares 0-2 edges per dataset at n 1000 (shrink factor .10).

## F4 synthetic (planted edges, null level; rule A4)

`scratchpad/xmethod/cdl_f4.py` (Kaggle `cdl-f4-b`, torch 2.10.0+cpu, numpy 2.4.2, scipy 1.18.1; 360 fits;
`results/cdl/f4/`, summaries `results/cdl/f4_summary.jsonl`, `results/cdl/f4_by_family.json`). Generator
`_classic_synth` R1 (as FIDELITY_CLASSIC F4): planted A0 -> Y0 +, A1 -> Y1 -, K0 -> Y0 +, K1 -> Y2 + at b = 1 (alt),
global null b = 0. tau = R-29 conformal cutoff of the P_placebo scores of 10 SEPARATE tune datasets of the same kind,
applied to 50 test datasets. 95 % Wilson CIs; m = number of candidate tests.

Power (alt data, tau): every planted edge 1.00 at n 1000 and 4000 (A0 -> Y0 .98 at n 500), sign correct 1.00.

Declaration rate of NULL candidates under tau, by source family:

| n | data | tau | P_placebo (m 150) | other action sources (A2; A0 / A1 off-target) | lagged-KPI sources |
|---|---|---|---|---|---|
| 500 | alt | .00094 | .040 [.019, .085] | .037 [.022, .063] (m 350) | .700 [.650, .746] (m 350) |
| 500 | global null | .00114 | .020 [.007, .057] | .022 [.012, .040] (m 450) | .711 [.668, .751] (m 450) |
| 1000 | alt | .00172 | .000 [.000, .025] | .009 [.003, .025] | .526 [.473, .578] |
| 1000 | global null | .00226 | .000 [.000, .025] | .002 [.000, .013] | .453 [.408, .500] |
| 4000 | alt | .00371 | .000 [.000, .025] | .000 [.000, .011] | .394 [.345, .446] |
| 4000 | global null | .00342 | .007 [.001, .037] | .024 [.014, .043] | .729 [.686, .768] |

Reading:
- PRIMARY family (action -> KPI, R-6): VALID. Null action sources are declared at the placebo's rate (0-.04, every
  CI lower bound <= .022), i.e. the placebo is exchangeable with the other null actions, which is what the R-29 rule
  needs; power 1.00.
- SECONDARY family (lagged KPI -> KPI): INVALID under the action-placebo tau (null rates .39-.73). The lagged KPIs are
  z-scored inputs (conference scaling; A1) while actions are min-max scaled to [0, 1]; their noise CMIs run ~3x larger
  (global null, mean null CMI action .0005 / .0008 / .0015 vs KPI .0014 / .0023 / .0041 at n 500 / 1000 / 4000), so
  a cut tuned on an action placebo does not transfer. Same structural issue as the other score-only adapters'
  KPI -> KPI family (FIDELITY_CLASSIC Q-C2), far larger here. Ruling Q3 (orchestrator): cdl's KPI -> KPI
  (secondary) family is reported as "not calibrated by the action placebo", with no claim made on it; the primary
  action family is unaffected.
- Pooled over both families (`f4_summary.jsonl`) the null-edge rate is .16-.32 and the global-null FWER .96-1.00:
  that is the KPI family.
- The conference's fixed threshold (.16) declares NOTHING at these n (all CMIs < .05: EMA shrink A4b plus noise-level
  CMIs): power 0, FPR 0.
