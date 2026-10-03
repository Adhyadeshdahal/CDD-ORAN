READY-TO-MERGE

# cdl (branch xm/cdl from feat/v2 8e4cb85, worktree CDD-ORAN-wt/xm-classic, 2026-10-03)

Brief: scratchpad/xmethod/briefs/cdl.md (R-50). Reference worktree (detached, read-only, removable):
CDD-ORAN-wt/cdl-ref = refactor/codebase 65b56f3. Detail: docs/xmethod/FIDELITY_CDL.md; data scratchpad/xmethod/results/cdl/.

## HAND-BACK
Deliverables
- `cdd_oran/xmethod/methods/_cdl_model.py`: port of refactor/codebase cdd_oran/models/cdl.py @65b56f3, verbatim
  except 4 `PORT:` lines (no base class; target_dim; informed_drop; fit_step / cmi_step).
- `cdd_oran/xmethod/methods/cdl.py`: adapter `cdl` (CDLMethod; registry classic.METHODS; runner path
  cdd_oran.xmethod.methods.cdl:CDLMethod). Native arm only; score = final EMA CMI; sign R-4; primary = R-29 placebo
  tau; conference CMI >= .16 = notes['native_declared']; R-37 second fit; torch seed tag 7804 (CONTRACT sec 6).
  Budget (Q2): steps = min(16 000, ceil(130 n / 128)).
- `tests/test_xmethod_cdl.py` 10 passed. `scripts/xm_classic_integration.py`: + cdl, + `--worlds` shard filter.
- Gate scripts scratchpad/xmethod/cdl_{fidelity,f4,gpu_check,lightning}.py, cdl_kaggle_*.sh. No new dependency.
Fidelity gates
- F1: authors' code, refactor/codebase 65b56f3 (model, train.py loop, configs/env_{i,ii}_cdl.yaml); torch 2.10.0.
- F2 PASS: golden probes reproduced (1.2e-7); original loop original vs port BITWISE equal (Env I + II, 1000 steps
  and full length). GPU does not reproduce CPU (Spearman .95) and is not faster -> CPU only.
- F3: Env II 16 / 16, recall = F1 = 1.00 (= paper): PASS. Env I (paper 1.00 / 1.00): seeds 46-49 10 / 10, 1.00 / 1.00;
  paper seed 45 9 / 10, F1 .947 (weak p3 -> K2 missed, CMI .08 vs .24-.93 at other seeds): PARTIAL at the paper's
  seed on CPU (paper ran CUDA). No false positive in any run.
- F4 (SYN R1, n 500 / 1000 / 4000, 50 test reps): power 1.00 on all planted edges (A0 -> Y0 .98 at n 500), signs
  correct. PRIMARY action family VALID: null actions declared at the placebo's rate, 0-.04. SECONDARY KPI -> KPI
  family INVALID under the action-placebo tau: .39-.73 (Q3: reported as not calibrated). Fixed .16 rule declares nothing at these n.
- F5: no DEV tuning beyond the R-29 tau. F6: deviation table in FIDELITY_CDL.md (A1-A9, P1-P4).
- Cost (R-13; clean, 1 thread, uv.lock stack): E2 R2 n 1000 33.6 / 4000 128.5 / 24000 513.6 CPU-s, RSS ~300 MB;
  E4 R3 n 4000 (2 fits) 43 s. SCALES: {8000, 24000} feasible.
- Integration (n 1000, kappa .25, tau on DEV 3000001-3, run + score on 3000000): all 12 cells run (table in doc).
Deviations / caveats
- Q1 mapping and Q2 budget as ruled (no meta-action; uniform mask; epoch budget).
- EMA from 0 shrinks CMIs by ~1 - .99^(steps / 100) (.05 at n 500 .. .80 at the cap): per-(cell, n) tau unaffected;
  the .16 secondary is not comparable across n (F4 SYN n <= 4000: 0 declared; integration n 1000: 0-2 per dataset).
- F3 jobs ran Kaggle's torch 2.11.0+cpu (my pinned install failed in the seed job: gymnasium is not on the PyTorch
  index); CPU runs are bitwise reproducible per host, not across hosts (.082 vs .087). F2 / F4 / cost / integration
  ran on the uv.lock stack (torch 2.10.0+cpu, numpy 2.4.2, scipy 1.18.1).
- The superseded rule-(a) Kaggle job cdl-f4-a (mine) was deleted to free a shared slot; its partials are in
  results/cdl/f4_rule_a.
Compute: Kaggle cdl-f3-a (failed setup), cdl-f3-b, cdl-cost-a (T4), cdl-f4-a (deleted), cdl-f4-b, cdl-int-a,
cdl-f3-seeds. Lightning: cdl-a / b / c, ~40 studio-min ~= 0.10 credits, all STOPPED 13:25 UTC.

## Questions
- Q1 Variable mapping. ANSWERED (orchestrator): own state node per action / lagged KPI / context column, targets Y,
  constant action input, uniform mask; both deviations in FIDELITY_CDL.md.
- Q2 Training budget. ANSWERED (orchestrator): (b) ~130 epochs, cap 16k steps; not (c).
- Q3 KPI -> KPI family. The single action-placebo tau cannot calibrate cdl's KPI -> KPI family: lagged-KPI
  inputs are z-scored (conference scaling) and their null CMIs run ~3x the action ones, so F4 null rates are
  .39-.73. Options: (a) report cdl's KPI -> KPI family as "not calibrated" (no claim; primary family unaffected);
  (b) a KPI-source placebo for score-only methods (e.g. a permuted lagged-KPI column, a protocol change for all
  score-only arms); (c) min-max scale the lagged KPIs like the actions (adapter deviation from the conference;
  would need a re-run of F4).
  ANSWERED (orchestrator): (a); KPI -> KPI family reported as 'not calibrated by the action placebo', no claim;
  primary action family unaffected. Noted in FIDELITY_CDL.md F4.
