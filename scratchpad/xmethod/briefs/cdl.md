# Brief: cdl (port the authors' conference method CDL into Study A, ruling R-50)

Source: git branch refactor/codebase: cdd_oran/models/cdl.py (CDL model: per-dimension feature extractors, masked
predictor, CMI per edge from masked vs full log-likelihood, EMA tau .99, threshold), configs/env_i_cdl.yaml /
env_ii_cdl.yaml (lr .001, threshold .16 (paper .2), batch 128, grad clip 10, 20k/50k steps, init 4000),
cdd_oran/experiments/train.py, tests/golden/Environment{I,II}_CDL.json. Paper: archive/xApp-Nana-Conference-2026/
sections/03_proposed_method.tex (CMI-Based Causal Edge Discovery) + 04_experimental_setup.tex.
Read CONTRACT.md (R-1..R-50; esp. R-2, R-4, R-13, R-22/23, R-29, R-37, R-46), cdd_oran/xmethod/{api,runner,score,
covariates}.py, methods/two_tower.py and _classic_common.py (score-only adapter pattern, R-37 diagnostic fit).

Steps
1. F1/F3 fidelity: copy cdl.py (and only what it needs) into cdd_oran/xmethod/methods/_cdl_model.py with a header
   citing branch + commit; reproduce the golden Environment I/II CDL results (or the closest reproducible quantity:
   learned graph / CMI values) with the original training loop; state tolerance. If the golden runs need the old env
   code, run them from a worktree of refactor/codebase.
2. Adapter cdd_oran/xmethod/methods/cdl.py (name "cdl"): train CDL OFFLINE on the Dataset rows (state = lagged KPIs
   (+ context if any), action = actions at t, target = Y), same architecture and hyper-parameters as the conference
   config; training budget fixed in GRADIENT STEPS (state the rule, e.g. conference 20k steps, independent of n; or
   epochs if that is the paper's rule); score per candidate (source, target) = final EMA CMI; sign via R-4
   (pcorr given R-3 Z); declared = score > conformal placebo tau (R-29, via ClassicBase-style tune()); notes report
   the conference fixed-threshold (.16) declarations as secondary; R-37: fit without P_placebo_conf, its candidates
   from a second fit; RNG seed from tag 7804 (register in CONTRACT note), deterministic torch where possible.
3. F4: synthetic null level / planted edges as the other score-only methods (FIDELITY_CLASSIC pattern); cost per
   dataset at n 1000 / 4000 / 24000 (CPU and GPU), R-13 budget.
4. Integration: one DEV seed per cell through runner.load_method + score.score; tests.
Docs: FIDELITY_CLASSIC.md (or new FIDELITY_CDL.md) section. Branch xm/cdl from feat/v2 (current tip). Compute per
R-46 (Kaggle / Lightning / Colab in parallel; Lightning credits logged; ask before cumulative 3). Status
scratchpad/xmethod/status/cdl.md (READY-TO-MERGE line 1; questions "- Q<n> ..."). Commits: no co-author line,
subject < 70 chars, few commits. Never push.
