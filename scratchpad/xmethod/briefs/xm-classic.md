# Brief: xm-classic (PC, NOTEARS, SHAP-DAG, two-tower, correlation, Granger)
Read first: scratchpad/xmethod/CONTRACT.md, cdd_oran/xmethod/api.py, scripts/e2_baseline_shap_dag.py,
scripts/e2_baseline_gnn.py, cdd_oran/decision/baselines_disc.py (how these baselines were adapted on E6, incl. the
two-tower l1 note), docs/benchmark/E6_PUBLISHED_BASELINES.md.
Deliver adapters in cdd_oran/xmethod/methods/ implementing api.Method: pc, notears, shap_dag, two_tower, corr,
granger (E3 only).
- pc: causal-learn (pip name causal-learn, add with uv), standard CI test (Fisher-z, and KCI if affordable); use
  background knowledge that actions are exogenous (no edges into actions) and time order (t -> t+1); read off
  action->KPI and lagged KPI->KPI edges. notears: the authors' reference code (Zheng et al. 2018; vendor with
  licence + commit, or the maintained package), linear version, default lambda / w_threshold.
- shap_dag and two_tower: wrap the existing scripts (do not edit them); match how the papers apply them; record
  every adaptation.
- Fidelity gates F1-F6 (CONTRACT sec 4). F3: PC and NOTEARS reproduce a package / paper example; SHAP-DAG and
  two-tower reproduce their E2 behaviour from the earlier studies.
- Until xm-harness lands, test on your own synthetic api.Dataset objects; then the orchestrator tells you to rebase.
- Measure CPU-s and peak RAM at n in {500, 1000, 4000, 8000, 24000}.
Tests: tests/test_xmethod_classic.py.
