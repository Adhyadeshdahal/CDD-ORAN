# Brief: xm-citests (independence-test family adapters + fidelity gates)
Read first: scratchpad/xmethod/CONTRACT.md, cdd_oran/xmethod/api.py, docs/benchmark/METHOD_NAMES.md (MSCR),
cdd_oran/discovery/mscr.py, cdd_oran/e1slice/discovery_v2.py (partial correlation), cdd_oran/e2slice/discovery.py
(pdCor), cdd_oran/e2slice/discovery_rcot_v2.py + docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md (RCoT v2).
Deliver adapters in cdd_oran/xmethod/methods/ implementing api.Method: mscr, pcorr, pdcor, rcot2, cmi_knn.
- Reuse the existing implementations (wrap, do not edit); each keeps its own null (row permutation / asymptotic)
  because that is what is being tested. cmi_knn: kNN CMI (KSG-type, e.g. tigramite CMIknn) with its standard
  permutation null; record package version.
- Fidelity gates F1-F6 (CONTRACT sec 4) for each. F2: rcot2 vs R package RCIT (Strobl) on identical inputs, run R on
  Kaggle or Colab; pdcor vs the `dcor` package. F4 on synthetic planted / null data (state n, reps, CIs).
- Until xm-harness lands, test on your own synthetic api.Dataset objects; then the orchestrator tells you to rebase.
- Measure CPU-s and peak RAM per run at n in {500, 1000, 4000, 8000, 24000}; state which methods cannot scale
  (pdCor is O(n^2) memory).
Tests: tests/test_xmethod_citests.py.
