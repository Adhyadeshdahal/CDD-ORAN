# Brief: cmi-fast (exact speed-up of CMI-kNN, ruling R-49)

CMI-kNN is the authors' own conference method; we may optimise its implementation but NOT change its statistic or
its null. Read CONTRACT.md (R-9, R-13, R-26, R-41, R-46, R-49), docs/xmethod/FIDELITY_CITESTS.md (cmi_knn rows,
F2-GPU), cdd_oran/xmethod/methods/cmi_knn.py (+ torch backend), tigramite CMIknn source (installed package: the
estimator, get_shuffle_significance / local permutation with shuffle_neighbors, knn, transform).
Cost today: ~3400 CPU-s (tigramite) or ~170 GPU-s per dataset at n 1000; ~2400 GPU-s at n 4000; ~19000 at n 8000.

Goal: make each dataset at n 1000 / 4000 / 8000 much cheaper with IDENTICAL results:
1. Profile first (where does time go: CMI estimate, neighbour search in Z, shuffle generation, per-shuffle re-estimate).
2. Exact optimisations, e.g. compute Z-space neighbour lists once per test and reuse them for every shuffle (they do
   not change under the local shuffle of X); batch many shuffles in one GPU pass; vectorise the per-shuffle CMI
   recomputation; reuse distance work that does not involve X. Keep the RNG stream so shuffles are the same draws.
3. Equivalence gate: on the F2 datasets (6 worlds) + DEV seeds 3_000_100-104 at n 500 / 1000 / 4000, the new backend
   gives the same statistic (exact or float tolerance stated, e.g. 1e-9 rel) and the SAME p-values / decisions as the
   current torch backend and tigramite (count mismatches; target 0).
4. Measure cost per dataset at n 1000 / 4000 / 8000 (GPU T4 and CPU), report speed-up.
No approximate neighbour search. Branch xm/cmi-fast from feat/v2 (the commit after the squash); implement as a new
backend option of cmi_knn (e.g. backend "torch_fast"), tests, FIDELITY_CITESTS.md section. Compute per R-46: Kaggle
GPU / Colab T4 / Lightning, in parallel. Status scratchpad/xmethod/status/cmi-fast.md (READY-TO-MERGE line 1;
questions "- Q<n> ..."). Commits: no co-author line, subject < 70 chars, few commits. Never push.
