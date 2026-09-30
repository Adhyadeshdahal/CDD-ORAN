"""Decision stack (legacy-free): discovery, referees and the E6 / E6-P decision pipeline.

Submodules are imported on demand (no eager heavy imports). Map (docs/ARCHITECTURE.md has the full picture):

- plans / arbiter / world model: ``plans``, ``arbiter``, ``world_model``, ``effect_model``, ``gate``, ``rank_eval``,
  ``adapters.e6``.
- E6 template discovery: ``trace``, ``features``, ``discovery``, ``probe``, ``collect``, ``crt`` (mscr-crt-v1).
- E6-P units, labels and collection: ``units_p``, ``labels_p``, ``collect_p``, ``gt_p``, ``slots_p``, ``ope``.
- E6-P discovery tests: ``crt_units`` / ``crt_units_v2`` (MSCR-CRT v1 / v2), ``pmrt`` (PMRT, the Predictable
  Matched-Filter Randomization Test), ``fdr_layer`` (PMRT's declaration layer), ``eprocess_units`` (exact e-process
  companion); scoring and benchmarks: ``edge_score``, ``baselines_disc``, ``disc_bench``.
- E6-P referees: ``referee_p`` (regime referee, K-L0), ``mapgate`` (map-driven MapGateV2, option (a)).
- Deprecated import paths (label-only rename, docs/benchmark/METHOD_NAMES.md): ``crt_units_plus`` -> ``pmrt``,
  ``mscr_multi`` -> ``fdr_layer``.
"""
