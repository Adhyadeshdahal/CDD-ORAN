"""O-RAN environments package.

Two environment systems live here — do not conflate them (see ``cdd_oran/envs/README.md``):

- **LIVE:** the E-series redesign in :mod:`cdd_oran.envs.v2` (``V2Env``, ``E1V2Env..E4V2Env``). This is
  the active benchmark; the live discovery/decision pipelines are in :mod:`cdd_oran.e1slice` /
  :mod:`cdd_oran.e2slice` and ``scripts/e*_gate.py``.
- **RETIRED:** legacy Environment I–IV in :mod:`cdd_oran.envs.legacy` (quarantined 2026-09-22). Import
  legacy names explicitly, e.g. ``from cdd_oran.envs.legacy import get_env``.

This package intentionally imports **nothing** at module top level, so importing ``cdd_oran.envs`` (which
happens transitively whenever a ``cdd_oran.envs.v2.*`` submodule is imported) loads no legacy code.
"""
