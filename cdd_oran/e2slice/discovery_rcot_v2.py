"""RCoT-v2 E2 graph discovery -- the frozen RCoT method with ONE constant changed: ``block_perm_reps``
99 -> 299. Everything else is IDENTICAL to v1 (``discovery_rcot.py``) and is IMPORTED, never copied.

Why v2 (the diagnosed bug, provable a priori)
---------------------------------------------
Frozen RCoT-v1 (``PROTOCOL_COMMIT`` ``eba381a``, ``block_perm_reps = 99``) fixed the lagged KPI->KPI
over-selection but recovered only ~2-3 of the 16 true NCP->KPI edges per seed. The diagnosis
(``reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md``) traced this NOT to low test power (the block_perm
conditional test rejects the true edge raw at ~80%) but to a **BH x permutation-resolution interaction**:
with ``B = 99`` the permutation p-value floor is ``1/(B+1) = 0.010``, which sits ABOVE the per-target
BH-FDR leading threshold ``q/m = 0.05/14 = 0.00357``. So a strong-but-quantized true-edge p-value is
discarded at the BH selection step even though the test detects the edge. Raising ``B`` to 299 lowers the
floor to ``1/300 = 0.00333 < 0.00357``, making the strongest true edge selectable on its own. A truth-free
B-sweep (diagnosis §B2) confirmed the fix: BH-power ``0.086 -> 0.742`` while the KPI->KPI FP stays flat at
``~0.010`` (block_perm-grade). The looser ``analytic_hbe`` null was REJECTED because its FP grows with n
(to 0.120 at n=4000), re-opening the over-selection v1 was frozen to close.

What changes, and what does NOT
-------------------------------
- CHANGED: ``block_perm_reps`` 99 -> 299 (``FROZEN_BLOCK_PERM_REPS_V2``). This is the ONLY difference.
- UNCHANGED (imported from ``discovery_rcot.py``, not reimplemented): every RCoT numeric (median-heuristic
  bandwidth, RFF residualization, statistic ``T = n*||Cxy||_F^2``, HBE, block_perm geometry), and every
  other §11 constant -- ``dz = 25``, ``dxy = 5``, ``ridge = 1e-6``, ``q = 0.05``, ``block_size = 25``,
  ``residual_epsilon = 1e-12``, ``rng_seed = 0``, ``null_method = "block_perm"``, the relative
  residual-collapse guard, the guard-fraction HALT, the BH-FDR selection, and the conditioning set (the
  other 13 candidates). v2 is a THIN sibling: it only supplies its own frozen config, protocol_commit, and
  DISTINCT artifact filenames so v1's on-disk records are never overwritten.

Freeze status
-------------
FROZEN. ``PROTOCOL_COMMIT_V2`` is the SHA of the committed
``docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md`` (the freeze point), recorded in every
``discovery_rcot_v2.json`` and re-derived fail-closed on load, mirroring the v1 freeze mechanism.
v1 (``eba381a``, ``block_perm_reps = 99``) is SUPERSEDED but stays fully intact and loadable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cdd_oran.e2slice.discovery_rcot import (
    RCoTDiscoveryConfig,
    load_discovery_rcot,
    write_discovery_rcot,
)

# Full SHA of the frozen v2 protocol commit (``docs: freeze E2 RCoT-v2 discovery protocol (009)``).
# Recorded in every discovery_rcot_v2.json and re-derived fail-closed on load. Distinct from v1's
# PROTOCOL_COMMIT (eba381a) so v1 and v2 masks never cross-load.
PROTOCOL_COMMIT_V2 = "8052e10675b38831a50f76e759add55d8acda1b6"

# The one and only v2 change: 99 -> 299 permutations (floor 1/300 = 0.00333 < BH threshold 0.05/14 =
# 0.00357). See the module docstring / diagnosis report for the a-priori justification.
FROZEN_BLOCK_PERM_REPS_V2 = 299

# Distinct artifact filenames -- v2 writes BESIDE v1, never over it.
_DISCOVERY_RCOT_V2_FILENAME = "discovery_rcot_v2.json"
_RECOVERY_RCOT_V2_FILENAME = "recovery_rcot_v2.json"
_DISCOVER_RCOT_V2_DESCENDANTS = (_RECOVERY_RCOT_V2_FILENAME,)


def frozen_config_v2() -> RCoTDiscoveryConfig:
    """The frozen RCoT-v2 config: identical to v1's ``frozen_config()`` except ``block_perm_reps = 299``.

    Every other field is the v1 frozen default (block_perm null, dz=25, dxy=5, ridge=1e-6, q=0.05,
    block_size=25, residual_epsilon=1e-12, rng_seed=0), so the only thing that differs from v1 is the
    permutation count -- exactly the diagnosed fix.
    """
    return RCoTDiscoveryConfig(block_perm_reps=FROZEN_BLOCK_PERM_REPS_V2)


def write_discovery_rcot_v2(
    dataset_dir: str | Path, force: bool = False
) -> dict[str, Any]:
    """Discover the E2 graph via RCoT-v2 (block_perm, B=299) and write ``discovery_rcot_v2.json``.

    Thin wrapper over the v1 ``write_discovery_rcot`` orchestration: reuses the identical dataset load,
    ``discover_graph_rcot`` numerics, record builder, hashing, and atomic write, overriding ONLY the
    frozen config (``frozen_config_v2``), the artifact filename, the descendant-guard set, and the
    ``protocol_commit`` (``PROTOCOL_COMMIT_V2``). v1's ``discovery_rcot.json`` is never touched.
    """
    return write_discovery_rcot(
        dataset_dir,
        force=force,
        config_factory=frozen_config_v2,
        filename=_DISCOVERY_RCOT_V2_FILENAME,
        descendants=_DISCOVER_RCOT_V2_DESCENDANTS,
        protocol_commit=PROTOCOL_COMMIT_V2,
    )


def load_discovery_rcot_v2(dataset_dir: str | Path) -> dict[str, Any]:
    """Load ``discovery_rcot_v2.json`` fail-closed, re-deriving the v2 frozen constants + protocol_commit.

    Thin wrapper over the v1 ``load_discovery_rcot`` fail-closed loader: reuses the identical content-hash
    check, shape contract, and guarded-never-selected check, re-deriving every §11 constant against the
    v2 frozen config (so ``block_perm_reps`` must be 299) and the ``protocol_commit`` against
    ``PROTOCOL_COMMIT_V2``. A v1 mask (block_perm_reps=99, ``eba381a``) will NOT load as v2, and vice versa.
    """
    return load_discovery_rcot(
        dataset_dir,
        filename=_DISCOVERY_RCOT_V2_FILENAME,
        expected_protocol_commit=PROTOCOL_COMMIT_V2,
        config_factory=frozen_config_v2,
    )
