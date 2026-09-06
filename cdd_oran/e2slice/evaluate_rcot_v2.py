"""POST-FREEZE RCoT-v2 recovery scoring (§9). Thin sibling to ``evaluate_rcot.py`` for the v2 method.

Loads the hash-bound ``discovery_rcot_v2.json`` FIRST (via ``load_discovery_rcot_v2``, which re-derives
the v2 frozen constants -- ``block_perm_reps = 299`` -- and ``PROTOCOL_COMMIT_V2`` fail-closed), then
scores against ``E2V2Env().true_adj_matrix()`` into a SEPARATE ``recovery_rcot_v2.json`` record. Every
metric helper is REUSED from the v1 evaluator (which itself reuses the pdCor metric machinery): there is
no new metric code and no free parameter in the scoring. v1's ``recovery_rcot.json`` is never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cdd_oran.e2slice.discovery_rcot_v2 import (
    PROTOCOL_COMMIT_V2,
    load_discovery_rcot_v2,
)
from cdd_oran.e2slice.evaluate_rcot import load_recovery_rcot, score_recovery_rcot

_RECOVERY_RCOT_V2_FILENAME = "recovery_rcot_v2.json"


def score_recovery_rcot_v2(dataset_dir: str | Path) -> dict[str, Any]:
    """POST-FREEZE RCoT-v2 recovery scoring: score the persisted v2 graph against E2 truth (§9).

    Thin wrapper over the v1 ``score_recovery_rcot``: reuses the identical embedding + shared metric
    helpers, overriding ONLY the discovery loader (``load_discovery_rcot_v2``) and the output filename
    (``recovery_rcot_v2.json``). The recovery record's ``protocol_commit`` flows through from the loaded
    v2 discovery record, so it is ``PROTOCOL_COMMIT_V2`` automatically.
    """
    return score_recovery_rcot(
        dataset_dir,
        discovery_loader=load_discovery_rcot_v2,
        out_filename=_RECOVERY_RCOT_V2_FILENAME,
    )


def load_recovery_rcot_v2(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    expected_discovery_hash: str | None = None,
) -> dict[str, Any]:
    """Load ``recovery_rcot_v2.json`` fail-closed, checking ``PROTOCOL_COMMIT_V2``.

    Thin wrapper over the v1 ``load_recovery_rcot`` (identical content-hash / parent-hash / numeric
    checks), overriding only the filename and the expected protocol_commit.
    """
    return load_recovery_rcot(
        dataset_dir,
        expected_dataset_hash=expected_dataset_hash,
        expected_discovery_hash=expected_discovery_hash,
        filename=_RECOVERY_RCOT_V2_FILENAME,
        expected_protocol_commit=PROTOCOL_COMMIT_V2,
    )
