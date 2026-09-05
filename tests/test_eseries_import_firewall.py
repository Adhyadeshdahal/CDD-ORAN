"""Import-firewall tests: the E-series must not transitively load the RETIRED legacy envs.

Background: ``cdd_oran/envs/__init__.py`` EAGERLY imports ``env_i..iv``. Historically the pure
precision/recall/F1 helper ``_prf`` lived in ``analysis/threshold_sweep.py``, which does a
module-level ``from cdd_oran.envs import get_env`` -- so any consumer of ``_prf`` (recovery_metrics,
the e1slice/e2slice evaluators) transitively pulled in the legacy envs just to reach a pure metric.

This suite proves ``_prf`` now lives in the pure ``cdd_oran.analysis.prf`` module and that the
analysis layer (``prf``, ``recovery_metrics``) no longer loads the legacy envs. Each check runs in a
FRESH subprocess so an unrelated in-process import cannot mask the result.

Known remaining seam (documented as a strict xfail below): the e1slice/e2slice EVALUATORS import
``cdd_oran.envs.v2.{e1,e2}`` at module level for §9 recovery-truth, and importing ANY
``cdd_oran.envs.v2.*`` submodule executes the parent ``cdd_oran/envs/__init__.py`` -> env_i..iv. The
``_prf`` move alone cannot break that; closing it needs a separate, larger decision (defer the
evaluators' env-truth imports, or make ``envs/__init__`` lazy) -- out of scope for this hygiene fix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Legacy modules that must NOT be pulled in by a pure analysis import.
_LEGACY = ("cdd_oran.envs", "cdd_oran.envs.env_i", "cdd_oran.envs.env_ii",
           "cdd_oran.envs.env_iii", "cdd_oran.envs.env_iv")


def _loaded_legacy_modules(import_target: str) -> list[str]:
    """Import ``import_target`` in a fresh interpreter; return which legacy modules it loaded."""
    code = (
        f"import {import_target}\n"
        "import sys, json\n"
        f"print(json.dumps([m for m in {_LEGACY!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


def test_prf_module_is_env_free():
    assert _loaded_legacy_modules("cdd_oran.analysis.prf") == []


def test_recovery_metrics_is_env_free():
    # The core delivered decoupling: recovery_metrics (the E-series' PRF consumer) no longer loads
    # the legacy envs, because it now imports _prf from the pure cdd_oran.analysis.prf module.
    assert _loaded_legacy_modules("cdd_oran.analysis.recovery_metrics") == []


def test_threshold_sweep_reexports_prf_for_legacy_callers():
    # Legacy callers (auto_threshold, edge_stability) import threshold_sweep._prf; it must still be
    # available and be the SAME object as the pure implementation (no divergence).
    from cdd_oran.analysis import prf, threshold_sweep

    assert threshold_sweep._prf is prf._prf


def test_prf_output_is_unchanged():
    # Behavior/number guard: _prf on a known graph gives the exact pre-move result.
    import numpy as np

    from cdd_oran.analysis.prf import _prf

    gt = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
    # tp=3 (positions (0,0),(1,1),(2,0)), fp=1 ((1,2)), fn=2 ((0,2),(2,1)), tn=3
    out = _prf(pred, gt)
    assert out == {
        "precision": 3 / 4,
        "recall": 3 / 5,
        "f1": 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5)),
        "accuracy": 6 / 9,
        "edges": 4,
        "tp": 3,
        "fp": 1,
        "fn": 2,
    }


@pytest.mark.parametrize("evaluate_module", [
    "cdd_oran.e1slice.evaluate",
    "cdd_oran.e2slice.evaluate",
])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Remaining seam (out of scope for the _prf hygiene fix): the evaluators import "
        "cdd_oran.envs.v2.{e1,e2} for recovery-truth, which executes the eager "
        "cdd_oran/envs/__init__.py and loads env_i..iv. Closing this needs a separate decision "
        "(defer the env-truth imports, or make envs/__init__ lazy). When fixed, this xpasses -> "
        "remove the xfail marker."
    ),
)
def test_eseries_evaluate_is_env_free_known_remaining_seam(evaluate_module: str):
    assert _loaded_legacy_modules(evaluate_module) == []
