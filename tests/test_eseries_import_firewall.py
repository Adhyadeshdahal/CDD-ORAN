"""Import-firewall tests: the E-series must not transitively load the RETIRED legacy envs.

Background: the RETIRED legacy envs now live in ``cdd_oran/envs/legacy/`` (quarantined 2026-09-22) and
``cdd_oran/envs/__init__.py`` imports nothing at module top. Historically the pure precision/recall/F1
helper ``_prf`` lived in ``analysis/threshold_sweep.py``, which does a module-level
``from cdd_oran.envs.legacy import get_env`` -- so any consumer of ``_prf`` (recovery_metrics, the
e1slice/e2slice evaluators) transitively pulled in the legacy envs just to reach a pure metric.

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

# The RETIRED legacy env CLASS modules that must NOT be pulled in. Note: the parent package
# ``cdd_oran.envs`` itself IS expected to load whenever a live ``cdd_oran.envs.v2.*`` submodule is
# imported (v2 is a subpackage of it); after the Option-B fix that package is env-free (it imports
# env_i..iv only lazily), so the firewall is precisely "env_i..iv are not loaded", not "the envs
# package is untouched".
_LEGACY = ("cdd_oran.envs.legacy.env_i", "cdd_oran.envs.legacy.env_ii",
           "cdd_oran.envs.legacy.env_iii", "cdd_oran.envs.legacy.env_iv")


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
def test_eseries_evaluate_is_env_free(evaluate_module: str):
    # The E-series evaluators import cdd_oran.envs.v2.{e1,e2} for §9 recovery-truth. After the
    # Option-B root fix (cdd_oran/envs/__init__.py imports env_i..iv LAZILY, not at module top),
    # importing a cdd_oran.envs.v2.* submodule -- and hence importing the evaluators -- no longer
    # eager-loads the retired legacy envs.
    assert _loaded_legacy_modules(evaluate_module) == []


@pytest.mark.parametrize("v2_module", [
    "cdd_oran.envs.v2.e1",
    "cdd_oran.envs.v2.e2",
])
def test_v2_env_submodule_import_is_legacy_free(v2_module: str):
    # Root-cause guard: importing ANY live v2 env submodule must not pull the legacy env_i..iv via
    # the parent cdd_oran/envs/__init__.py.
    assert _loaded_legacy_modules(v2_module) == []


def test_legacy_get_env_mean_std_reexports_still_work():
    # Behavior-neutral: the lazy PEP 562 __getattr__ on cdd_oran.envs.legacy must still serve the
    # re-exported legacy names to any archival caller, identical to a direct submodule import.
    import cdd_oran.envs.legacy as legacy
    from cdd_oran.envs.legacy.env_i import ORANEnvironment1 as DirectEnv1
    from cdd_oran.envs.legacy.statistics import get_env_i_mean_std as direct_mean_std

    assert legacy.ORANEnvironment1 is DirectEnv1
    assert legacy.get_env_i_mean_std is direct_mean_std
    assert legacy.get_env_iii_mean_std.__name__ == "get_env_iii_mean_std"


def _loaded_modules_matching(import_target: str, prefixes: tuple[str, ...]) -> list[str]:
    """Import ``import_target`` in a fresh interpreter; return loaded modules under ``prefixes``."""
    code = (
        f"import {import_target}\n"
        "import sys, json\n"
        f"p = {prefixes!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if any(m == x or m.startswith(x + '.') for x in p))))\n"
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


@pytest.mark.parametrize("live_module", [
    "cdd_oran.planners.sequence",
    "cdd_oran.benchmark.learned_world_model",
    "cdd_oran.benchmark.rollout",
    "cdd_oran.decision.arbiter",
])
def test_live_v2_path_loads_no_legacy_package(live_module: str):
    # Plan 012 step 1: the live V2/E-series decision path must not load ANY module of the retired
    # legacy package. cdd_oran.planners/__init__ is lazy, so importing the sequence planner no
    # longer drags in the legacy planners (cost.py -> cdd_oran.envs.legacy).
    assert _loaded_modules_matching(live_module, ("cdd_oran.envs.legacy",)) == []


def test_decision_arbiter_is_torch_free():
    assert _loaded_modules_matching("cdd_oran.decision.arbiter", ("torch",)) == []


def test_planners_package_lazy_names_still_resolve():
    # Behavior-neutral: the lazy PEP 562 __getattr__ serves the same objects as direct imports.
    import cdd_oran.planners as planners
    from cdd_oran.planners.ensemble import EnsembleAggregator
    from cdd_oran.planners.qacm import QACM

    assert planners.QACM is QACM
    assert planners.EnsembleAggregator is EnsembleAggregator
    assert callable(planners.get_planners) and callable(planners.build_aggregator)
    with pytest.raises(AttributeError):
        _ = planners.NoSuchPlanner
