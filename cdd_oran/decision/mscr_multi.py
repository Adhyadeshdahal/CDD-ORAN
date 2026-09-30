"""DEPRECATED import path: this module is now ``cdd_oran.decision.fdr_layer`` (the statistic-agnostic declaration
layer of PMRT: weighted BY / one-sided / DAGGER / e-BH; docs/benchmark/METHOD_NAMES.md).

Thin compatibility shim so historical scripts keep importing: every attribute is forwarded to ``fdr_layer`` (the same
objects, so results are identical). ``MULTI_VERSION`` keeps its historical value "mscr-multi-v1"
(``fdr_layer.LEGACY_VERSION``). The byte-exact pre-rename source is ``git show 4fc2cd9:cdd_oran/decision/mscr_multi.py``.
"""
from __future__ import annotations

import warnings

from . import fdr_layer as _fdr

_RENAMED = {"MULTI_VERSION": "LEGACY_VERSION"}

warnings.warn("cdd_oran.decision.mscr_multi is deprecated: use cdd_oran.decision.fdr_layer", DeprecationWarning,
              stacklevel=2)


def __getattr__(name: str):
    return getattr(_fdr, _RENAMED.get(name, name))


def __dir__():
    return sorted(set(dir(_fdr)) | set(_RENAMED))


__all__ = sorted(set(_fdr.__all__) | set(_RENAMED))
