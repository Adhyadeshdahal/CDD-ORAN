"""DEPRECATED import path: this module is now ``cdd_oran.decision.pmrt`` (PMRT, the Predictable Matched-Filter
Randomization Test; formerly labelled "MSCR+"; docs/benchmark/METHOD_NAMES.md).

Thin compatibility shim so historical scripts keep importing: every attribute is forwarded to ``pmrt`` (the same
objects, so results are identical), with the pre-rename names mapped to the new ones. ``PLUS_VERSION`` keeps its
historical value "mscr-crt-units-plus-v0" (``pmrt.LEGACY_VERSION``); results computed through the shim carry
``pmrt.PMRT_VERSION``. The byte-exact pre-rename source is ``git show 4fc2cd9:cdd_oran/decision/crt_units_plus.py``.
"""
from __future__ import annotations

import warnings

from . import pmrt as _pmrt

_RENAMED = {
    "PlusConfig": "PmrtConfig", "PlusData": "PmrtData", "PLUS_STREAM": "PMRT_STREAM", "PLUS_VERSION": "LEGACY_VERSION",
    "load_plus_pool": "load_pmrt_pool", "plus_data": "pmrt_data", "subset_plus": "subset_pmrt",
    "plus_data_from_records": "pmrt_data_from_records", "fit_plus": "fit_pmrt", "family_tests_plus": "family_tests_pmrt",
    "run_crt_units_plus": "run_pmrt", "edges_plus": "edges_pmrt",
}

warnings.warn("cdd_oran.decision.crt_units_plus is deprecated: use cdd_oran.decision.pmrt (PMRT)", DeprecationWarning,
              stacklevel=2)


def __getattr__(name: str):
    return getattr(_pmrt, _RENAMED.get(name, name))


def __dir__():
    return sorted(set(dir(_pmrt)) | set(_RENAMED))


__all__ = sorted(set(_pmrt.__all__) | set(_RENAMED))
