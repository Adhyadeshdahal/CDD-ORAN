"""Registry of the xm-classic adapters: ``METHODS[name]()`` -> an api.Method.

Lazy: an adapter's module (and its dependencies: causal-learn, shap, torch) is imported only when it is looked up,
so a job that runs one method does not need the others' packages (e.g. a cloud kernel without causal-learn).
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping

_PATHS = {"pc": ("pc", "PC"), "notears": ("notears", "Notears"), "shap_dag": ("shap_dag", "ShapDag"),
          "two_tower": ("two_tower", "TwoTowerM"), "corr": ("corr", "Corr"), "granger": ("granger", "Granger"),
          "pcorr_hac": ("pcorr_hac", "PcorrHac")}


class _Lazy(Mapping):
    def __getitem__(self, name: str):
        mod, cls = _PATHS[name]
        return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)

    def __iter__(self) -> Iterator[str]:
        return iter(_PATHS)

    def __len__(self) -> int:
        return len(_PATHS)


METHODS = _Lazy()
