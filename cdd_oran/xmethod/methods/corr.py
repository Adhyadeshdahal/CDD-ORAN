"""``corr``: marginal Pearson correlation baseline (xm-classic, CONTRACT sec 3 "simple").

score = |r(source, target)| with source = the action at t or the lagged KPI at t and target = the KPI at t+1;
p = the two-sided Pearson t-test p-value (``scipy.stats.pearsonr``; valid for i.i.d. rows, approximate under the
serial dependence of R2); sign = sign(r). Declaration (primary): pooled BY per family (action | kpi) at q = .05.
No hyperparameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from cdd_oran.xmethod import api

from ._classic_common import ClassicBase, Scored, columns, resolve, target_index


class Corr(ClassicBase):
    name = "corr"
    version = "1.0"
    uses_p = True
    method_idx = 5
    defaults: dict[str, Any] = {"q": 0.05}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            x = None if j is None else (cols.actions[:, j] if fam == "action" else cols.lags[:, j])
            y = cols.Y[:, ti]
            if x is None or np.std(x) == 0 or np.std(y) == 0:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            r, p = stats.pearsonr(x, y)
            out.append(Scored(s, t, fam, float(abs(r)), int(np.sign(r)), float(p)))
        return out, {"sign_rule": "sign(r)", "test": "scipy.stats.pearsonr (t-test)"}
