"""xmethod adapter ``mscr``: MSCR-v2 (``cdd_oran.discovery.mscr`` internals, wrapped unchanged).

Statistic S* = max over single conditioners g != i of the stratified correlation ratio; null = the frozen
within-stratum random-partition bank, S*_null = max_{g != i} bank. Frozen constants NC=6, NB=8, MIN=40.
Banks are built by the frozen ``_build_banks`` with the frozen per-target streams ``default_rng([seed, j, n_perm])``
(a second column group g >= 1, R3 / R4 P_placebo_conf, uses ``[seed, j, n_perm, g]``; audit L4). Ruling R-40:
S* is a max over SINGLE conditioners, so Z_eq enters only as more single conditioners: mscr_eq is reported as
"single-conditioner max statistic; cannot condition on the joint design set" and is excluded from C2b.

Deviations from the v2 E2 use (F6): (1) B raised from native 2999 to sequential 9999 for pooled-BY resolution
(R-9): the bank holds 9999 draws and the p-value is the Besag-Clifford rule (h = 20) applied to the bank draws in
stream order (the shared bank gives no compute saving from stopping); ``{"n_perm": 2999, "bc_h": None}``
reproduces ``discover_mscr`` exactly. (2) Every candidate source is tested (lagged KPIs too, when they are
candidates; v2 tested parameters only). (3) Conditioner pool = the common Z (``_citests_common``). (4) Bank seed
derived from the citests RNG tag (v2: the dataset seed). (5) Primary family = pooled BY (R-6 families); v2's
per-target BY over the candidates is kept as ``declared_native``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cdd_oran.discovery.mscr import (
    MSCR_VERSION,
    MSCRConfig,
    _build_banks,
    _s_star,
    by_declare,
    frozen_config,
    resolve_n_jobs,
)
from cdd_oran.xmethod.methods._citests_common import (
    B_MAX,
    BC_H,
    CITestBase,
    Prepared,
    method_seed,
    progress,
    sequential_p_from_draws,
)


class MSCRMethod(CITestBase):
    name = "mscr"
    version = f"xm-{MSCR_VERSION}-bc"
    method_key = 1

    def default_config(self) -> dict[str, Any]:
        c = frozen_config()
        return {"nc": c.nc, "nb": c.nb, "min_stratum": c.min_stratum, "n_perm": B_MAX, "bc_h": BC_H,
                "device": "cpu", "n_jobs": None, "seed": None}

    def native_config(self) -> dict[str, Any]:
        return {**self.default_config(), "n_perm": frozen_config().n_perm, "bc_h": None, "arm": "native"}

    def _test(self, prep: Prepared, data, cfg: dict[str, Any]) -> dict[str, Any]:
        m = len(prep.pairs)
        score, p, k_used = np.zeros(m), np.zeros(m), np.zeros(m, dtype=int)
        for g, (cols, es) in enumerate(prep.column_groups()):
            self._group(prep, data, cfg, cols, es, score, p, k_used, g)
        mcfg = frozen_config()
        native = np.zeros(m, dtype=bool)
        for j in {j for _, j in prep.pairs}:
            es = [e for e, (_, jj) in enumerate(prep.pairs) if jj == j and prep.family[e] != "placebo_conf"]
            native[es] = by_declare(p[es], mcfg.q)
        return {"score": score, "p": p, "declared_native": native,
                "notes": {"mc_draws": k_used.tolist(),
                          "native_rule": "per-target BY q=.05 over the target's candidate sources"}}

    def _group(self, prep: Prepared, data, cfg, cols, es, score, p, k_used, group_index: int = 0) -> None:
        """MSCR on S[:, cols] (conditioner pool = cols) for the candidates ``es``."""
        tested = sorted({prep.pairs[e][0] for e in es})
        rest = [i for i in cols if i not in tested]
        x = prep.S[:, tested + rest]
        if x.shape[1] < 2:
            raise ValueError("mscr: need >= 2 source columns")
        mcfg = MSCRConfig(nc=cfg["nc"], nb=cfg["nb"], min_stratum=cfg["min_stratum"], n_perm=cfg["n_perm"],
                          q=frozen_config().q)
        seed = cfg["seed"] if cfg.get("seed") is not None else method_seed(data.seed, self.method_key)
        targets = sorted({prep.pairs[e][1] for e in es})
        # streams as discover_mscr: target column j of Y uses default_rng([seed, j, n_perm]); a further column group
        # g >= 1 (e.g. the P_placebo_conf group in R3 / R4) gets its own stream [seed, j, n_perm, g] (audit L4)
        key = (lambda j: [int(seed), j, mcfg.n_perm]) if group_index == 0 else \
            (lambda j: [int(seed), j, mcfg.n_perm, group_index])
        rngs = [np.random.default_rng(key(j)) for j in targets]
        banks = _build_banks(x, prep.Y[:, targets], mcfg, rngs, n_jobs=resolve_n_jobs(cfg.get("n_jobs")),
                             device=cfg.get("device", "cpu"))
        bank_of = dict(zip(targets, banks, strict=True))
        pos = {i: c for c, i in enumerate(tested)}
        for e in es:
            i, j = prep.pairs[e]
            denom, null, strata = bank_of[j]
            c = pos[i]
            score[e] = _s_star(x[:, c], c, denom, strata, mcfg)
            keep = np.ones(null.shape[0], dtype=bool)
            keep[c] = False
            s_null = np.max(null[keep], axis=0)
            p[e], k_used[e] = sequential_p_from_draws(score[e], s_null, cfg["bc_h"])
            progress(cfg, int(np.count_nonzero(k_used)), len(prep.pairs))
