"""E6 adapter: region map (cell -> site: 7 macro sites + 3 picos), knob -> region, arbiter factory.

Every E6 knob is keyed by its own cell first: ("cio", s, n) belongs to the source cell s, (typ, c) to cell c.
"""
from __future__ import annotations

import numpy as np


def region_map(env) -> np.ndarray:
    """cell -> region id (the site of the cell; picos are their own regions)."""
    return np.asarray(env.plant.lay.cell_site)


def regions(site) -> list[int]:
    return sorted({int(x) for x in site})


def knob_region(site, knob) -> int:
    return int(site[knob[1]])


def make_arbiter(env, world_model, lam_e, w_ll, **kw):
    """WG3Arbiter wired to ``env``: region map, RNG seed = cfg.seed, decisions start at the end of warm-up."""
    from ..arbiter import WG3Arbiter
    kw.setdefault("seed", env.cfg.seed)
    kw.setdefault("start_s", float(env.cfg.warmup_s))
    return WG3Arbiter(world_model, region_map(env), lam_e, w_ll, **kw)
