"""Layout, large-scale gains and AR(1) Rayleigh fading (Table I "Channel model").

* Sites: omni cells on a hexagonal lattice with the given ISD, filled in ring order (centre, then ring-1 sites at
  0, 60, 120, ... deg), so 4 cells = a compact cluster. No wrap-around (paper silent).
* UEs: uniform over the union of the cells' hexagons, at least ``min_dist_m`` from every site.
* Path loss: TR 36.814 V9.0.0 Table B.1.2.1-1 UMa (= ITU-R M.2135 UMa) with its LOS probability, or the TR 36.814
  Table A.2.1.1-2 macro model 128.1 + 37.6 log10(R km) (``pathloss="36814_macro"``).
* Shadowing: log-normal, ``shadow_db`` std, site correlation ``shadow_site_corr`` (TR 36.814 Table A.2.1.1-2).
* Fading: h(t) = rho h(t-1) + sqrt(1 - rho^2) w(t), w ~ CN(0, 1) (Baddour & Beaulieu 2005, the paper's [39]).
"""
from __future__ import annotations

import numpy as np

SQ3 = np.sqrt(3.0)


def hex_sites(n: int, isd: float) -> np.ndarray:
    """First ``n`` sites of a hexagonal lattice in ring order (centre, ring 1 counter-clockwise from 0 deg, ...)."""
    pts = [(0.0, 0.0)]
    ring = 1
    while len(pts) < n:
        # axial walk around ring `ring`
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        q, r = ring, 0                                      # start at 0 deg
        cand = []
        for d in range(6):
            for _ in range(ring):
                cand.append((q, r))
                dq, dr = dirs[(d + 2) % 6]
                q, r = q + dq, r + dr
        for q, r in cand:
            pts.append((isd * (q + 0.5 * r), isd * (SQ3 / 2.0) * r))
        ring += 1
    return np.asarray(pts[:n], float)


def _in_hex(dx, dy, R):
    """Point in the pointy-top hexagon of circumradius R centred at 0 (the Voronoi cell of a lattice whose nearest
    neighbours lie along the x axis)."""
    ax, ay = np.abs(dx), np.abs(dy)
    return (ax <= SQ3 / 2 * R) & (ay <= R - ax / SQ3)


def drop_ues(rng, sites: np.ndarray, n: int, isd: float, min_dist: float) -> np.ndarray:
    R = isd / SQ3                                           # hexagon circumradius
    out = np.empty((n, 2))
    i = 0
    while i < n:
        s = sites[rng.integers(len(sites))]
        dx, dy = rng.uniform(-R, R), rng.uniform(-R, R)
        if not _in_hex(dx, dy, R):
            continue
        p = s + (dx, dy)
        if np.min(np.hypot(*(sites - p).T)) < min_dist:
            continue
        out[i] = p
        i += 1
    return out


def pl_36814_uma(d, fc, hbs, hut, W, h, los):
    """TR 36.814 Table B.1.2.1-1 UMa path loss [dB]; d = 2-D distance [m], fc [GHz]."""
    d = np.maximum(d, 10.0)
    hb1, hu1 = hbs - 1.0, hut - 1.0
    dbp = 4 * hb1 * hu1 * fc * 1e9 / 3e8
    pl_los = np.where(d < dbp, 22.0 * np.log10(d) + 28.0 + 20.0 * np.log10(fc),
                      40.0 * np.log10(d) + 7.8 - 18.0 * np.log10(hb1) - 18.0 * np.log10(hu1) + 2.0 * np.log10(fc))
    pl_nlos = (161.04 - 7.1 * np.log10(W) + 7.5 * np.log10(h) - (24.37 - 3.7 * (h / hbs) ** 2) * np.log10(hbs)
               + (43.42 - 3.1 * np.log10(hbs)) * (np.log10(d) - 3.0) + 20.0 * np.log10(fc)
               - (3.2 * np.log10(11.75 * hut) ** 2 - 4.97))
    return np.where(los, pl_los, pl_nlos)


def p_los_36814_uma(d):
    """TR 36.814 Table B.1.2.1-2 UMa LOS probability."""
    d = np.maximum(d, 1e-9)
    return np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)


def pl_36814_macro(d):
    """TR 36.814 Table A.2.1.1-2 macro model: 128.1 + 37.6 log10(R), R in km (2 GHz)."""
    return 128.1 + 37.6 * np.log10(np.maximum(d, 1.0) / 1000.0)


def large_scale_gain(cfg, rng, sites, ues):
    """-> (G_mean (B, U) linear power gain incl. shadowing and antenna gain, dist (B, U), los (B, U))."""
    d = np.hypot(sites[:, None, 0] - ues[None, :, 0], sites[:, None, 1] - ues[None, :, 1])
    B, U = d.shape
    if cfg.pathloss == "36814_uma":
        los = rng.random((B, U)) < p_los_36814_uma(d)
        pl = pl_36814_uma(d, cfg.fc_ghz, cfg.h_bs_m, cfg.h_ut_m, cfg.street_w_m, cfg.bldg_h_m, los)
    elif cfg.pathloss == "36814_macro":
        los = np.zeros((B, U), bool)
        pl = pl_36814_macro(d)
    else:
        raise ValueError(f"unknown pathloss {cfg.pathloss!r}")
    c = cfg.shadow_site_corr
    sf = cfg.shadow_db * (np.sqrt(c) * rng.standard_normal(U)[None, :] + np.sqrt(1 - c) * rng.standard_normal((B, U)))
    g_db = -pl - sf + cfg.antenna_gain_db
    return 10.0 ** (g_db / 10.0), d, los


def cn(rng, shape):
    """CN(0, 1) samples."""
    z = rng.standard_normal((2,) + tuple(shape))
    return (z[0] + 1j * z[1]) * np.sqrt(0.5)
