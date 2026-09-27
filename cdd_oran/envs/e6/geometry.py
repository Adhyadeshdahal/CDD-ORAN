"""E6 geometry and large-scale radio: 7-site hex layout with wrap-around, macro sectors + co-channel picos,
TR 38.901 UMa / UMi-street-canyon path loss with spatially consistent LOS and correlated shadowing, and 3GPP
sector antenna patterns. Everything is precomputed per seed into per-cell gain maps on a GRID_M grid, so a tick
only gathers ``gain[cell, iy, ix]``.

Wrap-around: the 7-site cluster tiles the plane with translation generator (i, j) = (2, 1) on the hex lattice
(i^2 + ij + j^2 = 7) and its 60-degree rotations; distances use the nearest image of each site.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from . import config as C

SQ3 = np.sqrt(3.0)


def _lattice(i, j, d):
    return np.stack([d * (np.asarray(i) + np.asarray(j) / 2.0), d * np.asarray(j) * SQ3 / 2.0], -1)


SITES_IJ = np.array([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)])
WRAP_IJ = np.array([(0, 0), (2, 1), (-1, 3), (-3, 2), (-2, -1), (1, -3), (3, -2)])


class Layout:
    def __init__(self, cfg: C.E6Config, rng: np.random.Generator):
        d = C.ISD_M
        self.sites = _lattice(SITES_IJ[:, 0], SITES_IJ[:, 1], d)                 # (7, 2)
        self.shifts = _lattice(WRAP_IJ[:, 0], WRAP_IJ[:, 1], d)                  # (7, 2) incl. (0, 0)
        # macro sectors: 3 per site, boresight 30/150/270 deg
        self.cell_pos, self.cell_bore, self.cell_site, self.is_macro = [], [], [], []
        for s in range(7):
            for b in (30.0, 150.0, 270.0):
                self.cell_pos.append(self.sites[s])
                self.cell_bore.append(b)
                self.cell_site.append(s)
                self.is_macro.append(True)
        # hotspot centres (hidden): 150-250 m from a random macro site, random bearing
        hs = []
        for _ in range(cfg.n_hotspots):
            s = rng.integers(7)
            r, a = rng.uniform(150, 250), rng.uniform(0, 2 * np.pi)
            hs.append(self.sites[s] + r * np.array([np.cos(a), np.sin(a)]))
        self.hotspots = self.wrap(np.array(hs))
        for k in range(cfg.n_pico):   # one pico per hotspot cluster (co-channel CRE target)
            self.cell_pos.append(self.hotspots[k % len(self.hotspots)])
            self.cell_bore.append(0.0)
            self.cell_site.append(7 + k)
            self.is_macro.append(False)
        self.cell_pos = np.array(self.cell_pos)
        self.cell_bore = np.array(self.cell_bore)
        self.cell_site = np.array(self.cell_site)
        self.is_macro = np.array(self.is_macro)
        self.n_cells = len(self.cell_pos)
        self.neighbours = self._neighbours()

    # ------------------------------------------------------------------------------------------ wrap-around
    def _cands(self):
        k = np.arange(-2, 3)
        a, b = np.meshgrid(k, k)
        v1, v2 = self.shifts[1], self.shifts[2]
        return a.ravel()[:, None] * v1 + b.ravel()[:, None] * v2                  # (25, 2)

    def wrap(self, p):
        """Map points into the fundamental 7-site cluster (nearest-site hexagon of the origin cluster)."""
        p = np.atleast_2d(p)
        q = p[:, None, :] - self._cands()[None, :, :]                             # (n, 25, 2)
        dn = np.linalg.norm(q[:, :, None, :] - self.sites[None, None], axis=-1).min(-1)
        return q[np.arange(len(p)), dn.argmin(1)]

    def wrap_near(self, p):
        """Cheap re-wrap for points that moved at most a few metres out of the cluster (7 cluster shifts only)."""
        q = p[:, None, :] + self.shifts[None, :, :]                               # (n, 7, 2)
        dn = ((q[:, :, None, :] - self.sites[None, None]) ** 2).sum(-1).min(-1)
        return q[np.arange(len(p)), dn.argmin(1)]

    def link_geometry(self, pts):
        """Nearest-image 2D distance (n, cells) and azimuth from each cell (deg)."""
        rel = pts[:, None, None, :] - (self.cell_pos[None, :, None, :] + self.shifts[None, None, :, :])
        dist = np.linalg.norm(rel, axis=-1)                                          # (n, cells, 7)
        k = dist.argmin(-1)
        r = np.take_along_axis(rel, k[..., None, None], axis=2)[:, :, 0, :]
        d2 = np.maximum(np.take_along_axis(dist, k[..., None], axis=2)[..., 0], 10.0)
        az = np.degrees(np.arctan2(r[..., 1], r[..., 0]))
        return d2, az

    def _neighbours(self, k=6):
        d2, _ = self.link_geometry(self.cell_pos + 1e-3)
        nb = []
        for c in range(self.n_cells):
            order = [j for j in np.argsort(d2[c]) if j != c and self.cell_site[j] != self.cell_site[c]]
            same = [j for j in range(self.n_cells) if j != c and self.cell_site[j] == self.cell_site[c]]
            nb.append(sorted(set(same + order[:k])))
        return nb


# ---------------------------------------------------------------------------------------------- path loss
def pl_uma(d2, fc=C.FREQ_GHZ, hbs=C.MACRO_H, hut=C.UE_H):
    d3 = np.sqrt(d2 ** 2 + (hbs - hut) ** 2)
    dbp = 4 * (hbs - 1) * (hut - 1 if hut > 1 else 0.5) * fc * 1e9 / 3e8
    dbp = max(dbp, 4 * (hbs - 1) * 0.5 * fc * 1e9 / 3e8)
    los = np.where(d2 <= dbp, 28 + 22 * np.log10(d3) + 20 * np.log10(fc),
                   28 + 40 * np.log10(d3) + 20 * np.log10(fc) - 9 * np.log10(dbp ** 2 + (hbs - hut) ** 2))
    nlos = np.maximum(los, 13.54 + 39.08 * np.log10(d3) + 20 * np.log10(fc) - 0.6 * (hut - 1.5))
    p = np.where(d2 <= 18, 1.0, 18 / d2 + np.exp(-d2 / 63) * (1 - 18 / d2))
    return los, nlos, p


def pl_umi(d2, fc=C.FREQ_GHZ, hbs=C.PICO_H, hut=C.UE_H):
    d3 = np.sqrt(d2 ** 2 + (hbs - hut) ** 2)
    dbp = 4 * (hbs - 1) * 0.5 * fc * 1e9 / 3e8
    los = np.where(d2 <= dbp, 32.4 + 21 * np.log10(d3) + 20 * np.log10(fc),
                   32.4 + 40 * np.log10(d3) + 20 * np.log10(fc) - 9.5 * np.log10(dbp ** 2 + (hbs - hut) ** 2))
    nlos = np.maximum(los, 22.4 + 35.3 * np.log10(d3) + 21.3 * np.log10(fc) - 0.3 * (hut - 1.5))
    p = np.where(d2 <= 18, 1.0, 18 / d2 + np.exp(-d2 / 36) * (1 - 18 / d2))
    return los, nlos, p


def antenna_db(az, d2, bore, macro, hbs=C.MACRO_H, hut=C.UE_H):
    if not macro:
        return np.full(np.shape(az), C.PICO_GAIN_DBI)
    phi = (az - bore + 180.0) % 360.0 - 180.0
    ah = -np.minimum(12 * (phi / C.HPBW_DEG) ** 2, C.AM_DB)
    if not C.VERTICAL:
        return C.MACRO_GAIN_DBI + ah
    theta = np.degrees(np.arctan2(hbs - hut, d2))
    av = -np.minimum(12 * ((theta - C.TILT_DEG) / C.VBW_DEG) ** 2, C.SLAV_DB)
    return C.MACRO_GAIN_DBI - np.minimum(-(ah + av), C.AM_DB)


def _field(shape, rng, sigma_px):
    z = gaussian_filter(rng.normal(size=shape), sigma_px, mode="wrap")
    return z / (z.std() + 1e-12)


class GainMaps:
    """Per-cell large-scale gain (dB, incl. Tx power, antenna, path loss, shadowing) on a grid over the
    fundamental cluster's bounding box."""

    def __init__(self, lay: Layout, cfg: C.E6Config, rng: np.random.Generator):
        ext = C.ISD_M + C.ISD_M / SQ3 + 2 * C.GRID_M
        self.x0 = -ext
        n = int(np.ceil(2 * ext / C.GRID_M)) + 1
        self.n = n
        xs = self.x0 + C.GRID_M * np.arange(n)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel()], -1)
        d2, az = lay.link_geometry(pts)
        sig_px = C.SF_DECORR_M / C.GRID_M / 2.0
        common = _field((n, n), rng, sig_px).ravel()
        site_fields, los_fields = {}, {}
        gain = np.empty((lay.n_cells, n * n), np.float32)
        for c in range(lay.n_cells):
            s = int(lay.cell_site[c])
            if s not in site_fields:
                site_fields[s] = _field((n, n), rng, sig_px).ravel()
                los_fields[s] = norm.cdf(_field((n, n), rng, sig_px).ravel())      # spatially consistent U(0,1)
            z = np.sqrt(0.5) * common + np.sqrt(0.5) * site_fields[s]
            if lay.is_macro[c]:
                los, nlos, p = pl_uma(d2[:, c])
                sl, sn = C.SF_SIGMA_MACRO
                ptx = C.MACRO_PTX_DBM
            else:
                los, nlos, p = pl_umi(d2[:, c])
                sl, sn = C.SF_SIGMA_PICO
                ptx = C.PICO_PTX_DBM
            is_los = los_fields[s] < p
            pl = np.where(is_los, los, nlos) + cfg.pl_offset_db
            sf = z * np.where(is_los, sl, sn)
            gain[c] = ptx + antenna_db(az[:, c], d2[:, c], lay.cell_bore[c], bool(lay.is_macro[c])) - pl - sf
        self.gain = gain.reshape(lay.n_cells, n, n)

    def lookup(self, pts):
        """Nearest-grid gains (n_pts, cells) in dBm for points already wrapped into the cluster."""
        ix = np.clip(np.rint((pts[:, 0] - self.x0) / C.GRID_M).astype(int), 0, self.n - 1)
        iy = np.clip(np.rint((pts[:, 1] - self.x0) / C.GRID_M).astype(int), 0, self.n - 1)
        return self.gain[:, iy, ix].T
