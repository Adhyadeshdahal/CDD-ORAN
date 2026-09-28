"""E6-P xApps (power / protected-slice extension; config ``cfg.e6p`` = ``config.E6PConfig``, default OFF).

Same interface as the v1 xApps (``xapps.XApp``): they act only on DELIVERED KPM reports (the E6-P additions to the
``fast`` report, ``ric.KPM._make_p``) and on the APPLIED configuration, emit ``{"xapp", "ver", "knob", "cur", "prop",
"t"}`` requests through ``XApp.req`` and inherit the per-seed implementation variant (threshold scale, blocked-action
behaviour, cadence phase). The env appends them AFTER the base mix (``cfg.e6p.xapps``), so the base xApps keep their
indices and RNG draws. The threshold scale multiplies only the dimensionless thresholds (load and violation shares), not
the dB thresholds. Every threshold / step is a PROVISIONAL ``E6PConfig`` field.

  * ``ESPico`` (role name "ES"): the v1 ``ES`` xApp unchanged for macro carriers, plus a pico-sleep part (TS 28.541
    DESManagementFunction roles, v1 ES constants): the pico (original cell) sleeps when its load < u_off and its
    candidate macro (strongest gain at the pico site) has full-band load < u_mac for 60 s; it wakes when the candidate's
    active-capacity util > u_on. Swapped in for v1 ``ES`` by the env when ``cfg.e6p`` is enabled and ``es_pico``.
  * ``PowerES``: lowers the Tx-power offset of a lightly loaded macro (full-band load < pes_u_low for pes_hold_s) by
    pes_step_db and restores it toward nominal when the ACTIVE-capacity util exceeds pes_u_high. Never above 0 dB.
  * ``Coverage``: raises the Tx-power offset where the cell's edge-SINR percentile is weak both in absolute terms and
    relative to its neighbours' (CCO-style), or where its protected-UE floors fail; releases a positive offset once the
    edge is comfortable or no longer below its neighbours'.
  * ``SliceGuarantee``: raises the protected work-conserving min PRB share where protected per-UE floors are violated,
    and lowers it after a slack spell (floors met and the min share mostly unused).
"""
from __future__ import annotations

import warnings

import numpy as np

from .ric import knob_get
from .xapps import ES, XApp


def _nanmean(stack):
    a = np.array(stack, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN column -> NaN (no data)
        return np.nanmean(a, 0)


class _PXApp(XApp):
    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.P = env.cfg.e6p
        self.hist = []

    def observe(self, reps):
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "fast":
                self.hist = (self.hist + [rep])[-int(max(self.cadence, 1)):]


class ESPico(ES):
    """v1 ES (macro capacity carriers, unchanged) + pico sleep / wake; see the module docstring."""
    name = "ES"

    def __init__(self, env, idx):
        super().__init__(env, idx)
        lay = self.p.lay
        self.picos = [c for c in range(lay.n_cells) if not lay.is_macro[c]]
        g = self.p.gm.lookup(lay.cell_pos[self.picos])                  # (picos, cells) outdoor gain at the pico site
        mac = np.nonzero(lay.is_macro)[0]
        self.cand = {pc: int(mac[np.argmax(g[i, mac])]) for i, pc in enumerate(self.picos)}
        self.pico_low_since = {pc: None for pc in self.picos}

    def propose(self, now):
        out = super().propose(now)                                      # v1 macro part (also updates self.u)
        if self.u is None:
            return out
        full = self.u * self.p.n_car / self.p.n_trx
        for pc in self.picos:
            m = self.cand[pc]
            if self.p.asleep[pc]:
                self.pico_low_since[pc] = None
                if self.u[m] > self.u_on:
                    r = self.req(("sleep", pc), 0.0, now)
                    if r:
                        out.append(r)
                continue
            if self.p.waking_until[pc] > now:
                continue
            if full[pc] < self.u_off and full[m] < self.u_mac:
                self.pico_low_since[pc] = self.pico_low_since[pc] if self.pico_low_since[pc] is not None else now
                if now - self.pico_low_since[pc] >= 60:
                    r = self.req(("sleep", pc), 1.0, now)
                    if r:
                        out.append(r)
            else:
                self.pico_low_since[pc] = None
        return out


class PowerES(_PXApp):
    name = "PowerES"

    def __init__(self, env, idx):
        self.cadence = env.cfg.e6p.pes_cadence_s
        super().__init__(env, idx)
        P = self.P
        self.u_low, self.u_high = P.pes_u_low * self.scale, P.pes_u_high * self.scale
        self.cells = [c for c in range(self.p.nc) if self.p.ptx_cells[c]]
        self.low_since = {c: None for c in self.cells}
        self.u = None

    def propose(self, now):
        rep = self.reports.get("fast")
        if rep is None:
            return []
        self.u = rep["prb_util"] if self.u is None else 0.6 * self.u + 0.4 * rep["prb_util"]
        full = self.u * self.p.n_car / self.p.n_trx                     # load as a share of the full band
        lo = self.P.ptx_range_db[0]
        out = []
        for c in self.cells:
            if self.p.asleep[c]:
                continue
            k = ("ptx", c)
            cur = knob_get(self.p, k)
            if full[c] < self.u_low:
                self.low_since[c] = self.low_since[c] if self.low_since[c] is not None else now
                if now - self.low_since[c] >= self.P.pes_hold_s and cur > lo + 1e-9:
                    out.append(self.req(k, max(lo, cur - self.P.pes_step_db), now))
            else:
                self.low_since[c] = None
                if self.u[c] > self.u_high and cur < -1e-9:                   # ACTIVE-capacity util (spec 4.2)
                    out.append(self.req(k, min(0.0, cur + self.P.pes_step_db), now))
        return [r for r in out if r]


class Coverage(_PXApp):
    name = "Coverage"

    def __init__(self, env, idx):
        self.cadence = env.cfg.e6p.cov_cadence_s
        super().__init__(env, idx)
        self.floor_frac = self.P.cov_floor_frac * self.scale
        self.cells = [c for c in range(self.p.nc) if self.p.ptx_cells[c]]

    def propose(self, now):
        if not self.hist:
            return []
        P = self.P
        edge = _nanmean([h["edge_sinr_p"] for h in self.hist])
        below = _nanmean([h["prot_below_frac"] for h in self.hist]) if "prot_below_frac" in self.hist[-1] else \
            np.full(self.p.nc, np.nan)
        hi = P.ptx_range_db[1]
        out = []
        for c in self.cells:
            if self.p.asleep[c] or np.isnan(edge[c]):
                continue
            k = ("ptx", c)
            cur = knob_get(self.p, k)
            nb = [n for n in self.p.lay.neighbours[c] if not self.p.asleep[n] and not np.isnan(edge[n])]
            nb_edge = float(np.mean(edge[nb])) if nb else np.inf
            weak = edge[c] < P.cov_sinr_low_db and edge[c] < nb_edge - P.cov_margin_db
            fail = weak or (not np.isnan(below[c]) and below[c] > self.floor_frac)
            if fail and cur < hi - 1e-9:
                out.append(self.req(k, min(hi, cur + P.cov_step_db), now))
            elif not fail and (edge[c] > P.cov_sinr_ok_db or edge[c] >= nb_edge) and cur > 1e-9:
                out.append(self.req(k, max(0.0, cur - P.cov_step_db), now))
        return [r for r in out if r]


class SliceGuarantee(_PXApp):
    name = "SliceGuarantee"

    def __init__(self, env, idx):
        self.cadence = env.cfg.e6p.sg_cadence_s
        super().__init__(env, idx)
        P = self.P
        self.hi, self.lo = P.sg_viol_hi * self.scale, P.sg_viol_lo * self.scale
        self.good_since = {}

    def propose(self, now):
        if not self.hist:
            return []
        P = self.P
        below = _nanmean([h["prot_below_frac"] for h in self.hist])
        used = np.mean([h["prot_used_share"] for h in self.hist], 0)
        mlo, mhi = P.prot_min_range
        out = []
        for c in range(self.p.nc):
            k = ("prot_min", c)
            cur = knob_get(self.p, k)
            b = below[c]
            if not np.isnan(b) and b > self.hi:
                self.good_since.pop(c, None)
                if cur < mhi - 1e-9:
                    out.append(self.req(k, min(mhi, cur + P.sg_step), now))
            elif cur > mlo + 1e-9 and (np.isnan(b) or b < self.lo) and used[c] < P.sg_slack * cur:
                self.good_since.setdefault(c, now)
                if now - self.good_since[c] >= P.sg_hold_s:
                    out.append(self.req(k, max(mlo, cur - P.sg_step), now))
                    self.good_since[c] = now
            else:
                self.good_since.pop(c, None)
        return [r for r in out if r]


P_XAPPS = {"PowerES": PowerES, "Coverage": Coverage, "SliceGuarantee": SliceGuarantee}
_NEEDS = {"PowerES": "ptx_on", "Coverage": "ptx_on", "SliceGuarantee": "prot_on"}


def build_p_xapps(env, names, start_idx):
    """Instantiate the E6-P xApps named in ``cfg.e6p.xapps`` (indices continue after the base mix)."""
    P = env.cfg.e6p
    out = []
    for j, nm in enumerate(names):
        if nm not in P_XAPPS:
            raise ValueError(f"unknown E6-P xApp {nm!r} (known: {sorted(P_XAPPS)})")
        if not getattr(P, _NEEDS[nm]):
            raise ValueError(f"E6-P xApp {nm} needs e6p.{_NEEDS[nm]}=True")
        if any(x.name == nm for x in env.xapps) or nm in names[:j]:
            raise ValueError(f"E6-P xApp {nm} deployed twice")
        out.append(P_XAPPS[nm](env, start_idx + j))
    return out


__all__ = ["P_XAPPS", "Coverage", "ESPico", "PowerES", "SliceGuarantee", "build_p_xapps"]
