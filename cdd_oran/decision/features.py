"""Per-cell, cell-exchangeable feature panel from E6 traces (``e6-trace/1``) for template discovery.

One row per (episode, step, cell) on a fixed step grid (``step_s``, default 10 s). Step k covers the plant window
(t_k, t_k + step_s], t_k = t_min + k * step_s (trace second t = the ticks that ran just before the decisions of t).

Columns (``Panel.kind``):
  knob_own    configuration of the cell IN FORCE AT THE START of the window (``Trace.config_at(t_k + 1)``, the
              pre-action configuration of the window's first second): own_hys (dB), own_ttt (ms), own_ll_ratio
              (M4 only), own_carrier (active share of the band: macro n_car / MACRO_NTRX, pico 1.0), own_sleep
              (pico 0/1, macro 0), cio_out_mean / cio_out_max (CIO the cell sets toward its neighbours, dB).
  knob_nbr    neighbours' knobs acting on this cell: cio_in_mean (mean CIO neighbours set TOWARD this cell),
              nbr_carrier (mean own_carrier of the neighbours).
  state_nbr   nbr_util_lag: mean neighbour PRB utilisation over the previous window (a context state, not a knob).
  kpi         delivered-KPM KPI families over the window: prb_util, ll_delay_p95 (s), embb_thp_p5 (b/s), rlf,
              too_late, too_early (= too_early + wrong_cell + ping-pong, MRO's "early" count), all per second;
              energy_w (W). Reports finer than the step are averaged (nanmean over the window's reports); coarser
              reports (mob 30 s, energy 60 s) give the value of the report whose window covers the step, so those
              families repeat over consecutive steps: thin at >= their granularity (``KPI_GRAN_S``).
  kpi_lag     <family>_lag: the same family one report period earlier (lag = ceil(gran / step) steps).
  context     is_macro (cell class; constant per cell).
Neighbours = cells linked by any CIO knob (either direction), derived from ``meta["knobs"]``; region = ``meta
["cell_region"]`` (= cell site). Only runtime-observable arrays are read (``Trace.features()``): the ``lab_*``
labels never enter the panel.

Ownership: ``knob_ownership(trace)`` counts, from the logged requests, which knob families each xApp writes
(requests offered / changes applied). Nothing is hardcoded.

Degenerate columns (constant, or almost all one value) are DROPPED and recorded (``Panel.dropped``): MSCR's
equal-count binning breaks ties by row order, so a constant or tied column would bin by time/cell order (known
MSCR BUG-2) and a constant target inflates S* by rounding (BUG-1). ``discovery`` re-checks on the rows it uses.

E6-P panel (``build_panel_p``; scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md): the E6 panel plus
  knob_own    own_ptx (Tx-power offset dB; cells without a ptx knob, i.e. picos under ptx_scope "macro", get 0 = their
              fixed offset), own_prot_min (protected min PRB share);
  knob_nbr    nbr_sleep, nbr_ptx, nbr_prot_min (mean over the CIO neighbours, like nbr_carrier);
  state_nbr   nbr_act_ue_lag: mean neighbour active UEs (fast report act_ue summed over slices) over the previous
              window;
  kpi         prot_viol (protected violated UEs per second = fast report prot_below_frac * prot_eval, 0 when no
              protected UE was evaluated), prot_act_ue (fast), edge_sinr_p (fast, dB); each with a <family>_lag.
Obs-only: built from ``Trace.features()`` (delivered reports, logged requests, pre-action configuration); no lab_*
array, no plant attribute. The 1 % minority filter judges own_sleep on PICO rows only (``drop_degenerate_p``): a
macro never sleeps, so on all rows a pico sleep of any length is a < 1 % minority almost by construction (21 of 24
cells are macros); the column is kept iff it is not degenerate on the pico rows (recorded in ``Panel.notes``).
NOTE: ``discovery.discover_template`` re-checks degeneracy per target on ALL its thinned rows (unchanged E6 rule).
"""
from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np

from cdd_oran.envs.e6.config import MACRO_NTRX

from .trace import LABEL_PREFIX, Trace

KNOB_FAMILIES = ("cio", "hys", "ttt", "ll_ratio", "carrier", "sleep", "ptx", "prot_min")
KPI_FAMILIES = ("prb_util", "ll_delay_p95", "embb_thp_p5", "rlf", "too_late", "too_early", "energy_w")
KPI_FAMILIES_PX = ("prot_viol", "prot_act_ue", "edge_sinr_p")          # E6-P additions (build_panel_p)
KPI_FAMILIES_P = KPI_FAMILIES + KPI_FAMILIES_PX
KPI_GRAN_S = {"prb_util": 1, "ll_delay_p95": 1, "embb_thp_p5": 5, "rlf": 30, "too_late": 30, "too_early": 30,
              "energy_w": 60, "prot_viol": 1, "prot_act_ue": 1, "edge_sinr_p": 1}
_KPI_SRC = {"prb_util": "fast", "ll_delay_p95": "fast", "embb_thp_p5": "thp", "rlf": "mob", "too_late": "mob",
            "too_early": "mob", "energy_w": "energy"}
# column -> (kind, knob/kpi family, scope)
KNOB_COLUMNS = {"own_hys": ("knob_own", "hys"), "own_ttt": ("knob_own", "ttt"),
                "own_ll_ratio": ("knob_own", "ll_ratio"), "own_carrier": ("knob_own", "carrier"),
                "own_sleep": ("knob_own", "sleep"), "cio_out_mean": ("knob_own", "cio"),
                "cio_out_max": ("knob_own", "cio"), "cio_in_mean": ("knob_nbr", "cio"),
                "nbr_carrier": ("knob_nbr", "carrier")}
KNOB_COLUMNS_P = {"own_ptx": ("knob_own", "ptx"), "own_prot_min": ("knob_own", "prot_min"),
                  "nbr_sleep": ("knob_nbr", "sleep"), "nbr_ptx": ("knob_nbr", "ptx"),
                  "nbr_prot_min": ("knob_nbr", "prot_min")}
PICO_JUDGED = ("own_sleep",)          # columns whose minority filter is judged on pico rows (drop_degenerate_p)
KINDS = ("knob_own", "knob_nbr", "state_nbr", "kpi", "kpi_lag", "context")


@dataclasses.dataclass
class Panel:
    """Pooled per-cell panel. ``data[col]`` is (n,) float64; rows indexed by ``episode``, ``t`` (step start), ``cell``."""
    data: dict
    kind: dict                 # column -> one of KINDS
    family: dict               # column -> knob family (knob_*), KPI family (kpi, kpi_lag), or the column name
    episode: np.ndarray
    t: np.ndarray
    cell: np.ndarray
    step_s: int
    cell_region: np.ndarray    # (C,) region (site) of each cell
    neighbours: list           # neighbours[c] = sorted cell ids
    ownership: dict            # xapp -> {knob family: {"req": n, "applied": n}}
    xapps: list                # xApps present in the episodes
    dropped: dict = dataclasses.field(default_factory=dict)   # column -> reason
    notes: dict = dataclasses.field(default_factory=dict)     # column -> note (e.g. judged on pico rows)

    @property
    def n(self) -> int:
        return len(self.t)

    def columns(self, *kinds) -> list:
        return [c for c in self.data if self.kind[c] in kinds]

    def scope(self, col) -> str:
        return "neighbour" if self.kind[col] in ("knob_nbr", "state_nbr") else "own"

    def subset(self, rows) -> Panel:
        rows = np.asarray(rows)
        return dataclasses.replace(self, data={k: v[rows] for k, v in self.data.items()}, episode=self.episode[rows],
                                   t=self.t[rows], cell=self.cell[rows], dropped=dict(self.dropped))

    def drop_degenerate(self, min_minor_frac: float = 0.01, kinds=None) -> Panel:
        """Drop columns (of ``kinds``, default all) that are degenerate on this panel's rows; record why."""
        kinds = kinds or KINDS
        bad = degenerate_columns({c: v for c, v in self.data.items() if self.kind[c] in kinds}, min_minor_frac)
        if not bad:
            return self
        out = dataclasses.replace(self, data={k: v for k, v in self.data.items() if k not in bad},
                                  kind={k: v for k, v in self.kind.items() if k not in bad},
                                  family={k: v for k, v in self.family.items() if k not in bad},
                                  dropped={**self.dropped, **bad})
        return out


def degenerate_columns(cols: dict, min_minor_frac: float = 0.01) -> dict:
    """{column: reason} for columns that MSCR cannot bin validly on these rows: all-NaN, constant (incl. float
    rounding: spread <= 1e-9 * (1 + |mean|)), or the modal value covering more than 1 - ``min_minor_frac`` of
    the finite rows (a near-constant knob carries no testable variation and its ties would dominate the bins)."""
    bad = {}
    for c, v in cols.items():
        v = np.asarray(v, float)
        f = v[np.isfinite(v)]
        if len(f) == 0:
            bad[c] = "all_nan"
            continue
        if float(np.ptp(f)) <= 1e-9 * (1.0 + abs(float(np.mean(f)))):
            bad[c] = "constant"
            continue
        _, counts = np.unique(f, return_counts=True)
        if counts.max() > (1.0 - min_minor_frac) * len(f):
            bad[c] = f"near_constant(modal share {counts.max() / len(f):.3f})"
    return bad


# ------------------------------------------------------------------------------------------------ helpers
def neighbours_from_knobs(knobs, n_cells: int) -> list:
    """Cells linked by a CIO knob in either direction (the E6 knob list carries forward + reverse pairs)."""
    nb = [set() for _ in range(n_cells)]
    for k in knobs:
        if k[0] == "cio":
            s, n = int(k[1]), int(k[2])
            nb[s].add(n)
            nb[n].add(s)
    return [sorted(x) for x in nb]


def knob_ownership(trace: Trace) -> dict:
    """xapp -> {knob family: {"req": requests offered, "applied": requests that changed the knob}} from the log."""
    a, meta = trace.arrays, trace.meta
    ok = meta["out_codes"].index("ok")
    own: dict = {x: {} for x in meta["xapps"]}
    for x, k, o in zip(a["rq_xapp"], a["rq_knob"], a["rq_out"], strict=True):
        d = own[meta["xapps"][int(x)]].setdefault(meta["knobs"][int(k)][0], {"req": 0, "applied": 0})
        d["req"] += 1
        d["applied"] += int(o == ok)
    return own


def merge_ownership(owns) -> dict:
    out: dict = {}
    for own in owns:
        for x, fams in own.items():
            dst = out.setdefault(x, {})
            for f, d in fams.items():
                e = dst.setdefault(f, {"req": 0, "applied": 0})
                e["req"] += d["req"]
                e["applied"] += d["applied"]
    return out


def writers(ownership: dict, family: str, min_req: int = 1) -> list:
    """xApps that requested changes of knob ``family`` at least ``min_req`` times."""
    return sorted(x for x, fams in ownership.items() if fams.get(family, {}).get("req", 0) >= min_req)


def _kpi_windows(a: dict, gran: str, tk: np.ndarray, step: int, n_cells: int, fields) -> dict:
    """Per step, per cell values of report ``fields`` (callables report-index -> (C,)) for windows (t_k, t_k+step]."""
    out = {name: np.full((len(tk), n_cells), np.nan) for name in fields}
    key = f"kpm_{gran}_t0"
    if key not in a or len(a[key]) == 0:
        return out
    t0, t1 = a[key], a[f"kpm_{gran}_t1"]
    for i, tt in enumerate(tk):
        inside = np.nonzero((t0 >= tt) & (t1 <= tt + step))[0]
        if len(inside) == 0:               # coarser report: the one whose window covers the step
            inside = np.nonzero((t0 <= tt) & (t1 >= tt + step))[0][:1]
        if len(inside) == 0:
            continue
        for name, fn in fields.items():
            vals = np.stack([fn(j) for j in inside])
            with warnings.catch_warnings():             # all-NaN cells (no LL / eMBB samples) stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                out[name][i] = np.nanmean(vals, 0)
    return out


# ------------------------------------------------------------------------------------------------ builder
def build_panel(trace: Trace, step_s: int = 10, t_min: int = 0, episode: int = 0,
                drop_degenerate: bool = True, min_minor_frac: float = 0.01) -> Panel:
    """Per-cell panel of one trace (see module docstring). Labels are never read."""
    a = trace.features()
    assert not any(k.startswith(LABEL_PREFIX) for k in a)
    meta = trace.meta
    knobs = [tuple(k) for k in meta["knobs"]]
    kidx = {k: i for i, k in enumerate(knobs)}
    region = np.asarray(meta["cell_region"], int)
    C = len(region)
    nb = neighbours_from_knobs(knobs, C)
    is_macro = np.array([("carrier", c) in kidx for c in range(C)])
    t_end = int(a["t"][-1])
    step = int(step_s)
    tk = np.arange(int(t_min), t_end - step + 1, step)
    K = len(tk)

    cfg = np.stack([trace.config_at(int(t) + 1) for t in tk]) if K else np.zeros((0, len(knobs)))

    def per_cell(fam, default):
        idx = [kidx.get((fam, c), -1) for c in range(C)]
        out = np.full((K, C), float(default))
        for c, i in enumerate(idx):
            if i >= 0:
                out[:, c] = cfg[:, i]
        return out, any(i >= 0 for i in idx)

    cols, have = {}, {}
    cols["own_hys"], have["own_hys"] = per_cell("hys", np.nan)
    cols["own_ttt"], have["own_ttt"] = per_cell("ttt", np.nan)
    cols["own_ll_ratio"], have["own_ll_ratio"] = per_cell("ll_ratio", np.nan)
    car, have["own_carrier"] = per_cell("carrier", MACRO_NTRX)
    cols["own_carrier"] = np.where(is_macro[None, :], car / MACRO_NTRX, 1.0)
    cols["own_sleep"], have["own_sleep"] = per_cell("sleep", 0.0)
    cio_out = [[kidx[("cio", c, n)] for n in range(C) if ("cio", c, n) in kidx] for c in range(C)]
    cio_in = [[kidx[("cio", s, c)] for s in range(C) if ("cio", s, c) in kidx] for c in range(C)]
    nan = np.full(K, np.nan)
    cols["cio_out_mean"] = np.stack([cfg[:, i].mean(1) if i else nan for i in cio_out], 1)
    cols["cio_out_max"] = np.stack([cfg[:, i].max(1) if i else nan for i in cio_out], 1)
    cols["cio_in_mean"] = np.stack([cfg[:, i].mean(1) if i else nan for i in cio_in], 1)
    have.update(cio_out_mean=True, cio_out_max=True, cio_in_mean=True)
    cols["nbr_carrier"] = np.stack([cols["own_carrier"][:, n].mean(1) if n else nan for n in nb], 1)
    have["nbr_carrier"] = have["own_carrier"]

    # KPI families from delivered reports
    fast = _kpi_windows(a, "fast", tk, step, C, {
        "prb_util": lambda j: a["kpm_fast_prb_util"][j],
        "ll_delay_p95": lambda j: a["kpm_fast_ll_delay_p95"][j]})
    thp = _kpi_windows(a, "thp", tk, step, C, {"embb_thp_p5": lambda j: a["kpm_thp_embb_thp_p5"][j]})

    def mob_rate(field_names):
        def fn(j):
            win = float(a["kpm_mob_t1"][j] - a["kpm_mob_t0"][j])
            v = sum(np.asarray(a[f"kpm_mob_{x}"][j], float) for x in field_names)
            return (v.sum(1) if v.ndim == 2 else v) / win
        return fn

    mob = _kpi_windows(a, "mob", tk, step, C, {"rlf": mob_rate(["rlf"]), "too_late": mob_rate(["too_late"]),
                                                 "too_early": mob_rate(["too_early", "wrong_cell", "pingpong"])})
    en = _kpi_windows(a, "energy", tk, step, C, {
        "energy_w": lambda j: np.asarray(a["kpm_energy_energy_j"][j], float)
        / float(a["kpm_energy_t1"][j] - a["kpm_energy_t0"][j])})
    kpis = {**fast, **thp, **mob, **en}

    lag_cols = {}
    for fam in KPI_FAMILIES:
        L = max(1, math.ceil(KPI_GRAN_S[fam] / step))
        lagged = np.full((K, C), np.nan)
        if K > L:
            lagged[L:] = kpis[fam][:-L]
        lag_cols[fam + "_lag"] = lagged
    util_lag = lag_cols["prb_util_lag"]
    nbr_util = np.stack([util_lag[:, n].mean(1) if n else nan for n in nb], 1)

    data, kind, family = {}, {}, {}

    def put(name, arr, k, fam):
        data[name] = np.asarray(arr, float).reshape(-1)       # row order: step-major, then cell
        kind[name], family[name] = k, fam

    dropped = {}
    for name, (k, fam) in KNOB_COLUMNS.items():
        if have.get(name, True):
            put(name, cols[name], k, fam)
        else:
            dropped[name] = "knob_absent_in_mix"
    put("nbr_util_lag", nbr_util, "state_nbr", "prb_util")
    for fam in KPI_FAMILIES:
        put(fam, kpis[fam], "kpi", fam)
    for name, v in lag_cols.items():
        put(name, v, "kpi_lag", name[:-4])
    put("is_macro", np.broadcast_to(is_macro, (K, C)).astype(float), "context", "is_macro")

    p = Panel(data=data, kind=kind, family=family, episode=np.full(K * C, int(episode)),
              t=np.repeat(tk, C).astype(np.int64), cell=np.tile(np.arange(C), K), step_s=step,
              cell_region=region, neighbours=nb, ownership=knob_ownership(trace), xapps=list(meta["xapps"]),
              dropped=dropped)
    return p.drop_degenerate(min_minor_frac) if drop_degenerate else p


def concat_panels(panels, drop_degenerate: bool = True, min_minor_frac: float = 0.01) -> Panel:
    """Pool panels of several episodes (same layout). Columns absent from any panel are dropped (recorded)."""
    panels = list(panels)
    if not panels:
        raise ValueError("no panels")
    p0 = panels[0]
    for p in panels[1:]:
        if p.step_s != p0.step_s or not np.array_equal(p.cell_region, p0.cell_region):
            raise ValueError("panels differ in step or layout")
    common = [c for c in p0.data if all(c in p.data for p in panels)]
    dropped = {}
    for p in panels:
        dropped.update(p.dropped)
        dropped.update({c: "absent_in_some_episode" for c in p.data if c not in common})
    for c in common:
        dropped.pop(c, None)
    out = Panel(data={c: np.concatenate([p.data[c] for p in panels]) for c in common},
                kind={c: p0.kind[c] for c in common}, family={c: p0.family[c] for c in common},
                episode=np.concatenate([p.episode for p in panels]), t=np.concatenate([p.t for p in panels]),
                cell=np.concatenate([p.cell for p in panels]), step_s=p0.step_s, cell_region=p0.cell_region,
                neighbours=p0.neighbours, ownership=merge_ownership(p.ownership for p in panels),
                xapps=sorted({x for p in panels for x in p.xapps}), dropped=dropped)
    return out.drop_degenerate(min_minor_frac) if drop_degenerate else out


def build_panels(traces, step_s: int = 10, t_min: int = 0, min_minor_frac: float = 0.01) -> Panel:
    """Pooled panel of several traces (episode id = position); degenerate columns judged on the pooled rows."""
    return concat_panels([build_panel(tr, step_s, t_min, i, drop_degenerate=False) for i, tr in enumerate(traces)],
                         min_minor_frac=min_minor_frac)


# ------------------------------------------------------------------------------------------------ E6-P panel
def drop_degenerate_p(panel: Panel, min_minor_frac: float = 0.01) -> Panel:
    """``Panel.drop_degenerate`` with the E6-P sleep rule: a ``PICO_JUDGED`` column flagged on all rows is re-judged on
    the pico rows (is_macro == 0) and kept (``notes``) when it is not degenerate there."""
    bad = degenerate_columns(panel.data, min_minor_frac)
    notes = dict(panel.notes)
    if "is_macro" in panel.data:
        pico = panel.data["is_macro"] < 0.5
        for c in PICO_JUDGED:
            if c in bad and pico.any():
                why = degenerate_columns({c: panel.data[c][pico]}, min_minor_frac)
                if not why:
                    notes[c] = f"kept: judged on {int(pico.sum())} pico rows (all rows: {bad.pop(c)})"
                else:
                    bad[c] = f"{bad[c]}; pico rows: {why[c]}"
    return dataclasses.replace(panel, data={k: v for k, v in panel.data.items() if k not in bad},
                               kind={k: v for k, v in panel.kind.items() if k not in bad},
                               family={k: v for k, v in panel.family.items() if k not in bad},
                               dropped={**panel.dropped, **bad}, notes=notes)


def build_panel_p(trace: Trace, step_s: int = 10, t_min: int = 0, episode: int = 0,
                  drop_degenerate: bool = True, min_minor_frac: float = 0.01) -> Panel:
    """E6-P per-cell panel (module docstring): ``build_panel`` + the E6-P knob / state / KPI columns. Obs-only."""
    p = build_panel(trace, step_s, t_min, episode, drop_degenerate=False)
    a = trace.features()
    assert not any(k.startswith(LABEL_PREFIX) for k in a)
    knobs = [tuple(k) for k in trace.meta["knobs"]]
    kidx = {k: i for i, k in enumerate(knobs)}
    C = len(p.cell_region)
    step = int(step_s)
    tk = np.unique(p.t)
    K = len(tk)
    cfg = np.stack([trace.config_at(int(t) + 1) for t in tk]) if K else np.zeros((0, len(knobs)))
    nb = p.neighbours
    nan = np.full(K, np.nan)

    def per_cell(fam, default):
        idx = [kidx.get((fam, c), -1) for c in range(C)]
        out = np.full((K, C), float(default))
        for c, i in enumerate(idx):
            if i >= 0:
                out[:, c] = cfg[:, i]
        return out, any(i >= 0 for i in idx)

    def nbr_mean(v):
        return np.stack([v[:, n].mean(1) if n else nan for n in nb], 1) if K else np.zeros((0, C))

    cols, have = {}, {}
    cols["own_ptx"], have["own_ptx"] = per_cell("ptx", 0.0)
    cols["own_prot_min"], have["own_prot_min"] = per_cell("prot_min", 0.0)
    sleep, have_sleep = per_cell("sleep", 0.0)
    cols["nbr_sleep"], have["nbr_sleep"] = nbr_mean(sleep), have_sleep
    cols["nbr_ptx"], have["nbr_ptx"] = nbr_mean(cols["own_ptx"]), have["own_ptx"]
    cols["nbr_prot_min"], have["nbr_prot_min"] = nbr_mean(cols["own_prot_min"]), have["own_prot_min"]

    def field(name):
        return lambda j: np.asarray(a[f"kpm_fast_{name}"][j], float)

    def act_ue(j):
        v = np.asarray(a["kpm_fast_act_ue"][j], float)
        return v.sum(1) if v.ndim == 2 else v

    def prot_viol(j):
        ev = np.asarray(a["kpm_fast_prot_eval"][j], float)
        fr = np.asarray(a["kpm_fast_prot_below_frac"][j], float)
        return np.where(ev > 0, np.rint(np.nan_to_num(fr) * ev), 0.0)

    fields = {"act_ue": act_ue}
    if "kpm_fast_prot_eval" in a and "kpm_fast_prot_below_frac" in a:
        fields["prot_viol"] = prot_viol
    for name in ("prot_act_ue", "edge_sinr_p"):
        if f"kpm_fast_{name}" in a:
            fields[name] = field(name)
    fast = _kpi_windows(a, "fast", tk, step, C, fields)

    def lag(v, L):
        out = np.full((K, C), np.nan)
        if K > L:
            out[L:] = v[:-L]
        return out

    data, kind, family = dict(p.data), dict(p.kind), dict(p.family)
    dropped = dict(p.dropped)

    def put(name, arr, k, fam):
        data[name] = np.asarray(arr, float).reshape(-1)       # row order: step-major, then cell
        kind[name], family[name] = k, fam

    for name, (k, fam) in KNOB_COLUMNS_P.items():
        if have[name]:
            put(name, cols[name], k, fam)
        else:
            dropped[name] = "knob_absent_in_mix"
    put("nbr_act_ue_lag", nbr_mean(lag(fast["act_ue"], 1)), "state_nbr", "act_ue")
    for fam in KPI_FAMILIES_PX:
        if fam not in fast:
            dropped[fam] = "report_field_absent"
            continue
        put(fam, fast[fam], "kpi", fam)
        put(fam + "_lag", lag(fast[fam], max(1, math.ceil(KPI_GRAN_S[fam] / step))), "kpi_lag", fam)
    out = dataclasses.replace(p, data=data, kind=kind, family=family, dropped=dropped)
    return drop_degenerate_p(out, min_minor_frac) if drop_degenerate else out


PANEL_REC_SCHEMA = "e6p-panel/1"


def panel_to_rec(panel: Panel) -> dict:
    """JSON-able record of a panel (float32 arrays via ``collect_p.enc``). The layout's ``cell_region`` (cell site) is
    NOT recorded (privileged on E6-P: ``gt_static``); ``panel_from_rec`` fills a placeholder (one region per cell)."""
    from .collect_p import enc
    return {"schema": PANEL_REC_SCHEMA, "step_s": int(panel.step_s), "n_rows": int(panel.n),
            "n_cells": len(panel.neighbours), "episode": enc(panel.episode), "t": enc(panel.t),
            "cell": enc(panel.cell), "columns": list(panel.data), "kind": dict(panel.kind),
            "family": dict(panel.family), "data": {c: enc(v) for c, v in panel.data.items()},
            "neighbours": [[int(x) for x in n] for n in panel.neighbours], "ownership": panel.ownership,
            "xapps": list(panel.xapps), "dropped": dict(panel.dropped), "notes": dict(panel.notes)}


def panel_from_rec(rec: dict, episode: int | None = None) -> Panel:
    """Inverse of ``panel_to_rec`` (values as float64; ``episode`` overrides the recorded episode id)."""
    from .collect_p import dec
    if rec.get("schema") != PANEL_REC_SCHEMA:
        raise ValueError(f"not a panel record: {rec.get('schema')!r}")
    n = int(rec["n_rows"])
    ep = np.full(n, int(episode)) if episode is not None else dec(rec["episode"]).astype(np.int64)
    return Panel(data={c: dec(rec["data"][c]).astype(float) for c in rec["columns"]}, kind=dict(rec["kind"]),
                 family=dict(rec["family"]), episode=ep, t=dec(rec["t"]).astype(np.int64),
                 cell=dec(rec["cell"]).astype(np.int64), step_s=int(rec["step_s"]),
                 cell_region=np.arange(int(rec["n_cells"])), neighbours=[list(x) for x in rec["neighbours"]],
                 ownership=rec["ownership"], xapps=list(rec["xapps"]), dropped=dict(rec["dropped"]),
                 notes=dict(rec.get("notes", {})))
