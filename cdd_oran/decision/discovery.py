"""TEMPLATE causal discovery for E6 with MSCR, conflict map, and a soft context prior (DESIGN.md REVISION v2 item 6).

Template = one cell-exchangeable graph: rows of all cells (and episodes) are pooled, so an edge "knob family ->
KPI family" (own or neighbour scope) is estimated once for every cell, not per cell.

Per KPI family y (``features.KPI_FAMILIES``):
  tested family   own knob families (knob_own, at the window start), neighbour knob aggregates (knob_nbr) and the
                  neighbour state (state_nbr); only these get p-values;
  conditioners    all own lagged KPIs (kpi_lag) and the cell class (context): MSCR's gate search conditions on them
                  but never tests them (lagged-KPI columns are conditioners only, as in MSCR-v2);
  test            ``discovery.mscr.discover_mscr`` (frozen statistic and p-values), UNCHANGED;
  selection       our own per-target Benjamini-Yekutieli at ``q`` over the KNOB columns only (the intervention
                  family), and separately over the neighbour-state columns (context ranking); MSCR's internal BY
                  over the whole tested family is not used;
  sign            coefficient of the column in an OLS (tiny ridge) of standardised y on all standardised candidate
                  and conditioner columns (a partial-regression direction, NOT an effect estimate).

What MSCR supports, and what is done here about it
  * MSCR's permutation null permutes y rows freely within each conditioner stratum: rows must be exchangeable
    under the null. It has no block / cluster permutation, and making it block-aware would need a change to
    ``mscr.py`` (frozen). So we THIN instead: per target, keep rows whose step start is a multiple of
    ``stride = max(thin_s, KPI_GRAN_S[y], step_s)`` (aligned to the report grid, so coarse KPIs are not
    duplicated), optionally rotating which cells are kept at each kept time (``cell_rotation``). Thinning makes
    rows approximately, NOT exactly, independent: knob values persist for minutes (carrier dwell 120 s, CIO
    random-walks) and cells at one time share load and neighbours. ``TargetDiagnostics`` reports the lag-1
    autocorrelation (same cell, consecutive kept rows) of y and of each knob, and Bartlett's effective sample size
    n_eff = n (1 - r_y r_x) / (1 + r_y r_x) per tested column (``dependence_ok`` = all n_eff >= nc * min_stratum),
    so the residual dependence is visible; with dependence_ok False the p-values are anti-conservative.
    p-values are therefore approximate; treat the graph as a ranked soft prior (below), not a certified FDR claim.
  * MSCR BUG-2 (ties binned by row order, so a constant / tied column bins by time): rows are SHUFFLED with a
    seeded permutation before MSCR, so ties are broken at random, and degenerate columns are dropped on the rows
    actually used (``features.degenerate_columns``, recorded). BUG-1 (constant y declares everything via
    rounding): degenerate targets are skipped (recorded) and y is standardised.
  * p-value floor 1 / (n_perm + 1): if it exceeds the BY rank-1 threshold q / (m c_m), nothing can ever be
    selected; flagged as ``power_floor_ok = False``.
  * MSCR declares ASSOCIATION edges: on observational/passively collected logs a p-value is not an
    interventional effect, and a missing edge is not evidence of no effect.

Graph use (never pruning by omission): every tested (knob, KPI) pair is kept with its p-value and a soft weight
w = floor + (1 - floor) * min(1, log(1/p) / log(n_perm + 1)) in [floor, 1]; ``declared`` is a flag, not a filter.
``conflict_map`` lists every (writer xApp, knob family, KPI owned by ANOTHER xApp) pair with its weight/p/sign;
``context_mask`` gives a region's ranked feature weights and neighbouring regions for the ContextSelector.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from cdd_oran.discovery.mscr import MSCR_VERSION, MSCRConfig, by_declare, discover_mscr

from .features import KPI_FAMILIES, KPI_GRAN_S, Panel, degenerate_columns, writers

# KPI family -> the xApp whose objective it is (its trigger KPI in cdd_oran/envs/e6/xapps.py and its O-RAN / SON use
# case): MRO (TS 32.522 mobility robustness) acts on too-late / too-early / RLF counts; TS (MLB / traffic steering)
# offloads where eMBB users suffer (embb_thp_p5 < target); ES (energy saving) owns energy; SLICE (slice SLA) acts on
# LL p95 delay. prb_util is SHARED state read by TS and ES (a mediator), owned by nobody. Declared, not learned.
KPI_OWNER = {"rlf": "MRO", "too_late": "MRO", "too_early": "MRO", "embb_thp_p5": "TS", "energy_w": "ES",
             "ll_delay_p95": "SLICE", "prb_util": None}
# E6-P (P1-P3 xApps; pass as DiscoveryConfig(kpi_owner=KPI_OWNER_P)): energy is the objective of BOTH ES and PowerES
# (an owner may be a tuple: a knob of any listed owner is then not a conflict on that KPI); the protected-slice KPIs
# belong to SliceGuarantee, the edge SINR to Coverage; prb_util / prot_act_ue / rlf are shared state, owned by nobody
# (no MRO in E6-P). Declared, not learned.
KPI_OWNER_P = {"energy_w": ("ES", "PowerES"), "prot_viol": "SliceGuarantee", "edge_sinr_p": "Coverage",
               "embb_thp_p5": None, "ll_delay_p95": None, "prb_util": None, "prot_act_ue": None, "rlf": None,
               "too_late": None, "too_early": None}


@dataclasses.dataclass(frozen=True)
class DiscoveryConfig:
    mscr: MSCRConfig = MSCRConfig()     # frozen MSCR-v2 constants by default (n_perm 2999)
    q: float = 0.05                     # per-target BY level over the knob columns
    thin_s: int = 60                    # min seconds between kept rows of one cell (>= each KPI's granularity)
    cell_rotation: int = 1              # keep cells with (cell + kept-time index) % m == 0 (1 = all cells)
    seed: int = 0                       # row shuffle and MSCR bank seed
    min_minor_frac: float = 0.01        # degenerate-column threshold (features.degenerate_columns)
    max_nan_frac: float = 0.1           # drop a column (recorded) if more of the target's rows lack it
    weight_floor: float = 0.05          # soft-prior floor: no tested column ever gets weight 0
    ridge: float = 1e-3                 # sign regression ridge (on standardised columns)
    targets: tuple = KPI_FAMILIES
    kpi_owner: dict | None = None       # KPI family -> owner xApp (or tuple of owners); None = KPI_OWNER (E6)


@dataclasses.dataclass
class TargetDiagnostics:
    kpi: str
    status: str                         # "ok" | "skipped: <reason>"
    n_rows: int = 0
    n_cells: int = 0
    n_times: int = 0
    stride_s: int = 0
    tested: tuple = ()
    conditioners: tuple = ()
    dropped: dict = dataclasses.field(default_factory=dict)
    acf1_y: float = float("nan")
    acf1_x: dict = dataclasses.field(default_factory=dict)
    n_eff: dict = dataclasses.field(default_factory=dict)   # column -> Bartlett n (1 - r_y r_x) / (1 + r_y r_x)
    dependence_ok: bool = True          # every tested column has n_eff >= nc * min_stratum
    power_floor_ok: bool = True


@dataclasses.dataclass
class TemplateGraph:
    """Template graph over (tested column -> KPI family). ``edges``: one dict per tested pair (declared or not)."""
    edges: list
    diagnostics: dict                   # kpi -> TargetDiagnostics
    ownership: dict                     # xapp -> {knob family: {"req", "applied"}}
    xapps: list
    kpi_owner: dict
    cell_region: np.ndarray
    neighbours: list
    config: DiscoveryConfig
    mscr_version: str = MSCR_VERSION

    def edge(self, column: str, kpi: str) -> dict | None:
        return next((e for e in self.edges if e["column"] == column and e["kpi"] == kpi), None)

    def declared_edges(self) -> list:
        return [e for e in self.edges if e["declared"]]

    def to_dict(self) -> dict:
        return {"mscr_version": self.mscr_version, "config": _jsonable(dataclasses.asdict(self.config)),
                "edges": self.edges, "diagnostics": {k: dataclasses.asdict(v) for k, v in self.diagnostics.items()},
                "ownership": self.ownership, "xapps": self.xapps, "kpi_owner": self.kpi_owner,
                "cell_region": [int(x) for x in self.cell_region], "neighbours": self.neighbours,
                "conflicts": conflict_map(self)}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------------ rows
def thin_rows(panel: Panel, stride_s: int, cell_rotation: int = 1) -> np.ndarray:
    """Row indices kept by thinning: step start on the ``stride_s`` grid; with ``cell_rotation`` m > 1 only cells
    with (cell + kept-time index) % m == 0 at each kept time (fewer same-time rows, more distinct times)."""
    stride = max(int(stride_s), int(panel.step_s))
    keep = panel.t % stride == 0
    if cell_rotation > 1:
        j = panel.t // stride
        keep &= (panel.cell + j) % int(cell_rotation) == 0
    return np.nonzero(keep)[0]


def _acf1(v: np.ndarray, panel: Panel, rows: np.ndarray, stride: int) -> float:
    """Lag-1 autocorrelation of v over consecutive kept rows of the same (episode, cell), pooled."""
    ep, cell, t = panel.episode[rows], panel.cell[rows], panel.t[rows]
    order = np.lexsort((t, cell, ep))
    v, ep, cell, t = v[order], ep[order], cell[order], t[order]
    ok = (ep[1:] == ep[:-1]) & (cell[1:] == cell[:-1]) & (t[1:] - t[:-1] == stride)
    a, b = v[:-1][ok], v[1:][ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _n_eff(n: int, ry: float, rx: float) -> float:
    """Bartlett's effective sample size for a correlation between two AR(1)-like series (NaN acf -> 0)."""
    ry, rx = (0.0 if not np.isfinite(r) else max(r, 0.0) for r in (ry, rx))
    return float(n * (1.0 - ry * rx) / (1.0 + ry * rx))


def _signs(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xs = (x - x.mean(0)) / x.std(0)
    ys = (y - y.mean()) / y.std()
    n, m = xs.shape
    return np.linalg.solve(xs.T @ xs / n + ridge * np.eye(m), xs.T @ ys / n)


def soft_weight(p: float, n_perm: int, floor: float) -> float:
    if not np.isfinite(p):
        return floor
    return floor + (1.0 - floor) * min(1.0, math.log(1.0 / p) / math.log(n_perm + 1.0))


# ------------------------------------------------------------------------------------------------ discovery
def discover_template(panel: Panel, config: DiscoveryConfig | None = None, n_jobs: int | None = None,
                      device: str | None = None) -> TemplateGraph:
    """Run MSCR per KPI family on the pooled, thinned, shuffled panel (see module docstring)."""
    cfg = config or DiscoveryConfig()
    tested_all = panel.columns("knob_own", "knob_nbr", "state_nbr")
    cond_all = panel.columns("kpi_lag", "context")
    edges, diags = [], {}
    min_rows = cfg.mscr.nc * cfg.mscr.min_stratum
    for ti, kpi in enumerate(cfg.targets):
        if kpi not in panel.data:
            diags[kpi] = TargetDiagnostics(kpi, "skipped: target column absent or degenerate on the panel "
                                           f"({panel.dropped.get(kpi, 'absent')})")
            continue
        stride = max(cfg.thin_s, KPI_GRAN_S.get(kpi, 1), panel.step_s)
        rows = thin_rows(panel, stride, cfg.cell_rotation)
        y = panel.data[kpi][rows]
        cols = tested_all + cond_all
        # columns often NaN on the target's rows (coarse lags early in an episode, KPIs without samples) would cost
        # too many rows in the complete-case filter: drop them first (recorded)
        ok_y = rows[np.isfinite(y)]
        dropped = {c: "mostly_nan" for c in cols
                   if len(ok_y) and np.mean(~np.isfinite(panel.data[c][ok_y])) > cfg.max_nan_frac}
        cols = [c for c in cols if c not in dropped]
        ok = np.isfinite(y)
        for c in cols:
            ok &= np.isfinite(panel.data[c][rows])
        rows, y = rows[ok], y[ok]
        dropped.update(degenerate_columns({c: panel.data[c][rows] for c in cols}, cfg.min_minor_frac))
        tested = [c for c in tested_all if c in cols and c not in dropped]
        cond = [c for c in cond_all if c in cols and c not in dropped]
        d = TargetDiagnostics(kpi, "ok", n_rows=len(rows), n_cells=len(np.unique(panel.cell[rows])),
                              n_times=len(np.unique(np.stack([panel.episode[rows], panel.t[rows]]), axis=1)))
        d.stride_s, d.tested, d.conditioners, d.dropped = stride, tuple(tested), tuple(cond), dropped
        diags[kpi] = d
        if degenerate_columns({kpi: y}, cfg.min_minor_frac):
            d.status = f"skipped: degenerate target ({degenerate_columns({kpi: y}, cfg.min_minor_frac)[kpi]})"
            continue
        if not tested:
            d.status = "skipped: no testable column"
            continue
        if len(tested) + len(cond) < 2:
            d.status = "skipped: MSCR needs >= 2 columns"
            continue
        if len(rows) < min_rows:
            d.status = f"skipped: {len(rows)} rows < nc * min_stratum = {min_rows}"
            continue
        d.acf1_y = _acf1(y, panel, rows, stride)
        d.acf1_x = {c: _acf1(panel.data[c][rows], panel, rows, stride) for c in tested}
        d.n_eff = {c: _n_eff(len(rows), d.acf1_y, r) for c, r in d.acf1_x.items()}
        d.dependence_ok = all(v >= min_rows for v in d.n_eff.values())

        perm = np.random.default_rng([cfg.seed, 4242, ti]).permutation(len(rows))
        rows, y = rows[perm], y[perm]
        x = np.stack([panel.data[c][rows] for c in tested + cond], 1)
        ys = (y - y.mean()) / y.std()
        res = discover_mscr(x, ys, n_params=len(tested), seed=cfg.seed * 1000 + ti, config=cfg.mscr,
                            n_jobs=n_jobs, device=device)
        pv, s_star = res.pvals[0], res.s_star[0]
        coef = _signs(x, y, cfg.ridge)[: len(tested)]
        is_knob = np.array([panel.kind[c] in ("knob_own", "knob_nbr") for c in tested])
        declared = np.zeros(len(tested), bool)
        for grp in (is_knob, ~is_knob):
            if grp.any():
                declared[grp] = by_declare(pv[grp], cfg.q)
        m_knob = max(int(is_knob.sum()), 1)
        c_m = sum(1.0 / k for k in range(1, m_knob + 1))
        d.power_floor_ok = 1.0 / (cfg.mscr.n_perm + 1) <= cfg.q / (m_knob * c_m)
        for i, c in enumerate(tested):
            edges.append({"kpi": kpi, "column": c, "family": panel.family[c], "kind": panel.kind[c],
                          "scope": panel.scope(c), "p": float(pv[i]), "s_star": float(s_star[i]),
                          "coef": float(coef[i]), "sign": int(np.sign(coef[i])), "declared": bool(declared[i]),
                          "weight": soft_weight(float(pv[i]), cfg.mscr.n_perm, cfg.weight_floor)})
    owner = KPI_OWNER if cfg.kpi_owner is None else cfg.kpi_owner
    return TemplateGraph(edges=edges, diagnostics=diags, ownership=panel.ownership, xapps=list(panel.xapps),
                         kpi_owner={k: v for k, v in owner.items()}, cell_region=panel.cell_region,
                         neighbours=panel.neighbours, config=cfg)


# ------------------------------------------------------------------------------------------------ conflict map
def conflict_map(graph: TemplateGraph, declared_only: bool = False, min_req: int = 1) -> list:
    """Every knob edge whose knob family is written by xApp X and whose KPI is owned by another ACTIVE xApp Y.

    Writers come from the logged requests (``features.knob_ownership``); KPI owners from ``graph.kpi_owner``.
    Pairs are kept whether or not MSCR declared the edge (``declared``/``weight``/``p`` say how strong the evidence
    is), unless ``declared_only``. Sorted by p. Own-KPI harm (X's knob -> X's KPI) is NOT a conflict and is not
    listed; it stays visible in ``graph.edges``."""
    out = []
    for e in graph.edges:
        if e["kind"] not in ("knob_own", "knob_nbr") or (declared_only and not e["declared"]):
            continue
        owner = graph.kpi_owner.get(e["kpi"])
        owners = tuple(owner) if isinstance(owner, (tuple, list)) else (owner,)
        live = [o for o in owners if o is not None and o in graph.xapps]
        if not live:
            continue
        owner = live[0] if len(live) == 1 else "+".join(live)
        for x in writers(graph.ownership, e["family"], min_req):
            if x not in live:
                out.append({"src_xapp": x, "knob_family": e["family"], "column": e["column"], "scope": e["scope"],
                            "kpi": e["kpi"], "dst_xapp": owner, "p": e["p"], "sign": e["sign"], "coef": e["coef"],
                            "declared": e["declared"], "weight": e["weight"]})
    return sorted(out, key=lambda r: (r["p"], r["src_xapp"], r["kpi"], r["column"]))


# ------------------------------------------------------------------------------------------------ context prior
@dataclasses.dataclass
class ContextPrior:
    """SOFT prior for a region's effect model (never a hard prune: every weight >= the floor, untested columns 1)."""
    region: int
    cells: list
    column_weights: dict                # column -> weight (max over KPI targets)
    target_weights: dict                # kpi -> {column: weight}
    ranked: list                        # [(column, weight)] descending
    neighbour_regions: list             # [(region, weight)] descending, weight in [floor, 1]


def context_mask(graph: TemplateGraph, region: int, kpis=None) -> ContextPrior:
    """Which context matters for ``region`` (a cell site): template column weights (tested columns: soft weight from
    their MSCR p-value; conditioners: 1.0, always kept) and neighbouring regions ranked by how much of the region's
    neighbour-scope evidence they carry (each neighbour cell n of a region cell c contributes the max neighbour-
    scope column weight / |neighbours(c)|, as the aggregates are neighbour means)."""
    floor = graph.config.weight_floor
    kpis = list(kpis) if kpis is not None else [k for k, d in graph.diagnostics.items() if d.status == "ok"]
    target_w = {}
    for k in kpis:
        d = graph.diagnostics.get(k)
        if d is None:
            continue
        w = {c: 1.0 for c in d.conditioners}
        w.update({e["column"]: e["weight"] for e in graph.edges if e["kpi"] == k})
        target_w[k] = w
    col_w: dict = {}
    for w in target_w.values():
        for c, v in w.items():
            col_w[c] = max(col_w.get(c, floor), v)
    nbr_cols = [e for e in graph.edges if e["scope"] == "neighbour" and e["kpi"] in target_w]
    w_nbr = max((e["weight"] for e in nbr_cols), default=floor)
    reg = np.asarray(graph.cell_region)
    cells = [int(c) for c in np.nonzero(reg == int(region))[0]]
    score: dict = {}
    for c in cells:
        nb = [n for n in graph.neighbours[c] if reg[n] != int(region)]
        for n in nb:
            r = int(reg[n])
            score[r] = score.get(r, 0.0) + w_nbr / len(nb)
    top = max(score.values(), default=0.0)
    nregs = sorted(((r, floor + (1 - floor) * (s / top if top > 0 else 0.0)) for r, s in score.items()),
                   key=lambda z: (-z[1], z[0]))
    return ContextPrior(region=int(region), cells=cells, column_weights=col_w, target_weights=target_w,
                        ranked=sorted(col_w.items(), key=lambda z: (-z[1], z[0])), neighbour_regions=nregs)


def region_weight_matrix(graph: TemplateGraph, kpis=None) -> tuple[list, np.ndarray]:
    """(regions, W (R, R)) for ``effect_model.ContextSelector("weights", W)``: row r = ``context_mask(graph, r)``
    neighbour-region weights; every other off-diagonal entry gets the floor (soft prior, nothing omitted); diagonal 0.
    Regions are sorted region ids (the order of ``effect_model.topology_weights``)."""
    regions = sorted({int(x) for x in np.asarray(graph.cell_region)})
    rix = {r: i for i, r in enumerate(regions)}
    W = np.full((len(regions), len(regions)), graph.config.weight_floor)
    np.fill_diagonal(W, 0.0)
    for r in regions:
        for r2, w in context_mask(graph, r, kpis).neighbour_regions:
            W[rix[r], rix[r2]] = w
    return regions, W
