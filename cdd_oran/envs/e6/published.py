"""E6 re-implementations of PUBLISHED xApp-conflict methods, as E6 arbiters that see ONLY ``obs``.

Spec, paper quotes, E6 mappings, settings chosen in each method's favour and fidelity risks:
``docs/benchmark/E6_PUBLISHED_BASELINES.md``. Status per method:

  QACM          Wadud et al., IEEE TGCN 8(3) 2024, Alg. 1 / Eq. (3), with the authors' rule-based CDC (Wadud et al.,
                INFOCOM'25 workshop, Sec. IV) and a KPI predictor (ANN vs PR, Sec. VI-B) fitted on logs  -> IMPLEMENTED
  CMF           Adamczyk et al., IEEE ComMag 2023 (DCD / ICD + prioritised CR Agent); SBD / P-x variants of
                Wadud'25 Sec. V                                                                        -> IMPLEMENTED
  PACIFISTA     del Prever et al., IEEE TMC 2025, Sec. 7-8 (INT severity + delta_TOL deploy greedy)   -> IMPLEMENTED
  Djidjev       Djidjev & Kaminski arXiv:2606.06459 Eq. (4) + arXiv:2606.06663 Alg. 1 (detector only;
                resolver is OURS, declared)                                                            -> IMPLEMENTED
  Sharma        arXiv:2510.13031 (XGBoost -> SHAP -> DAG -> DoWhy ATE / CausalForestDML CATE)        -> INTERFACE ONLY
  two-tower     arXiv:2601.13213 (supervised two-tower + sparsemax)                                    -> INTERFACE ONLY
  GRAPHICA      Al Shami et al., arXiv:2503.03523 (GCN conflict classifier, no mitigation)            -> INTERFACE ONLY

Training-data contract (every learned piece): ``fit(episodes)`` where each episode is the ``env.log`` list of an
``E6Env(log=True)`` run (per second: {"t", "config", "reports", ...}). Collect DEV logs with ``run_logged``; never TEST.
"""
from __future__ import annotations

import bisect
import itertools

import numpy as np

from . import config as C
from .baselines import PRIORITY, subset
from .ric import LIMITS, _quantise

# ------------------------------------------------------------------------------------------ xApp descriptors
# Operator-declared xApp descriptors: input control parameters I_x and KPIs K_x (QACM Sec. III; Wadud'25 Sec. IV
# "the MNO is expected to provide the xApp details, including their ICPs and KPIs"). span_s = control time span of a
# decision (CMF DCD "the control duration expected by the xApp") [INFERRED = the xApp's cadence].
MANIFEST = {"MRO": {"icp": ("cio", "ttt", "hys"), "kpi": "ho_fail", "span_s": 30.0},
            "TS": {"icp": ("cio",), "kpi": "embb_thp", "span_s": 10.0},
            "ES": {"icp": ("carrier",), "kpi": "energy", "span_s": 10.0},
            "SLICE": {"icp": ("ll_ratio",), "kpi": "ll_delay", "span_s": 1.0}}
# QoS thresholds q with QACM's delta: 1 = KPI minimised (violated above q), 0 = maximised (violated below q).
# From E6's SLA targets; ho_fail = MRO's nominal too-late trigger ratio [INFERRED, E6 has no numeric RLF target];
# energy has no SLA target -> None = always satisfied (in QACM's favour on the SVR endpoint).
QOS = {"ll_delay": (C.LL_DELAY_TARGET_S, 1), "embb_thp": (C.EMBB_THP_TARGET_BPS, 0), "ho_fail": (0.02, 1),
       "energy": None}
KPI_GRAN = {"ll_delay": "fast", "util": "fast", "embb_thp": "thp", "ho_fail": "mob", "energy": "energy"}


def kpi_of(rep, name):
    """Per-cell KPI array from one KPM report of the matching granularity (NaN = no samples)."""
    if name == "ll_delay":
        return np.asarray(rep["ll_delay_p95"], float)
    if name == "util":
        return np.asarray(rep["prb_util"], float)
    if name == "embb_thp":
        return np.asarray(rep["embb_thp_p5"], float)
    if name == "energy":
        return np.asarray(rep["energy_j"], float)
    if name == "ho_fail":                                    # too-late share of HO attempts per serving cell
        tl = rep["too_late"].sum(1)
        den = rep["ho_att"].sum(1) + tl
        return np.where(den >= 3, tl / np.maximum(den, 1), np.nan)
    raise KeyError(name)


def violated(v, qos):
    if qos is None or not np.isfinite(v):
        return False
    q, delta = qos
    return v > q if delta == 1 else v < q


class KPIView:
    """Latest delivered KPM report per granularity, as an arbiter sees it (reports can arrive out of order)."""

    def __init__(self):
        self.rep = {}

    def ingest(self, reps):
        for r in reps:
            old = self.rep.get(r["gran"])
            if old is None or r["t1"] >= old["t1"]:
                self.rep[r["gran"]] = r

    def get(self, name):
        rep = self.rep.get(KPI_GRAN[name])
        return None if rep is None else kpi_of(rep, name)


# ------------------------------------------------------------------------------------------ knob helpers
def knob_cells(k):
    """Cells a knob acts on, with the cell's role: CIO[s, n] -> (s, src), (n, dst); per-cell knobs -> (c, self)."""
    return ((k[1], "src"), (k[2], "dst")) if k[0] == "cio" else ((k[1], "self"),)


def clip_step(k, cur, v):
    """ric.feasible without the dwell check: quantise, clamp to the range, clip to one actuator step."""
    lo, hi, mstep, _ = LIMITS[k[0]]
    v = min(max(_quantise(k, v), lo), hi)
    if k[0] == "ttt":
        i0, i1 = C.TTT_SET_MS.index(int(cur)), C.TTT_SET_MS.index(int(v))
        return float(C.TTT_SET_MS[i0 + int(np.clip(i1 - i0, -1, 1))])
    return float(cur + np.clip(v - cur, -mstep, mstep)) if mstep is not None else float(v)


def one_step_values(k, cur):
    """Every actuator-grid value reachable from ``cur`` in one change (the discrete range N of QACM Alg. 1)."""
    typ = k[0]
    lo, hi, mstep, _ = LIMITS[typ]
    if typ == "ttt":
        i = C.TTT_SET_MS.index(int(cur))
        return [float(v) for v in C.TTT_SET_MS[max(i - 1, 0):i + 2]]
    if typ == "sleep":
        return [0.0, 1.0]
    grid = {"cio": 1.0, "hys": 0.5, "ll_ratio": 0.05, "carrier": 1.0}[typ]
    n = int(round(mstep / grid))
    vals = {_quantise(k, cur + j * grid) for j in range(-n, n + 1)}
    return sorted(v for v in vals if lo - 1e-9 <= v <= hi + 1e-9)


def param_groups(manifest=MANIFEST):
    """Wadud'25 Alg. 1: PkG[kpi] = every ICP (knob type) of the xApps that own that KPI."""
    pkg = {}
    for d in manifest.values():
        pkg.setdefault(d["kpi"], set()).update(d["icp"])
    return pkg


def run_logged(cfg, arbiter=None, **kw):
    """Run one logged E6 episode (DEV seeds only) and return (score, env.log) for the ``fit(episodes)`` contract."""
    from .env import E6Env
    env = E6Env(cfg, log=True, **kw)
    return env.run(arbiter), env.log


# ------------------------------------------------------------------------------------------ KPI predictor (QACM VI-B)
ROLES = ("self", "src", "dst")


def _replay(log):
    """Per logged second i: (t, config before the second's decisions, config after, arbiter-visible KPIView)."""
    view = KPIView()
    prev = None
    for e in log:
        view.ingest(e["reports"])
        if prev is not None:
            yield e["t"], prev, e["config"], view
        prev = e["config"]


def transitions(episodes, kpis, null_ratio=1.0, seed=0):
    """(knob type, kpi) -> (X, y) regression samples from logged episodes.

    One sample per (applied knob change at second t, acted-on cell, kpi): features = [role one-hot, delta, value before,
    kpi now, prb_util now] as the arbiter saw them at t; target = the kpi of the first report of its granularity whose
    window starts at or after t (the change takes effect from the next tick). ``null_ratio`` x as many unchanged knobs
    of the same type (delta = 0) are sampled per second, so the model also predicts the no-change outcome that
    Alg. 1 compares against."""
    rng = np.random.default_rng(seed)
    out = {}
    for log in episodes:
        by_t0 = {}
        for e in log:
            for r in e["reports"]:
                by_t0.setdefault(r["gran"], {})[r["t0"]] = r
        idx = {g: (sorted(d), d) for g, d in by_t0.items()}
        for t, before, after, view in _replay(log):
            util = view.get("util")
            if util is None:
                continue
            changed = [k for k in after if abs(after[k] - before[k]) > 1e-9]
            rows = [(k, after[k] - before[k], before[k]) for k in changed]
            by_typ = {}
            for k in changed:
                by_typ[k[0]] = by_typ.get(k[0], 0) + 1
            for typ, n in by_typ.items():
                pool = [k for k in after if k[0] == typ and k not in changed]
                m = min(len(pool), int(round(n * null_ratio)))
                rows += [(pool[j], 0.0, before[pool[j]]) for j in rng.choice(len(pool), m, replace=False)]
            for name in kpis:
                now = view.get(name)
                g = KPI_GRAN[name]
                if now is None or g not in idx:
                    continue
                keys, d = idx[g]
                j = bisect.bisect_left(keys, t - 1e-9)
                if j == len(keys):
                    continue
                tgt = kpi_of(d[keys[j]], name)
                for k, dv, v0 in rows:
                    for c, role in knob_cells(k):
                        x = [role == r_ for r_ in ROLES] + [dv, v0, now[c], util[c]]
                        if np.all(np.isfinite(x)) and np.isfinite(tgt[c]):
                            X, y = out.setdefault((k[0], name), ([], []))
                            X.append(x)
                            y.append(tgt[c])
    return {key: (np.asarray(X, float), np.asarray(y, float)) for key, (X, y) in out.items()}


class _ANN:
    """QACM Sec. VI-B ANN: 4 hidden layers x 128 tanh, dropout 0.2 after each, linear output, Adam, MSE, 10 epochs,
    batch size 10."""

    def __init__(self, d, seed):
        import torch
        torch.manual_seed(seed)
        layers, w = [], d
        for _ in range(4):
            layers += [torch.nn.Linear(w, 128), torch.nn.Tanh(), torch.nn.Dropout(0.2)]
            w = 128
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(w, 1)).double()
        self.seed = seed

    def fit(self, X, y, epochs=10, batch=10):
        import torch
        g = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters())
        Xt, yt = torch.as_tensor(X), torch.as_tensor(y)[:, None]
        self.net.train()
        for _ in range(epochs):
            for b in torch.randperm(len(Xt), generator=g).split(batch):
                opt.zero_grad()
                torch.nn.functional.mse_loss(self.net(Xt[b]), yt[b]).backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X, float))).numpy()[:, 0]


class _PR:
    """QACM Sec. VI-B Polynomial Regression (degree 2, least squares)."""

    @staticmethod
    def _phi(X):
        X = np.asarray(X, float)
        cols = [np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])]
        cols += [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
        return np.stack(cols, 1)

    def fit(self, X, y):
        P = self._phi(X)
        self.w = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ y)
        return self

    def predict(self, X):
        return self._phi(X) @ self.w


def _r2(y, p):
    return 1.0 - float(np.sum((y - p) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-12)


class KPIPredictor:
    """Per (knob type, kpi) regression of the next-window cell KPI on the pending change (QACM Sec. VI-B: xApps are
    "pre-trained with offline KPI prediction models capable of estimating KPI values based on provided ICPs").

    KPIs are z-scored (Sec. VI-A). For every (type, kpi) both the paper's ANN and PR are fitted and the one with the
    higher held-out R^2 is kept (the paper picks by EVS / R^2 / MSE, Table III). Fewer than ``min_samples`` samples
    -> persistence (predict the current value): the method then sees no effect of that knob [declared fallback]."""

    def __init__(self, kpis=("ll_delay", "embb_thp", "ho_fail", "energy"), null_ratio=1.0, min_samples=30,
                 holdout=0.2, seed=0):
        self.kpis, self.null_ratio, self.min_samples, self.holdout, self.seed = kpis, null_ratio, min_samples, \
            holdout, seed
        self.models, self.report, self.zs, self.fscale = {}, {}, {}, {}

    def fit(self, episodes):
        episodes = [list(e) for e in episodes]
        data = transitions(episodes, self.kpis, self.null_ratio, self.seed)
        for name in self.kpis:                               # z-score statistics per KPI over all logged cells
            vals = [kpi_of(r, name) for log in episodes for e in log for r in e["reports"]
                    if r["gran"] == KPI_GRAN[name]]
            v = np.concatenate(vals) if vals else np.array([])
            v = v[np.isfinite(v)]
            self.zs[name] = (float(v.mean()), float(max(v.std(), 1e-9))) if len(v) > 1 else (0.0, 1.0)
        for (typ, name), (X, y) in data.items():
            if name not in self.zs:
                continue
            Xz, yz = self._feat(name, X), self.z(name, y)
            mu, sd = Xz[:, 3:].mean(0), Xz[:, 3:].std(0) + 1e-9
            self.fscale[(typ, name)] = (mu, sd)
            Xs = np.concatenate([Xz[:, :3], (Xz[:, 3:] - mu) / sd], 1)
            if len(yz) < self.min_samples:
                self.report[(typ, name)] = {"n": len(yz), "model": "persistence"}
                continue
            perm = np.random.default_rng(self.seed).permutation(len(yz))
            n_te = max(int(len(yz) * self.holdout), 1)
            te, tr = perm[:n_te], perm[n_te:]
            cands = {"ANN": _ANN(Xs.shape[1], self.seed).fit(Xs[tr], yz[tr]), "PR": _PR().fit(Xs[tr], yz[tr])}
            r2 = {m: _r2(yz[te], f.predict(Xs[te])) for m, f in cands.items()}
            best = max(r2, key=r2.get)
            self.models[(typ, name)] = cands[best]
            self.report[(typ, name)] = {"n": len(yz), "model": best, "r2": r2}
        return self

    def z(self, name, v):
        mu, sd = self.zs[name]
        return (np.asarray(v, float) - mu) / sd

    def _feat(self, name, X):
        X = np.array(X, float)
        X[:, 5] = self.z(name, X[:, 5])                      # current kpi, z-scored like the target
        return X

    def predict_z(self, typ, name, X):
        """z-scored predicted next-window KPI for raw feature rows [role one-hot, delta, value, kpi now, util now]."""
        Xz = self._feat(name, X)
        m = self.models.get((typ, name))
        if m is None:
            return Xz[:, 5]
        mu, sd = self.fscale[(typ, name)]
        return m.predict(np.concatenate([Xz[:, :3], (Xz[:, 3:] - mu) / sd], 1))


# ------------------------------------------------------------------------------------------ QACM
class QACM:
    """QACM (Wadud et al. 2024) as the CMC behind the authors' rule-based CDC, one conflicting parameter at a time.

    CDC (Wadud'25 Sec. IV, used pre-action = in QACM's favour): a pending request of xApp a on knob k is in conflict
    with every other deployed xApp b whose SLA-sensitive KPI currently violates its QoS threshold on a cell k acts on
    ("The CDC is triggered only when a KPI violation occurs"; b == a -> "no conflict"), and with every other xApp
    requesting the same knob in the same second (direct conflict). ``anticipate=True`` [INFERRED, off by default]
    also involves non-violated xApps. |X'| < 2 -> no conflict -> accept (Eq. 3j).

    CMC (Alg. 1): candidates p_l = the one-step actuator grid within [p_min_opt, p_max_opt] = [min, max] of the
    current value and the (step-clipped) proposals [INFERRED: the xApps' "individual optimal configuration range"];
    U_i(p_l) = mean over the acted-on cells of the predicted z-scored KPI of x_i (weighted average of an xApp's KPIs,
    Sec. VI-A); d_i = shortfall (delta=0) / excess (delta=1) from the z-scored threshold, s_i = 1 if met;
    fCost = sum_i w_i d_i zeta - (sum_i s_i)^2; the first strict minimum wins. Weights: equal (QACM) or ``weights``
    (QACM-P, normalised over X'). p_opt -> accept (= a proposal) / reject (= current value) / ("modify", p_opt)."""

    def __init__(self, predictor, manifest=MANIFEST, qos=QOS, weights=None, zeta=1e3, anticipate=False,
                 order="prop_first"):
        self.pred, self.manifest, self.qos, self.weights = predictor, manifest, qos, weights
        self.zeta, self.anticipate, self.order = zeta, anticipate, order
        self.view = KPIView()
        self.stats = {"conflicts": 0, "accept": 0, "reject": 0, "modify": 0}

    def involved(self, k, requesters, deployed):
        """CDC: the set X' of xApps in conflict over knob k (requesters included)."""
        X = set(requesters)
        for b in deployed:
            d = self.manifest.get(b)
            if b in X or d is None:
                continue
            v = self.view.get(d["kpi"])
            if v is None:
                continue
            if self.anticipate or any(violated(v[c], self.qos.get(d["kpi"])) for c, _ in knob_cells(k)):
                X.add(b)
        return X

    def utilities(self, k, cur, cands, xapp):
        """U_i(p_l) for every candidate (None if the xApp has no KPI signal on the acted-on cells)."""
        name = self.manifest[xapp]["kpi"]
        now, util = self.view.get(name), self.view.get("util")
        if now is None or util is None:
            return None
        rows, ncell = [], 0
        for c, role in knob_cells(k):
            if not (np.isfinite(now[c]) and np.isfinite(util[c])):
                continue
            ncell += 1
            rows += [[role == r_ for r_ in ROLES] + [p - cur, cur, now[c], util[c]] for p in cands]
        if not ncell:
            return None
        u = self.pred.predict_z(k[0], name, np.asarray(rows, float)).reshape(ncell, len(cands))
        return u.mean(0)

    def alg1(self, k, cur, props, X):
        """QACM Alg. 1 over the candidate grid; returns p_opt."""
        lo, hi = min([cur] + props), max([cur] + props)
        grid = [v for v in one_step_values(k, cur) if lo - 1e-9 <= v <= hi + 1e-9]
        if self.order == "prop_first":       # ties (equal fCost) keep the xApps' request: pass-through when blind
            head = list(dict.fromkeys(props + [cur]))
            grid = head + [v for v in grid if all(abs(v - h) > 1e-9 for h in head)]
        w = {x: (self.weights or {}).get(x, 1.0) for x in X}
        tot = sum(w.values()) or 1.0
        cost = np.zeros(len(grid))
        sat = np.zeros(len(grid))
        for x in sorted(X):
            name = self.manifest[x]["kpi"]
            q = self.qos.get(name)
            U = self.utilities(k, cur, grid, x) if q is not None else None
            if U is None:                     # no threshold / no signal: always satisfied, d = 0
                sat += 1
                continue
            qz = float(self.pred.z(name, q[0]))
            d = np.maximum(qz - U, 0.0) if q[1] == 0 else np.maximum(U - qz, 0.0)
            cost += w[x] / tot * d * self.zeta
            sat += (d <= 0)
        f = cost - sat ** 2
        best, fmin = grid[0], np.inf
        for p, fc in zip(grid, f, strict=True):            # Alg. 1 steps 24-26: strict improvement only
            if fmin > fc:
                best, fmin = p, fc
        return best

    def __call__(self, obs):
        self.view.ingest(obs["new_reports"])
        reqs = obs["requests"]
        dec = ["accept"] * len(reqs)
        deployed = [x for x in obs["static"]["xapps"] if x in self.manifest]
        by_knob = {}
        for i, r in enumerate(reqs):
            by_knob.setdefault(r["knob"], []).append(i)
        for k, idx in by_knob.items():
            X = self.involved(k, [reqs[i]["xapp"] for i in idx], deployed)
            if len(X) < 2:
                self.stats["accept"] += len(idx)
                continue
            self.stats["conflicts"] += 1
            cur = float(obs["config"].get(k, reqs[idx[0]]["cur"]))   # xApps may request knobs absent from config
            props = [clip_step(k, cur, reqs[i]["prop"]) for i in idx]
            p = self.alg1(k, cur, props, X)
            hit = [i for i, pc in zip(idx, props, strict=True) if abs(pc - p) < 1e-9]
            for i in idx:
                dec[i] = "accept" if i in hit else "reject"
            if abs(p - cur) > 1e-9 and not hit:
                dec[idx[-1]] = ("modify", p)
            for i in idx:
                self.stats[dec[i] if isinstance(dec[i], str) else "modify"] += 1
        return {"decisions": dec, "writes": []}


# ------------------------------------------------------------------------------------------ CMF priority / SBD
class CMF:
    """Conflict Mitigation Framework (Adamczyk et al. 2023): DCD + ICD over the "recently changed parameters" of the
    currently effective decisions, CR Agent = prioritisation ("If a given xApp is prioritized, each of its decisions
    takes effect on the network regardless of conflicts"). ``mode="sbd"`` = Wadud'25 set-back-to-default: a
    conflicting request is replaced by the knob's initial (3GPP default) value.

    Effective decision = an accepted request younger than its xApp's control time span (``span_s`` overrides the
    manifest). ICD parameter groups: ``groups="manifest"`` = Wadud'25 Alg. 1 PkG on the same cell (faithful);
    ``groups="cell"`` = every knob acting on a shared cell [INFERRED, broader]."""

    def __init__(self, order=PRIORITY, mode="priority", span_s=None, groups="manifest", manifest=MANIFEST):
        if mode not in ("priority", "sbd") or groups not in ("manifest", "cell"):
            raise ValueError((mode, groups))
        self.rank = {x: i for i, x in enumerate(order)}
        self.mode, self.span_s, self.groups, self.manifest = mode, span_s, groups, manifest
        self.pkg = [g for g in param_groups(manifest).values()]
        self.rcp = {}                                      # knob -> (xapp, t, span)
        self.default = None
        self.stats = {"direct": 0, "indirect": 0}

    def _span(self, x):
        return self.span_s if self.span_s is not None else self.manifest.get(x, {}).get("span_s", 10.0)

    def _related(self, k1, k2):
        c1, c2 = {c for c, _ in knob_cells(k1)}, {c for c, _ in knob_cells(k2)}
        if not c1 & c2:
            return False
        return self.groups == "cell" or any(k1[0] in g and k2[0] in g for g in self.pkg)

    def conflicts(self, a, k, now):
        """-> [(b, "direct"|"indirect")] for the effective decisions of other xApps."""
        out = []
        for k2, (b, t, span) in self.rcp.items():
            if b == a or now - t >= span:
                continue
            if k2 == k:
                out.append((b, "direct"))
            elif self._related(k, k2):
                out.append((b, "indirect"))
        return out

    def __call__(self, obs):
        now, reqs = obs["t"], obs["requests"]
        if self.default is None:
            self.default = dict(obs["config"])
        self.rcp = {k: v for k, v in self.rcp.items() if now - v[1] < v[2]}
        dec = [None] * len(reqs)
        for i in sorted(range(len(reqs)), key=lambda j: self.rank.get(reqs[j]["xapp"], 99)):
            a, k = reqs[i]["xapp"], reqs[i]["knob"]
            cf = self.conflicts(a, k, now)
            for _, typ in cf:
                self.stats[typ] += 1
            ra = self.rank.get(a, 99)
            if not cf or all(ra < self.rank.get(b, 99) for b, _ in cf):
                dec[i] = "accept"
                self.rcp[k] = (a, now, self._span(a))
            elif self.mode == "sbd" and k in self.default and abs(self.default[k] - obs["config"][k]) > 1e-9:
                dec[i] = ("modify", float(self.default[k]))
            else:
                dec[i] = "reject"
        return {"decisions": dec, "writes": []}


# ------------------------------------------------------------------------------------------ PACIFISTA
def int_distance(a, b):
    """PACIFISTA Table 1 INT = sqrt( (1/L) * integral |F1(x) - F2(x)| dx ), L = max(x) - min(x) over both samples."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    xs = np.union1d(a, b)
    if len(xs) < 2:
        return 0.0
    f1 = np.searchsorted(a, xs, side="right") / len(a)
    f2 = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.sqrt(np.sum(np.abs(f1 - f2)[:-1] * np.diff(xs)) / (xs[-1] - xs[0])))


class Pacifista:
    """PACIFISTA (del Prever et al., TMC 2025): offline per-app profiles -> pairwise KPM severity sigma^K (INT distance
    of the apps' standalone ECDFs, aggregated by H = weighted mean) -> Sec. 8 deploy greedy: deploy every app without
    any conflict, then repeatedly take the highest-priority remaining app a and deploy it iff
    max_{a* in A_DPLY} sigma^K(a, a*) <= delta_TOL. The runtime arbiter is ``subset(A_DPLY)`` (the paper decides which
    apps run; it has no per-request rule). Conflict existence: MRO/TS share CIO (direct, Eq. 1); every E6 knob moves
    the shared load/PRB KPMs, so every pair has a KPM conflict (Eq. 5-6) [INFERRED beta graph]."""

    KSTAR = ("util", "ll_delay", "embb_thp", "ho_fail", "energy")

    def __init__(self, delta_tol=0.25, order=PRIORITY, kpms=KSTAR, weights=None, t_min=0.0):
        self.delta_tol, self.order, self.kpms, self.t_min = delta_tol, order, kpms, t_min
        self.w = np.asarray(weights if weights is not None else [1.0] * len(kpms), float)
        self.samples, self.sigma = {}, {}

    def fit(self, profiles):
        """profiles: {xapp: [env.log of an episode running that xApp ALONE, e.g. arbiter=subset((xapp,))]}."""
        for x, logs in profiles.items():
            s = {}
            for name in self.kpms:
                v = [kpi_of(r, name) for log in logs for e in log for r in e["reports"]
                     if r["gran"] == KPI_GRAN[name] and r["t0"] >= self.t_min]
                v = np.concatenate(v) if v else np.array([])
                s[name] = v[np.isfinite(v)]
            self.samples[x] = s
        for a, b in itertools.combinations(sorted(self.samples), 2):
            d = [int_distance(self.samples[a][n], self.samples[b][n])
                 if len(self.samples[a][n]) and len(self.samples[b][n]) else 0.0 for n in self.kpms]
            self.sigma[(a, b)] = self.sigma[(b, a)] = float(np.sum(self.w * d) / len(d))
        return self

    def deploy_set(self, apps):
        rank = {x: i for i, x in enumerate(self.order)}
        rest = sorted(apps, key=lambda x: rank.get(x, 99))
        conflicted = {a for a in rest for b in rest if a != b}   # KPM conflict between every pair (see class doc)
        dply = [a for a in rest if a not in conflicted]
        rest = [a for a in rest if a in conflicted]
        while rest:
            a = rest.pop(0)
            if not dply or max(self.sigma.get((a, b), 0.0) for b in dply) <= self.delta_tol:
                dply.append(a)
        return tuple(dply)

    def arbiter(self, apps):
        return subset(self.deploy_set(apps))


# ------------------------------------------------------------------------------------------ Djidjev & Kaminski
LOCAL_PARAMS = ("cio_out", "cio_in", "hys", "ttt", "ll_ratio", "carrier", "sleep")


def local_param(k, c):
    """Boolean parameter column of knob k at cell c (CIO split into out-/in-going; per-cell knobs by type)."""
    if k[0] == "cio":
        return "cio_out" if k[1] == c else "cio_in"
    return k[0]


_MASKS = np.array(list(itertools.product((0, 1), repeat=len(LOCAL_PARAMS))), bool)   # all 2^7 parent sets


def infer_row(BP, bK, cur=None):
    """2606.06663 Eq. (4): argmin_l || b_K - l (x)_B B_P ||_0 over the window, exhaustive over 2^n_P; ties -> minimum
    Hamming distance to the current row estimate (then the sparsest)."""
    pred = (BP.astype(int) @ _MASKS.T.astype(int)) > 0                   # (W, 2^n) Boolean OR model
    err = (pred != bK[:, None]).sum(0)
    ham = np.zeros(len(_MASKS)) if cur is None else (_MASKS != cur).sum(1)
    key = err * 1e6 + ham * 1e3 + _MASKS.sum(1)
    return _MASKS[int(np.argmin(key))].copy()


class Djidjev:
    """Djidjev & Kaminski, DETECTOR ONLY + a declared resolver (neither paper has a mitigation rule).

    Booleanisation (2606.06459 Eq. 4): B_P = 1 if a local parameter changed during the KPI's aligned window
    [previous report t0, this report t1) [INFERRED lag alignment]; B_K = 1{|z| > z_th} with z the one-report KPI
    change standardised by mean/SD over null rows (no local change) estimated in ``fit`` [local null rows: INFERRED,
    E6 has almost no globally quiet seconds]. Tracking (2606.06663 Alg. 1): per (kpi, cell) row keep the last W columns;
    if the window is full and (row never estimated or the newest column is not explained) recompute it with Eq. (4).
    Resolver [OURS, INFERRED, most-favourable reading]: reject a request if some OTHER xApp's KPI row on an acted-on
    cell has the knob's parameter as a tracked parent and that KPI currently violates its QoS; else accept."""

    def __init__(self, z_th=4.5, window=16, kpis=("ll_delay", "embb_thp", "ho_fail"), qos=QOS, manifest=MANIFEST):
        self.z_th, self.W, self.kpis, self.qos, self.manifest = z_th, window, kpis, qos, manifest
        self.mu, self.sd = {}, {}
        self.view = KPIView()
        self.last = {}                     # gran -> last processed report
        self.changes = {}                  # second -> set of knobs changed then
        self.prev_cfg = None
        self.win = {}                      # (kpi, cell) -> list of (bP, bK)
        self.L = {}                        # (kpi, cell) -> Boolean parent row over LOCAL_PARAMS
        self.alarms = 0

    @staticmethod
    def _bp(changes, t0, t1, nc):
        bp = np.zeros((nc, len(LOCAL_PARAMS)), bool)
        for s in range(int(np.ceil(t0)), int(np.ceil(t1))):
            for k in changes.get(float(s), ()):
                for c, _ in knob_cells(k):
                    bp[c, LOCAL_PARAMS.index(local_param(k, c))] = True
        return bp

    def fit(self, episodes):
        """Null-row mean / SD of the one-report KPI change per (kpi, cell) from burn-in logs."""
        acc = {}
        for log in episodes:
            changes, prev_cfg, last = {}, None, {}
            for e in log:
                cfg = e["config"]
                if prev_cfg is not None:
                    changes[e["t"]] = {k for k in cfg if abs(cfg[k] - prev_cfg[k]) > 1e-9}
                prev_cfg = cfg
                for r in sorted(e["reports"], key=lambda r: r["t1"]):
                    g, p = r["gran"], last.get(r["gran"])
                    last[g] = r if p is None or r["t1"] > p["t1"] else p
                    if p is None or r["t1"] <= p["t1"]:
                        continue
                    for name in self.kpis:
                        if KPI_GRAN[name] != g:
                            continue
                        dk = kpi_of(r, name) - kpi_of(p, name)
                        bp = self._bp(changes, p["t0"], r["t1"], len(dk))
                        for c in np.nonzero(np.isfinite(dk) & ~bp.any(1))[0]:
                            acc.setdefault((name, int(c)), []).append(dk[c])
        for key, v in acc.items():
            v = np.asarray(v)
            self.mu[key], self.sd[key] = float(v.mean()), float(max(v.std(ddof=1) if len(v) > 1 else 0.0, 1e-8))
        return self

    def observe(self, obs):
        now, cfg = obs["t"], obs["config"]
        if self.prev_cfg is not None:        # obs config at t shows the changes applied at the end of second t-1
            ch = {k for k in cfg if abs(cfg[k] - self.prev_cfg.get(k, cfg[k])) > 1e-9}
            if ch:
                self.changes[now - 1] = ch
        self.prev_cfg = dict(cfg)
        self.view.ingest(obs["new_reports"])
        for r in sorted(obs["new_reports"], key=lambda r: r["t1"]):
            g, p = r["gran"], self.last.get(r["gran"])
            if p is not None and r["t1"] <= p["t1"]:
                continue
            self.last[g] = r
            if p is None:
                continue
            for name in self.kpis:
                if KPI_GRAN[name] != g:
                    continue
                dk = kpi_of(r, name) - kpi_of(p, name)
                bp = self._bp(self.changes, p["t0"], r["t1"], len(dk))
                for c in range(len(dk)):
                    key = (name, c)
                    if not np.isfinite(dk[c]) or key not in self.mu:
                        continue
                    bk = abs((dk[c] - self.mu[key]) / self.sd[key]) > self.z_th
                    w = self.win.setdefault(key, [])
                    w.append((bp[c], bk))
                    del w[:-self.W]
                    if len(w) == self.W:
                        cur = self.L.get(key)
                        explained = cur is not None and bool((bp[c] & cur).any()) == bk
                        if not explained:
                            new = infer_row(np.array([x for x, _ in w]), np.array([y for _, y in w]), cur)
                            self.alarms += int(cur is not None and not np.array_equal(new, cur))
                            self.L[key] = new
        self.changes = {s: v for s, v in self.changes.items() if s >= now - 120}

    def __call__(self, obs):
        self.observe(obs)
        dec = []
        for r in obs["requests"]:
            a, k = r["xapp"], r["knob"]
            bad = False
            for b, d in self.manifest.items():
                if b == a or b not in obs["static"]["xapps"] or d["kpi"] not in self.kpis:
                    continue
                v = self.view.get(d["kpi"])
                for c, _ in knob_cells(k):
                    row = self.L.get((d["kpi"], c))
                    if row is not None and row[LOCAL_PARAMS.index(local_param(k, c))] and v is not None \
                            and violated(v[c], self.qos.get(d["kpi"])):
                        bad = True
            dec.append("reject" if bad else "accept")
        return {"decisions": dec, "writes": []}


# ------------------------------------------------------------------------------------------ interface-only methods
class Sharma:
    """Sharma et al., arXiv:2510.13031: per-KPI XGBoost on knob values -> mean|SHAP| -> DAG (per-KPI relative cutoff,
    back-derived tau_rel in (0.061, 0.077]) -> DoWhy backdoor ATE + CausalForestDML CATE per DAG edge.

    Contract: ``fit(episodes)`` on DEV logs with RANDOMISED knob dither (their data were randomised; closed-loop
    noarb logs are confounded) -> {(knob type, kpi): CATE(x)}; runtime resolver [OURS, INFERRED from Sec. V-C
    "suggest parameter adjustments to remain within acceptable operational tolerances"]: accept if every predicted
    child-KPI change CATE * (prop - cur) keeps the KPI within its QoS tolerance, else ("modify", largest safe step
    toward prop), else reject.
    TODO(E6-published): blocked on xgboost / dowhy / econml (not installed) and on the dither-episode data contract."""

    def fit(self, episodes):
        raise NotImplementedError("Sharma: needs xgboost + dowhy + econml and randomised-dither DEV logs")

    def __call__(self, obs):
        raise NotImplementedError


class TwoTower:
    """Santos et al., arXiv:2601.13213: supervised two-tower (per-node L-sample vector -> Linear-ReLU-Linear, H = 16,
    L2-normalised, S = alpha |Zp||Zk|^T, BCE vs a ground-truth label matrix Y) -> row sparsemax -> A_learned, + A_known
    (subscriptions) -> rule-based identification of [8]. Detector only.
    Contract: ``fit(episodes, Y)`` where Y must come from a separately seeded E6 twin / sandbox (E6 truth = upper
    bound only); resolver [OURS]: flag a request whose (xapp, knob) is in an identified conflict with an active xApp,
    then a fixed swept policy (defer / reject / priority).
    TODO(E6-published): label source for Y and the identification rules of [8] (not in the paper) are open."""

    def fit(self, episodes, Y):
        raise NotImplementedError("two-tower: needs a declared label matrix Y and the [8] identification rules")

    def __call__(self, obs):
        raise NotImplementedError


class Graphica:
    """Al Shami et al., arXiv:2503.03523 (GRAPHICA): per-timestamp binary state graphs -> 2-layer GCN + mean pool + FC
    -> {normal, direct, implicit, indirect}, focal loss; RCA = nodes with > 1 incoming edge. No mitigation.
    Contract: ``fit(episodes, labels)`` with labels from the paper's structural definitions; resolver [OURS]:
    classify the hypothetical post-accept state, if not normal accept the highest-priority root-cause xApp and
    reject / defer the rest.
    TODO(E6-published): hidden sizes / node features unreported; torch_geometric not installed; structural labels
    make it re-learn a deterministic rule in E6 (see doc)."""

    def fit(self, episodes, labels):
        raise NotImplementedError("GRAPHICA: architecture details unreported, labels undefined for E6")

    def __call__(self, obs):
        raise NotImplementedError
