"""``pc``: the PC algorithm (Spirtes, Glymour, Scheines 2000) via causal-learn (xm-classic, CONTRACT sec 3).

Implementation: ``causallearn.search.ConstraintBased.PC.pc`` pieces (skeleton_discovery, background-knowledge
orientation, uc_sepset, Meek), package causal-learn (version in ``Result.version``), defaults alpha = .05,
stable = True, uc_rule = 0, uc_priority = 2. CI test: Fisher-z (default) or KCI (``config['indep_test'] = 'kci'``,
causal-learn defaults incl. its gamma approximation; affordable only at small n, see the hand-back).

Variables: every action column (incl. ``P_placebo``) and every non-NaN lagged KPI at t, the observed context (R3), and
every KPI at t+1. Background knowledge (brief): actions are exogenous and time flows t -> t+1:
  tiers {actions, lagged KPIs, context} = 0 -> {KPIs at t+1} = 1 (no edge from t+1 back to t);
  no edge into an action from a lagged KPI or a KPI at t+1, and none from an action into a lagged KPI (so an action
  and a lagged KPI are never adjacent); context -> action is allowed (R3: the logged policy reads the context).
Node set (R-25): the actions at t, the arm's BASE set of the shared helper ``design_covariates`` (no focal: the
concurrent actions are the action nodes themselves) and the KPIs at t+1. NATIVE arm (``'native'``, R-18): base =
lagged KPIs (``lag_kpi:<K>``, kind L) + context (``ctx:<c>``, kind C). EQUAL-INFORMATION arm (``'eq'``, R-17,
PRIMARY, default): base adds the setpoints (``sp:<P>``) and the actions at t-1 / t-2 (``<P>@t-<L>``, ``gap@t-<L>``)
as EXOGENOUS tier-0 nodes (kind X); rows with ``row_mask`` False are dropped and an X column identical to an earlier
node (E1 / E3: a lagged action equal to a lagged KPI) is dropped (``notes['eq_dropped']``). Background knowledge
forbids every edge into X from an action, lagged KPI or KPI at t+1, allows X -> action (a setpoint sets its action),
X -> lagged KPI, X -> KPI and X -- context / X -- X; X nodes are never candidates, only possible separating sets.
R-37: the PRIMARY graph has no ``P_placebo_conf`` node (nor helper columns carrying it, e.g. ``P_placebo_conf@t-1``), so
the E4 R3 / R4 diagnostic never separates another pair; its own candidates are read from a second PC run with every
node (``notes['diagnostic_graph']``).
NOT TESTABLE (R-22, eq arm): a candidate whose target is an EXACT linear fit of the source's eq conditioning set
(``cond_set``; the deterministic worlds E1 / E3) gets score NaN, never declared, listed in ``notes['not_testable_edges']``
with ``notes['not_testable_reason']`` (same rule as granger's eq arm).
Readout: candidate source -> target is the (source at t, KPI at t+1) adjacency, which the tiers orient source ->
target. ``native`` = adjacent in PC's output at alpha.

Score (for the CONTRACT sec 5 tau rule): -log10 of the LARGEST CI p-value PC computed for the pair during the
skeleton search = pcalg's ``pMax`` (Kalisch, Maechler, Colombo, Maathuis, Buehlmann 2012, JSS 47(11): "the maximal
p-value over all CI tests" of the pair). In causal-learn's stable skeleton every conditioning set of the removal depth
is tested (no break), so a removed pair has pMax > alpha and a kept pair pMax <= alpha: native == score > -log10 alpha. The tau rule then works
like a larger / smaller alpha on the recorded tests (not a re-run of the search; recorded deviation).
Sign: unsigned method -> orchestrator rule ``pcorr_given_Z``.
Singular correlation sub-matrices (deterministic worlds) make causal-learn's Fisher-z raise; that test is then
recomputed with the pseudo-inverse and the same formula (count and rate per dataset in
``notes['fisherz_pinv_fallbacks']`` / ``notes['fisherz_pinv_fallback_rate']`` = fallbacks / CI tests; for an exactly
singular set the partial correlation is undefined and the pinv value is one choice, so report the rate per cell).
"""
from __future__ import annotations

import importlib.metadata
from math import log, sqrt
from typing import Any

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from scipy.stats import norm

from cdd_oran.xmethod import api

from ._classic_common import (
    PLACEBO_CONF,
    ClassicBase,
    Scored,
    columns,
    cond_set,
    design_covariates,
    exact_fit,
    is_diagnostic_column,
    not_testable_notes,
    resolve,
    sign_pcorr_given_Z,
    source_column,
    target_index,
)


class _RecordingCIT:
    """Wraps a causal-learn CIT: records the max p-value per unordered pair; pinv fallback for singular Fisher-z."""

    def __init__(self, cit, data: np.ndarray):
        self.cit = cit
        self.method = cit.method
        self.maxp: dict[tuple[int, int], float] = {}
        self.n_tests = 0
        self.fallbacks = 0
        self._corr = None
        self._n = data.shape[0]
        self._data = data

    def _fisherz_pinv(self, X, Y, S):
        if self._corr is None:
            self._corr = np.corrcoef(self._data.T)
        var = [int(X), int(Y)] + [int(s) for s in S]
        inv = np.linalg.pinv(self._corr[np.ix_(var, var)])
        den = sqrt(abs(inv[0, 0] * inv[1, 1]))
        r = 0.0 if den == 0 else -inv[0, 1] / den
        if abs(r) >= 1:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        Z = 0.5 * log((1 + r) / (1 - r))
        stat = sqrt(max(self._n - len(S) - 3, 1)) * abs(Z)
        return float(2 * (1 - norm.cdf(abs(stat))))

    def __call__(self, X, Y, S=None, *args):
        S = () if S is None else tuple(S)
        try:
            p = float(self.cit(X, Y, S))
        except ValueError:
            if self.method != "fisherz":
                raise
            self.fallbacks += 1
            p = self._fisherz_pinv(X, Y, S)
        self.n_tests += 1
        key = (min(int(X), int(Y)), max(int(X), int(Y)))
        self.maxp[key] = max(self.maxp.get(key, -1.0), p)
        return p


def run_pc(D: np.ndarray, names: list[str], tiers: list[int], forbidden: set[tuple[int, int]], alpha: float,
           indep_test: str, stable: bool, uc_rule: int, uc_priority: int, max_k: int | None = None, **cit_kwargs):
    """causal-learn PC with tier + forbidden-edge background knowledge; returns (cg, recorder)."""
    rec = _RecordingCIT(CIT(D, indep_test, **cit_kwargs), D)
    # GraphNode equality is by name, so nodes of a throw-away CausalGraph identify the search graph's nodes
    nodes = CausalGraph(D.shape[1], names).G.nodes
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], t)
    for i, j in forbidden:
        bk.add_forbidden_by_node(nodes[i], nodes[j])
    cg = SkeletonDiscovery.skeleton_discovery(D, alpha, rec, stable, background_knowledge=bk, verbose=False,
                                              show_progress=False, node_names=names, max_k=max_k)
    orient_by_background_knowledge(cg, bk)
    if uc_rule != 0:
        raise ValueError("only uc_rule = 0 (causal-learn default) is wired")
    cg = UCSepset.uc_sepset(cg, uc_priority, background_knowledge=bk)
    cg = Meek.meek(cg, background_knowledge=bk)
    return cg, rec


class PC(ClassicBase):
    name = "pc"
    version = f"causal-learn {importlib.metadata.version('causal-learn')}"
    uses_p = False
    method_idx = 1
    arms = ("eq", "native")
    defaults: dict[str, Any] = {"alpha": 0.05, "indep_test": "fisherz", "stable": True, "uc_rule": 0,
                                "uc_priority": 2, "max_k": None, "arm": "eq"}

    def _graph(self, data: api.Dataset, cols, config: dict[str, Any], hn, hk, H, mask, with_conf: bool) -> dict:
        """Build the node set and run PC. ``with_conf`` False (R-37, primary graph): no ``P_placebo_conf`` node and
        no helper column carrying it; True: every node (read only for ``P_placebo_conf``'s own candidates)."""
        names, mats, kinds = [], [], []
        for i, a in enumerate(data.action_names):
            if a == PLACEBO_CONF and not with_conf:
                continue
            names.append(f"A:{a}")
            mats.append(np.asarray(data.X_action, float)[mask, i])
            kinds.append("A")
        eq_dropped = []
        for i, (nm, k) in enumerate(zip(hn, hk, strict=True)):
            if not with_conf and is_diagnostic_column(nm):
                continue
            c = H[:, i]
            if k == "X" and any(np.array_equal(c, m) for m in mats):
                eq_dropped.append(nm)
                continue
            names.append(f"{k}:{nm}")
            mats.append(c)
            kinds.append(k)
        for i, kp in enumerate(data.kpi_names):
            names.append(f"Y:{kp}")
            mats.append(cols.Y[mask, i])
            kinds.append("Y")
        D = np.column_stack(mats)
        keep = [i for i in range(D.shape[1]) if np.std(D[:, i]) > 0]
        idx = {names[i]: k for k, i in enumerate(keep)}
        D, kn, kk = D[:, keep], [names[i] for i in keep], [kinds[i] for i in keep]
        tiers = [1 if k == "Y" else 0 for k in kk]
        forbidden = set()
        for i, ki in enumerate(kk):
            for j, kj in enumerate(kk):
                if i == j:
                    continue
                if kj == "A" and ki in ("L", "Y", "A"):      # only context (and eq-arm X) into an action
                    forbidden.add((i, j))
                if ki == "A" and kj in ("L", "C", "X"):      # an action cannot cause the state it was set in
                    forbidden.add((i, j))
                if kj == "X" and ki in ("L", "Y"):           # design covariates are exogenous (eq arm)
                    forbidden.add((i, j))
        cg, rec = run_pc(D, kn, tiers, forbidden, config["alpha"], config["indep_test"], config["stable"],
                         config["uc_rule"], config["uc_priority"], config.get("max_k"))
        return {"idx": idx, "G": cg.G.graph, "rec": rec, "n_vars": D.shape[1], "eq_dropped": eq_dropped,
                "dropped_constant": [names[i] for i in range(len(names)) if i not in keep]}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        eq = config["arm"] == "eq"
        hn, H, mask = design_covariates(data, eq, eq)          # the arm's base set (R-25), row order
        if not eq:
            mask = np.ones(data.n, bool)
        H = np.asarray(H, float)[mask]
        hk = ["L" if nm.startswith("lag_kpi:") else "C" if nm.startswith("ctx:") else "X" for nm in hn]
        main = self._graph(data, cols, config, hn, hk, H, mask, with_conf=False)
        diag = (self._graph(data, cols, config, hn, hk, H, mask, with_conf=True)
                if any(s == PLACEBO_CONF for s, _ in data.candidates) else None)
        out, exact = [], []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            g = diag if s == PLACEBO_CONF else main
            sname = None if j is None else (f"A:{data.action_names[j]}" if fam == "action"
                                            else f"L:lag_kpi:{cols.lag_names[j]}")
            tname = f"Y:{data.kpi_names[ti]}"
            if sname not in g["idx"] or tname not in g["idx"]:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            if eq:
                cs = cond_set(data, fam, source_column(data, cols, fam, j)[1], "eq")
                if exact_fit(cols.Y[cs.mask, ti], cs.Z[cs.mask]):
                    exact.append(f"{s}->{t}")
                    out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                    continue
            a, b = g["idx"][sname], g["idx"][tname]
            mp = g["rec"].maxp.get((min(a, b), max(a, b)))
            score = float("nan") if mp is None else float(-np.log10(max(mp, 1e-300)))
            adj = bool(g["G"][a, b] != 0 or g["G"][b, a] != 0)
            out.append(Scored(s, t, fam, score, sign_pcorr_given_Z(data, cols, fam, j, ti, config["arm"]), None, adj))
        rec = main["rec"]
        notes = {"sign_rule": "pcorr_given_Z", "n_ci_tests": rec.n_tests, "fisherz_pinv_fallbacks": rec.fallbacks,
                 "fisherz_pinv_fallback_rate": rec.fallbacks / rec.n_tests if rec.n_tests else 0.0,
                 "n_vars": main["n_vars"], "dropped_constant": main["dropped_constant"],
                 "score": "-log10 max CI p over the pair's skeleton tests (pcalg pMax)"}
        if diag is not None:
            notes["diagnostic_graph"] = {"rule": f"R-37: {PLACEBO_CONF} candidates read from a second PC run with "
                                                 "every node; primary graph without it",
                                         "n_vars": diag["n_vars"], "n_ci_tests": diag["rec"].n_tests}
        if eq:
            notes.update({"eq_covariates": [nm for nm, k in zip(hn, hk, strict=True)
                                            if k == "X" and not is_diagnostic_column(nm)],
                          "eq_dropped": main["eq_dropped"],
                          "n_rows_used": int(mask.sum()), **not_testable_notes(exact)})
        return out, notes
