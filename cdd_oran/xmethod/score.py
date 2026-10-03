"""Scoring of one ``api.Result`` against its ``api.Truth`` (CONTRACT section 5), plus the placebo threshold rule.

Top-level metrics are computed over the PRIMARY set (ruling R-6): action -> KPI candidates whose source is not
``P_placebo`` / ``P_placebo_conf``; ``secondary`` (lagged KPI -> KPI) and ``all`` (both) repeat them. The tuning
placebo is reported separately (``placebo_declared``, ``placebo_fpr``) and so is the E4 R3/R4 diagnostic confounded
placebo (``placebo_conf_declared``, ``placebo_conf_fpr``; ruling R-10), so real-graph metrics do not depend on
either. Candidates the method did not return count as not declared (``n_missing``); an edge outside the candidate
set is an error.

Not-testable candidates (rulings R-22 / R-23): a candidate the method lists in ``Result.notes["not_testable_edges"]``
(or in ``notes["not_testable"]``; a list of "src->tgt" strings or (src, tgt) pairs, or a dict keyed by them; such
an edge has score NaN and p None) counts as NOT declared, whatever its ``declared`` flag (a true edge is a miss, a
null edge is no false positive). Every block reports ``n_not_testable_true`` / ``n_not_testable_null``; the record
also gives ``placebo_not_testable``, ``n_not_testable_overridden`` (listed edges flagged declared: an adapter bug)
and the sorted ``not_testable_edges``.

Definitions (R = declarations, V = declared nulls, TP = declared true edges, E = true edges, N = null edges):
  precision = TP / R (NaN if R = 0)        recall = TP / |E| (NaN if E empty)       F1 (NaN if undefined)
  fdp = V / max(R, 1)                      null_fpr = V / |N|
  sign_acc = correct / signed, over declared true edges with a truth sign and a nonzero method sign
             (NaN if none; ``sign_n`` gives the denominator)
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from cdd_oran.xmethod import api

PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"


def _div(a: float, b: float) -> float:
    return a / b if b else math.nan


def _prf(declared: set, edges: set, nulls: set) -> dict[str, Any]:
    tp, v = len(declared & edges), len(declared & nulls)
    r = len(declared)
    prec, rec = _div(tp, r), _div(tp, len(edges))
    if math.isnan(prec) or math.isnan(rec):
        f1 = math.nan
    else:
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n_declared": r, "tp": tp, "fp": v, "precision": prec, "recall": rec, "f1": f1,
            "fdp": v / max(r, 1), "null_fpr": _div(v, len(nulls))}


def not_testable_edges(result: api.Result) -> set[tuple[str, str]]:
    """The candidates a Result lists as not testable (module docstring); a malformed listing is an error."""
    out: set[tuple[str, str]] = set()
    for key in ("not_testable_edges", "not_testable"):
        v = (result.notes or {}).get(key)
        if v is None:
            continue
        if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
            raise ValueError(f"{result.method}: notes[{key!r}] must be a list or dict of edges, got {type(v).__name__}")
        for it in (v.keys() if isinstance(v, dict) else v):
            if isinstance(it, str):
                src, sep, tgt = it.partition("->")
                if not sep or not src or not tgt:
                    raise ValueError(f"{result.method}: bad not-testable edge {it!r} (want 'src->tgt')")
            else:
                src, tgt = it
            out.add((str(src), str(tgt)))
    return out


def score(result: api.Result, truth: api.Truth, candidates: Iterable[tuple[str, str]] | None = None,
          kpi_sources: Iterable[str] | None = None) -> dict[str, Any]:
    """Score ``result`` against ``truth``. ``candidates`` defaults to ``truth.edges | truth.null_edges``;
    ``kpi_sources`` (lagged-KPI source names) defaults to names starting with ``K``.

    Top level = the PRIMARY set (ruling R-6: action -> KPI candidates, placebo excluded and reported apart);
    ``secondary`` = lagged KPI -> KPI candidates; ``all`` = both."""
    cands = set(candidates) if candidates is not None else set(truth.edges) | set(truth.null_edges)
    by_edge = {}
    for e in result.edges:
        key = (e.source, e.target)
        if key not in cands:
            raise ValueError(f"{result.method}: edge {key} is not a candidate")
        if key in by_edge:
            raise ValueError(f"{result.method}: duplicate edge {key}")
        by_edge[key] = e
    ks = set(kpi_sources) if kpi_sources is not None else None
    is_kpi = (lambda s: s in ks) if ks is not None else (lambda s: s.startswith("K"))
    nt = not_testable_edges(result)
    if not nt <= cands:
        raise ValueError(f"{result.method}: not-testable edges outside the candidate set: {sorted(nt - cands)}")
    flagged = {k for k, e in by_edge.items() if e.declared}
    declared = flagged - nt                                 # R-23: not testable = not declared
    plac = {c for c in cands if c[0] == PLACEBO}
    conf = {c for c in cands if c[0] == PLACEBO_CONF}
    real = cands - plac - conf

    def block(sub: set) -> dict[str, Any]:
        edges, nulls = set(truth.edges) & sub, set(truth.null_edges) & sub
        out = _prf(declared & sub, edges, nulls)
        signed = [k for k in declared & edges if k in truth.signs and by_edge[k].sign != 0]
        correct = sum(int(np.sign(by_edge[k].sign) == truth.signs[k]) for k in signed)
        out.update(sign_acc=_div(correct, len(signed)), sign_n=len(signed), n_candidates=len(sub),
                   n_true=len(edges), n_null=len(nulls), n_missing=len(sub - set(by_edge)),
                   n_not_testable_true=len(nt & edges), n_not_testable_null=len(nt & nulls))
        return out

    out = block({c for c in real if not is_kpi(c[0])})
    pd = declared & plac
    out.update(placebo_declared=len(pd), placebo_fpr=_div(len(pd), len(plac)))
    cd = declared & conf
    out.update(placebo_conf_declared=len(cd), placebo_conf_fpr=_div(len(cd), len(conf)))
    out["secondary"] = block({c for c in real if is_kpi(c[0])})
    out["all"] = block(real)
    out.update(placebo_not_testable=len(nt & plac), n_not_testable_overridden=len(flagged & nt),
               not_testable_edges=sorted(f"{s}->{t}" for s, t in nt))
    out["declared_edges"] = sorted(f"{s}->{t}" for s, t in declared)
    return out


# ------------------------------------------------------------------------------- placebo threshold (section 5)
TAU_ALPHA = 0.05


def placebo_tau(results: Iterable[api.Result], max_declarations: int | None = None,
                alpha: float = TAU_ALPHA) -> float:
    """Threshold tau for ``score > tau`` declarations from the ``P_placebo`` edge scores of ``results`` (the tune
    datasets of one (arm, cell)); NaN scores are never counted nor declared. Truth is not used.

    Default (ruling R-29, conformal cutoff at level ``alpha``): with the M finite placebo scores sorted ascending,
    tau = the ceil((M + 1)(1 - alpha))-th smallest, or the largest if that index exceeds M; so a new placebo score
    exchangeable with them exceeds tau with probability <= alpha (<= 1 / (M + 1) in the capped case). M = 0 gives
    ``+inf`` (no calibration data: nothing declared).
    ``max_declarations=k`` (the superseded R-2 rule, kept for comparison): the smallest tau giving at most k placebo
    declarations = the (k + 1)-th largest score (ties fall below tau too); ``-inf`` if there are <= k scores."""
    s = sorted(e.score for r in results for e in r.edges
               if e.source == PLACEBO and e.score is not None and not math.isnan(e.score))
    if max_declarations is not None:
        return float(s[-1 - max_declarations]) if len(s) > max_declarations else -math.inf
    m = len(s)
    if m == 0:
        return math.inf
    k = math.ceil((m + 1) * (1.0 - alpha) - 1e-9)        # guard against (m + 1)(1 - alpha) = integer + rounding
    return float(s[min(k, m) - 1])


def tau_rule_name(alpha: float = TAU_ALPHA) -> str:
    """Label of the default ``placebo_tau`` rule, recorded with every tuned config."""
    return f"placebo_conformal_{alpha:g}"


def apply_threshold(result: api.Result, tau: float) -> api.Result:
    """Copy of ``result`` with ``declared = score > tau`` (NaN scores not declared); config records tau."""
    edges = tuple(api.EdgeResult(e.source, e.target, e.score, e.p, e.sign,
                                 bool(e.score is not None and not math.isnan(e.score) and e.score > tau))
                  for e in result.edges)
    return api.Result(method=result.method, version=result.version, edges=edges, cpu_s=result.cpu_s,
                      config={**result.config, "tau": tau}, notes=result.notes)
