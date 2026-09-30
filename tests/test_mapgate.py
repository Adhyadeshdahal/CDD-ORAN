"""cdd_oran.decision.mapgate: ctx-only policy, decision table on synthetic maps, GT-map implications."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from cdd_oran.decision import mapgate as MG
from cdd_oran.decision.units_p import UnitArbiter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_JSONS = {"gt_ext": (os.path.join(ROOT, "scratchpad", "e6_dev", "decision", "step1_v2_full.json"), ("gt",)),
            "gt1": (os.path.join(ROOT, "scratchpad", "e6_dev", "decision", "step1_v1_gt_tables.json"), ("dir",))}


def ctx(f, step, own=0.0, nbr=0.0):
    return {"knob": f, "step": float(step), "own_prot_below_frac": own, "nbr_max_prot_below_frac": nbr}


class OnlyCtx(dict):
    """A unit that raises on any key but "ctx"."""

    def __getitem__(self, k):
        if k != "ctx":
            raise AssertionError(f"MapGate read unit[{k!r}]")
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        raise AssertionError(f"MapGate read unit.get({k!r})")


def test_reads_only_ctx():
    g = MG.MapGate({("sleep", "nbr", "pv"): 1.0, ("sleep", "nbr", "e"): 1.0})
    assert g(OnlyCtx(ctx=ctx("sleep", 1))) == ("reject", 1.0)
    assert g(OnlyCtx(ctx=ctx("sleep", -1))) == ("accept", 1.0)
    assert g.n == {"defer_sleep": 1, "accept_sleep": 1}


@pytest.mark.parametrize("M, f, step, own, nbr, defer", [
    # harm and costly -> defer regardless of pressure
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, "x", 1, 0.0, 0.0, True),
    # other direction: no harm -> accept
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, "x", -1, 1.0, 1.0, False),
    # harm but saving energy -> defer iff the harmful relation's pressure > theta
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, "x", 1, 0.04, 0.0, False),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, "x", 1, 0.06, 0.0, True),
    # pressure is read on the HARMFUL relation only (own harmful, nbr pressure ignored)
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, "x", 1, 0.0, 0.9, False),
    ({("x", "nbr", "pv"): 2.0, ("x", "own", "e"): -5.0}, "x", 1, 0.9, 0.0, False),
    ({("x", "nbr", "pv"): 2.0, ("x", "own", "e"): -5.0}, "x", 1, 0.0, 0.9, True),
    # sum over own + nbr: cancelling edges -> no harm
    ({("x", "own", "pv"): 2.0, ("x", "nbr", "pv"): -3.0}, "x", 1, 1.0, 1.0, False),
    # no energy edge -> 0 >= 0 -> costly
    ({("x", "own", "pv"): 1.0}, "x", 1, 0.0, 0.0, True),
    # far edges are ignored
    ({("x", "far", "pv"): 9.0, ("x", "far", "e"): 9.0}, "x", 1, 1.0, 1.0, False),
    # fallback to v when no pv edge is declared; a declared-None pv blocks the fallback
    ({("x", "own", "v"): 3.0}, "x", 1, 0.0, 0.0, True),
    ({("x", "own", "v"): 3.0, ("x", "own", "pv"): None}, "x", 1, 0.0, 0.0, False),
    # zero step -> accept; NaN pressure -> 0
    ({("x", "own", "pv"): 1.0}, "x", 0, 1.0, 1.0, False),
    ({("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0}, "x", 1, math.nan, math.nan, False),
    # empty map -> accept everything
    ({}, "x", 1, 1.0, 1.0, False),
])
def test_decision_table_synthetic(M, f, step, own, nbr, defer):
    assert MG.MapGate(M, 0.05).decide(ctx(f, step, own, nbr))[0] is defer


def test_theta_edge_and_builders():
    M = {("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0, ("x", "nbr", "pv"): 0.5, ("x", "far", "v"): 1.0}
    assert not MG.MapGate(M, 0.05).decide(ctx("x", 1, own=0.05))[0]            # strict >
    assert MG.MapGate(M, 0.0).decide(ctx("x", 1, own=1e-6))[0]
    assert MG.own_only(M) == {("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0}
    assert MG.sign_flip(M, ("x",))[("x", "nbr", "pv")] == -0.5
    cells = [{"family": "x", "relation": "own", "kpi": "pv", "status": "TRUE", "mean": 2.0},
             {"family": "x", "relation": "own", "kpi": "e", "status": "NULL", "mean": 0.1},
             {"family": "x", "relation": "nbr", "kpi": "pv", "status": "INDET", "mean": 9.0}]
    assert MG.map_from_gt(cells) == {("x", "own", "pv"): 2.0, ("x", "own", "e"): None}
    assert MG.map_from_gt(cells, true_only=True) == {("x", "own", "pv"): 2.0}


def _gt_map(name):
    path, keys = GT_JSONS[name]
    if not os.path.exists(path):
        pytest.skip(f"{path} absent")
    d = json.load(open(path))
    for k in keys:
        d = d[k]
    return MG.map_from_gt(d["cells"], true_only=True)


@pytest.mark.parametrize("name", list(GT_JSONS))
def test_gt_map_implications(name):
    M = _gt_map(name)
    T = MG.decision_table(M, 0.05)
    assert T[("sleep", 1)] == "defer"                       # sleep -> nbr pv (+), nbr e (+): deferred
    assert T[("sleep", -1)] == "accept"                     # wake accepted
    assert T[("carrier", 1)] == "accept"                    # carrier-on accepted
    assert T[("carrier", -1)].startswith("defer iff own")   # carrier-off saves energy: only under own pressure
    assert T[("ptx", -1)] == "accept"                       # ptx-down accepted
    assert T[("ptx", 1)] == "defer"                         # ptx-up: own pv (+), own e (+) -> deferred
    assert T[("prot_min", 1)] == "accept" and T[("prot_min", -1)] == "defer"
    # the wrong maps change the sleep (own-only) and carrier (sign-flip) rows
    To = MG.decision_table(MG.own_only(M), 0.05)
    assert To[("sleep", 1)] == "accept" and To[("sleep", -1)] == "defer"
    Tf = MG.decision_table(MG.sign_flip(M, ("carrier",)), 0.05)
    assert Tf[("carrier", 1)].startswith("defer iff own") and Tf[("carrier", -1)] == "accept"
    assert {k: v for k, v in Tf.items() if k[0] != "carrier"} == {k: v for k, v in T.items() if k[0] != "carrier"}


def test_gt1_and_gt_ext_agree():
    assert MG.decision_table(_gt_map("gt1")) == MG.decision_table(_gt_map("gt_ext"))


STATIC = {"cells": 3, "neighbours": [[1], [0], [1]], "is_macro": np.array([True, True, False]),
          "knobs": [("carrier", 0), ("carrier", 1), ("sleep", 2), ("prot_min", 0), ("ptx", 0)],
          "xapps": {"ES": [], "SliceGuarantee": [], "PowerES": []}}


def _obs(t, reqs, cfg):
    return {"t": float(t), "requests": reqs, "config": dict(cfg), "new_reports": [], "static": STATIC,
            "locked": {}, "changes": 0, "churn_cap": None}


def test_unit_arbiter_from_t0_and_hold():
    M = {("sleep", "nbr", "pv"): 9.0, ("sleep", "nbr", "e"): 4.0, ("sleep", "own", "pv"): -0.7}
    arb = MG.mapgate_arbiter(M, record=True)
    assert isinstance(arb, UnitArbiter) and arb.warmup_s == 0.0 and arb.T == 60.0 and arb.open_rule == "feasible"
    cfg = {("carrier", 0): 2.0, ("carrier", 1): 2.0, ("sleep", 2): 0.0, ("prot_min", 0): 0.1, ("ptx", 0): 40.0}
    sl = {"xapp": "ES", "ver": 1, "knob": ("sleep", 2), "cur": 0.0, "prop": 1.0, "t": 0.0}
    out = arb(_obs(0, [sl], cfg))                           # t = 0: active immediately (no warm-up)
    assert out["decisions"] == ["reject"] and arb.units[0]["mode"] == "reject"
    assert arb(_obs(30, [sl], cfg))["decisions"] == ["reject"]          # held for the unit
    assert arb(_obs(61, [dict(sl, cur=1.0, prop=0.0)], cfg))["decisions"] == ["accept"]   # new unit: wake accepted
