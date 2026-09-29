"""Edge scoring of the E6-P discovery study (cdd_oran.decision.edge_score): TRUE u NULL confusion with INDET and
unmapped hypotheses excluded, sign accuracy on TP, the indirect (nbr) slice, far declarations, P1 / P2 rules and the
receiving-cell top-1 accuracy."""
from __future__ import annotations

import numpy as np

from cdd_oran.decision import edge_score as ES


def _ref():
    ref = {h: {"status": "NULL", "sign": 0} for h in ES.HYPOTHESES}
    ref[("sleep", "nbr", "load")] = {"status": "TRUE", "sign": 1}
    ref[("sleep", "nbr", "pv")] = {"status": "TRUE", "sign": 1}
    ref[("sleep", "nbr", "v")] = {"status": "TRUE", "sign": 1}
    ref[("ptx", "own", "e")] = {"status": "TRUE", "sign": 1}
    ref[("carrier", "own", "e")] = {"status": "INDET", "sign": 0}
    ref[("prot_min", "far", "pv")] = {"status": "INDET", "sign": 0}
    return ref


def _decl(on, sign=None):
    sign = sign or {}
    return {h: {"declared": h in on, "sign": sign.get(h, 1)} for h in ES.HYPOTHESES}


def test_confusion_counts_indet_and_unmapped():
    ref = _ref()
    d = _decl({("sleep", "nbr", "load"), ("sleep", "nbr", "pv"), ("ptx", "own", "e"), ("carrier", "own", "e"),
               ("ptx", "nbr", "load"), ("sleep", "far", "load")}, {("sleep", "nbr", "pv"): -1})
    d[("carrier", "own", "pv")] = {"declared": None, "sign": 0}             # unmapped
    m = ES.score_method(d, ref)
    ov, ind = m["overall"], m["indirect"]
    assert (ov["tp"], ov["fp"], ov["fn"]) == (3, 2, 1)                      # INDET carrier-own-e declared: ignored
    assert ov["n_indet"] == 2 and ov["n_unmapped"] == 1
    assert ov["precision"] == 3 / 5 and ov["recall"] == 3 / 4
    assert np.isclose(ov["f1"], 2 * 0.6 * 0.75 / 1.35)
    assert np.isclose(ov["sign_acc"], 2 / 3)
    assert (ind["tp"], ind["fp"], ind["fn"]) == (2, 1, 1)
    assert m["far_declared"] == 1 and m["far_fp"] == 1


def test_empty_declarations_and_nan_rules():
    m = ES.score_method(_decl(set()), _ref())
    assert np.isnan(m["overall"]["precision"]) and m["overall"]["recall"] == 0 and m["overall"]["f1"] == 0
    assert np.isnan(m["overall"]["sign_acc"])
    none_true = {h: {"status": "NULL", "sign": 0} for h in ES.HYPOTHESES}
    assert np.isnan(ES.confusion(_decl(set()), none_true)["recall"])


def test_p1_rules():
    ref = _ref()
    good = ES.score_method(_decl({("sleep", "nbr", "load"), ("sleep", "nbr", "pv"), ("ptx", "own", "e")}), ref)
    p1 = ES.p1_check(good)
    assert p1["pass"], p1
    far2 = ES.score_method(_decl({("sleep", "nbr", "load"), ("sleep", "nbr", "pv"), ("ptx", "own", "e"),
                                  ("sleep", "far", "e"), ("ptx", "far", "e")}), ref)
    p = ES.p1_check(far2)
    assert not p["pass"] and not p["parts"]["far_declarations"]
    wrong = ES.score_method(_decl({("sleep", "nbr", "load"), ("sleep", "nbr", "pv"), ("ptx", "own", "e")},
                                  {("ptx", "own", "e"): -1}), ref)
    assert not ES.p1_check(wrong)["parts"]["sign_accuracy"]                 # 2/3 < 0.90


def test_p2_pooled_and_two_of_three_folds():
    base = {"corr": {"pooled": 0.5, "fold0": 0.5, "fold1": 0.8, "fold2": 0.4},
            "int": {"pooled": 0.6, "fold0": float("nan"), "fold1": 0.1, "fold2": 0.1}}
    ok = ES.p2_check({"pooled": 0.6, "fold0": 0.5, "fold1": 0.7, "fold2": 0.4}, base)
    assert ok["pass"] and ok["folds_ok"] == 2 and ok["per_split"]["pooled"]["best_method"] == "int"
    assert not ES.p2_check({"pooled": 0.59, "fold0": 1, "fold1": 1, "fold2": 1}, base)["pass"]
    assert not ES.p2_check({"pooled": 0.9, "fold0": 0.4, "fold1": 0.7, "fold2": 0.4}, base)["pass"]


def test_top1_and_gt_receiving_and_physics_proxy():
    loc = {(1, 21): {"top1": 3}, (1, 22): {"top1": 5}, (2, 21): {"top1": None}, (3, 21): {"top1": 4}}
    gs = {1: {"cand": {"21": 3, "22": 0}}, 2: {"cand": {"21": 3}}, 3: {"cand": {}}}
    t = ES.top1_accuracy(loc, gs)
    assert (t["n"], t["hits"], t["no_localisation"]) == (2, 1, 1) and t["acc"] == 0.5
    rs = [{"cand": 3, "top1": 3, "top1_is_cand": True, "share_cand": 0.6},
          {"cand": 3, "top1": 2, "top1_is_cand": False, "share_cand": 0.2},
          {"cand": None, "top1": 2, "top1_is_cand": False, "share_cand": None}]
    g = ES.gt_receiving_summary(rs)
    assert g["n"] == 2 and g["rate"] == 0.5 and np.isclose(g["mean_share_cand"], 0.4)
    ref = ES.physics_reference()
    assert sum(v["status"] == "TRUE" for v in ref.values()) == len(ES.PHYSICS_PRIOR)
    assert all(ref[h]["status"] == "NULL" for h in ES.HYPOTHESES if h[1] == "far")
    tab = {"cells": [{"family": f, "relation": r, "kpi": k, "status": "NULL", "sign": 0} for f, r, k in ES.HYPOTHESES]}
    assert ES.gt_reference(tab)[("sleep", "own", "pv")]["status"] == "NULL"
