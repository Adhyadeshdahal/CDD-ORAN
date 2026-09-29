"""K-L0 (analysis only) of the step-2 regime referee (scratchpad/e6_dev/decision/STEP2_REFEREE_PLAN.md sec. 5).

  python scratchpad/e6_dev/e6p_referee_kl0.py [--json OUT] [--B 4000]

DEVELOPMENT data only: GT knockout labels runs/e6p-disc-gt-1, pi0 logs runs/e6p-disc-dev-1 + runs/e6p-disc-eval-1
(unit counts / leaf occupancy and a design-centred pi0 cross-check). Scale constants: runs/e6p-v2-summary.json
(P3 stratum 3 = surge-L40) and the AA (noarb) job records of runs/e6p-v2-s1 (per-episode prot_ue_s, viol_ue_s, rlf).

R units: Gate A v2 R = (V_AA - V_arm) / (V_AA - V_ref), V = 3600 * sum prot_viol / sum prot_ue_s (psvr, protected
violated seconds per protected UE-hour, pooled over seeds). A per-episode reduction of dP protected violated UE-s
changes V by 3600 * dP / prot_ue_s(AA, per episode) (prot_ue_s taken as unchanged by the referee), so
R_pred = 3600 * dP / prot_ue_s / (V_AA - V_ref) = dP / P_den with P_den = (V_AA - V_ref) * prot_ue_s / 3600.

Kill rule (plan sec. 5, primary beta = .07): R_pred < .35, or R_pred - R_pred(best global envelope) < .10, or
split-half sign agreement of the leaves < .9.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from e6p_disc_analyze import read_records  # noqa: E402

from cdd_oran.decision import crt_units as CU  # noqa: E402
from cdd_oran.decision import crt_units_v2 as CU2  # noqa: E402
from cdd_oran.decision import referee_p as RP  # noqa: E402
from cdd_oran.decision.collect_p import dec  # noqa: E402

RUNS = os.path.join(HERE, "runs")
GT = os.path.join(RUNS, "e6p-disc-gt-1", "all.jsonl")
DEV = os.path.join(RUNS, "e6p-disc-dev-1", "all.jsonl")
EVAL = os.path.join(RUNS, "e6p-disc-eval-1", "all.jsonl")
SUMMARY = os.path.join(RUNS, "e6p-v2-summary.json")
AA_JOBS = os.path.join(RUNS, "e6p-v2-s1", "all.jsonl")
BETAS = (0.03, 0.05, 0.07, 0.09)
PRIMARY_BETA = 0.07
CAP_FRAC = 0.05
KILL = {"R_min": 0.35, "margin_min": 0.10, "agree_min": 0.9}
for p in (GT, DEV, EVAL):
    assert not any(s in p for s in ("gtx", "ev2", "plx")), p            # sealed test data is never opened


def log(*a):
    print(*a, flush=True)


def constants() -> dict:
    S = json.load(open(SUMMARY, encoding="utf-8"))
    row = S["pairs"]["P3"]["rows"]["3"]
    arms = row["arms"]
    n_seeds = 8                                                       # E_kwh in the summary = pooled over 8 seeds
    aa = []
    with open(AA_JOBS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("arm") == "noarb" and r.get("pair") == "P3" and r.get("stratum") == 3 and r.get("stage") == "1":
                aa.append(r)
    assert len(aa) == n_seeds, len(aa)
    pu = float(np.mean([r["prot_ue_s"] for r in aa]))
    V_chk = 3600 * sum(r["prot_viol"] for r in aa) / sum(r["prot_ue_s"] for r in aa)
    assert abs(V_chk - row["V_AA"]) < 1e-6, (V_chk, row["V_AA"])
    S_A = (arms["freeze"]["E_kwh"] - arms["noarb"]["E_kwh"]) / n_seeds * 3.6e6
    den_psvr = row["V_AA_minus_V_ref"]
    return {"V_AA": row["V_AA"], "V_ref": row["V_ref"], "ref_arm": row["ref_arm"], "den_psvr": den_psvr,
            "Lambda": row["Lambda"], "prot_ue_s_AA": pu, "P_den": den_psvr * pu / 3600.0, "S_A_J": S_A,
            "E_saving_A_frac": row["E_saving_A_frac"], "v_AA": float(np.mean([r["viol_ue_s"] for r in aa])),
            "rlf_AA": float(np.mean([r["rlf"] for r in aa])), "pv_AA": float(np.mean([r["prot_viol"] for r in aa])),
            "R_static": row["R_static"], "R_static_arm": row["R_static_arm"],
            "R_qacm": arms["qacm"]["R"], "source": "runs/e6p-v2-summary.json P3/3 + runs/e6p-v2-s1 noarb jobs"}


# ---------------------------------------------------------------------------------------------- descriptives
def direction_counts(records) -> dict:
    """Per-episode unit counts by (family, direction) (all units) and 'fresh' gated units (no unit of the same
    (cell, knob, direction) in the previous 90 s: re-proposals after a pi0 reject are not fresh)."""
    n_ep = len(records)
    c, fresh = Counter(), Counter()
    for rec in records:
        last = {}
        for u in sorted(rec["units"], key=lambda u: u["t0"]):
            d = int(np.sign(u["step"]))
            name = {("carrier", -1): "carrier-off", ("carrier", 1): "carrier-on", ("sleep", 1): "sleep",
                    ("sleep", -1): "wake", ("ptx", -1): "ptx-down", ("ptx", 1): "ptx-up"}.get((u["knob"], d),
                                                                                          f"{u['knob']}{d:+d}")
            c[name] += 1
            key = (u["c"], u["knob"], d)
            if key not in last or u["t0"] - last[key] > 90:
                fresh[name] += 1
            last[key] = u["t0"]
    return {"per_episode": {k: v / n_ep for k, v in sorted(c.items())},
            "fresh_per_episode": {k: v / n_ep for k, v in sorted(fresh.items())}, "episodes": n_ep}


def gt_direction_effects(gt_recs, B=2000) -> list:
    """Paired accept-vs-reject effect of APPLYING each request type (network pv/e/v/rlf, own/nbr pv), all labels."""
    out = []
    groups = defaultdict(list)
    for rec in gt_recs:
        n = int(rec["n_cells"])
        for L in rec["gt_labels"]:
            if L["knob"] == "prot_min":
                continue
            d = int(np.sign(L["step"]))
            D = np.asarray(dec(L["delta"]), float).mean(0)            # (kpis, cells)
            kp = list(L["kpis"])
            own = [int(L["c"])]
            nbr = sorted(set(int(x) for x in L["exp"]) - set(own))
            vals = {k: float(D[kp.index(k)].sum()) for k in RP.EFFECT_KPIS}
            vals["pv_own"] = float(D[kp.index("pv"), own].sum())
            vals["pv_nbr"] = float(D[kp.index("pv"), nbr].sum())
            vals["pv_far"] = vals["pv"] - vals["pv_own"] - vals["pv_nbr"]
            assert n == D.shape[1]
            groups[(L["knob"], d)].append((int(rec["seed"]), vals))
    for (knob, d), lst in sorted(groups.items()):
        ep = np.array([s for s, _ in lst])
        row = {"knob": knob, "dir": d, "n": len(lst), "n_eps": int(len(np.unique(ep)))}
        for ki, k in enumerate(("pv", "pv_own", "pv_nbr", "pv_far", "e", "v", "rlf")):
            y = np.array([v[k] for _, v in lst])
            m, ci, _ = RP.cluster_boot(y, ep, np.random.default_rng([RP.REFEREE_TAG, 9, ki, d + 1]), B)
            row[k] = (m, ci)
        out.append(row)
    return out


def pi0_contrast(recs, B=2000) -> dict:
    """Design-centred pi0 cross-check (MSCR-v2 style, 'act' orientation): per gated family, beta of the network
    (own + nbr + far) post - pre 90 s KPI on v_u = LEVEL_V2[mode] - E_pi0[LEVEL_V2], residualised on episode FE +
    pre; episode-cluster bootstrap CI. Estimates APPLY vs reject under the pi0 continuation (not AA)."""
    data = CU.build_unit_data(recs)
    out = {}
    for fi, fam in enumerate(RP.GATED_FAMILIES):
        rows = data.rows_of(fam)
        rows = rows[np.array([RP.is_saving(fam, s) for s in data.step[rows]], bool)] if len(rows) else rows
        if len(rows) < 10:
            out[fam] = {"n": int(len(rows))}
            continue
        v = CU2.LEVEL_V2_ARR[data.mode[rows]] - CU2.design_mean(data, rows)
        ep = data.episode[rows]
        res = {"n": int(len(rows)), "n_eps": int(len(np.unique(ep)))}
        for ki, k in enumerate(RP.EFFECT_KPIS):
            y = sum(data.y[(r, k)][rows] for r in CU.RELATIONS)
            pre = sum(data.pre[(r, k)][rows] for r in CU.RELATIONS)
            r_, _ = CU2.residualise(y, pre, ep)
            beta = float(v @ r_ / (v @ v))
            eps, inv = np.unique(ep, return_inverse=True)
            num = np.bincount(inv, v * r_, len(eps))
            den = np.bincount(inv, v * v, len(eps))
            idx = np.random.default_rng([RP.REFEREE_TAG, 7, fi, ki]).integers(len(eps), size=(B, len(eps)))
            bb = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
            res[k] = (beta, tuple(float(x) for x in np.quantile(bb, RP.CI90)))
        out[fam] = res
    return out


# ---------------------------------------------------------------------------------------------- K-L0 core
def fit_trees(rows, swap: int, single: bool = False, all_eps=None) -> dict:
    trees = {}
    all_eps = all_eps if all_eps is not None else [r["ep"] for r in rows]
    for fam in RP.GATED_FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        sp, es = RP.split_halves(all_eps, swap)
        X = np.array([r["x"] for r in fr])
        y = np.array([r["y"]["pv"] for r in fr])
        ep = np.array([r["ep"] for r in fr])
        if single:
            trees[fam] = RP.HonestTree({}, RP.FEATURES, tuple(sorted(sp)), tuple(sorted(es)), "global envelope")
        else:
            trees[fam] = RP.fit_honest_tree(X, y, ep, sp, es)
    return trees


def nbar_of(trees, units, n_ep) -> dict:
    nb = Counter()
    for fam, tree in trees.items():
        fu = [u for u in units if u["family"] == fam]
        if not fu:
            continue
        for leaf in tree.leaf_of(np.array([u["x"] for u in fu])):
            nb[(fam, leaf)] += 1
    return {k: v / n_ep for k, v in nb.items()}


def evaluate(rows, units, n_ep, C, swap, single, B, betas=BETAS, all_eps=None) -> dict:
    trees = fit_trees(rows, swap, single, all_eps)
    effects = RP.effect_table(rows, trees, "est", B)
    nbar = nbar_of(trees, units, n_ep)
    res = {"trees": {f: {"leaves": t.leaves, "splits": t.describe(), "note": t.note} for f, t in trees.items()},
           "effects": [dict(e.__dict__) for e in effects], "nbar": {"|".join(k): v for k, v in nbar.items()},
           "beta": {}}
    for b in betas:
        ks = RP.knapsack(effects, nbar, b * C["S_A_J"], CAP_FRAC * C["v_AA"], CAP_FRAC * C["rlf_AA"])
        res["beta"][b] = {"R_pred": ks["obj"] / C["P_den"], "dP": ks["obj"], "defer": ["|".join(x) for x in ks["defer"]],
                          "e_J": ks["e"], "v": ks["v"], "rlf": ks["rlf"], "n_allowed": ks["n_allowed"]}
    ub = RP.knapsack(effects, nbar, np.inf, np.inf, np.inf, lb_rule=False)
    res["unconstrained_no_lb"] = {"R_pred": ub["obj"] / C["P_den"], "defer": ["|".join(x) for x in ub["defer"]],
                                  "e_J": ub["e"]}
    res["_trees"] = trees
    return res


def main(argv):
    B = int(argv[argv.index("--B") + 1]) if "--B" in argv else RP.N_BOOT
    C = constants()
    log(f"constants: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in C.items()})}")
    gt = read_records(GT, "gt")
    pi0 = read_records(DEV, "dev") + read_records(EVAL, "eval")
    n_ep = len(pi0)
    log(f"episodes: gt {len(gt)}, pi0 dev+eval {n_ep}")

    cnt_pi0, cnt_gt = direction_counts(pi0), direction_counts(gt)
    log(f"units / episode (pi0 dev+eval): {json.dumps({k: round(v, 2) for k, v in cnt_pi0['per_episode'].items()})}")
    log(f"  fresh (no same-direction unit in prior 90 s): "
        f"{json.dumps({k: round(v, 2) for k, v in cnt_pi0['fresh_per_episode'].items()})}")
    log(f"units / episode (gt eps): {json.dumps({k: round(v, 2) for k, v in cnt_gt['per_episode'].items()})}")

    dir_eff = gt_direction_effects(gt, B=2000)
    log("\nGT paired effect of APPLYING the request (accept - reject, 90 s, network unless noted; 90% CI):")
    for r in dir_eff:
        log(f"  {r['knob']:8s} dir {r['dir']:+d} n {r['n']:4d} eps {r['n_eps']:2d} | " + " ".join(
            f"{k} {r[k][0]:+9.1f} [{r[k][1][0]:+.1f},{r[k][1][1]:+.1f}]" for k in
            ("pv", "pv_own", "pv_nbr", "e", "v", "rlf")))

    rows = RP.label_rows(gt, dec)
    units = RP.unit_table(pi0)
    log(f"\ngated labels: {dict(Counter(r['family'] for r in rows))}; gated pi0 units: "
        f"{dict(Counter(u['family'] for u in units))} over {n_ep} episodes")

    pc = pi0_contrast(pi0)
    log(f"pi0 design-centred (apply vs reject, network, pi0 continuation): "
        f"{json.dumps(pc, default=lambda o: o)}")

    out = {"constants": C, "counts_pi0": cnt_pi0, "counts_gt": cnt_gt, "gt_direction_effects": dir_eff,
           "pi0_contrast": pc, "runs": {}}
    for swap in (0, 1):
        for single in (False, True):
            name = f"{'global' if single else 'referee'}_swap{swap}"
            out["runs"][name] = evaluate(rows, units, n_ep, C, swap, single, B, all_eps=[int(r["seed"]) for r in gt])
    agree = RP.sign_agreement(rows, {s: out["runs"][f"referee_swap{s}"]["_trees"] for s in (0, 1)}, ("pv",))
    agree_all = RP.sign_agreement(rows, {s: out["runs"][f"referee_swap{s}"]["_trees"] for s in (0, 1)},
                                  RP.EFFECT_KPIS)
    agree_sup = RP.sign_agreement(rows, {s: out["runs"][f"referee_swap{s}"]["_trees"] for s in (0, 1)}, ("pv",),
                                  supported_only=True)
    out["sign_agreement_pv"], out["sign_agreement_all"], out["sign_agreement_pv_supported"] = agree, agree_all, agree_sup
    for v in out["runs"].values():
        v.pop("_trees")

    for name in ("referee_swap0", "global_swap0", "referee_swap1", "global_swap1"):
        r = out["runs"][name]
        log(f"\n== {name}: trees {json.dumps(r['trees'])}")
        log(f"   nbar/ep {json.dumps({k: round(v, 2) for k, v in r['nbar'].items()})}")
        for e in r["effects"]:
            log(f"   {e['family']:8s} {e['leaf']:4s} n {e['n']:3d} eps {e['n_eps']:2d} sup {e['supported']!s:5s} | "
                "DEFER " + " ".join(f"{k} {e['defer'][k]:+9.1f} [{e['ci90'][k][0]:+.1f},{e['ci90'][k][1]:+.1f}] "
                                    f"(EB {e['shrunk'][k]:+.1f})" for k in RP.EFFECT_KPIS)
                + f" | benefit LB90 {e['benefit_lb90']:+.2f}")
        for b, x in r["beta"].items():
            log(f"   beta {b:.2f}: R_pred {x['R_pred']:+.3f} (dP {x['dP']:.1f} UE-s/ep) defer {x['defer']} "
                f"e {x['e_J'] / 1e3:.1f} kJ v {x['v']:+.0f} rlf {x['rlf']:+.2f} (allowed {x['n_allowed']})")
        u = r["unconstrained_no_lb"]
        log(f"   no budget, no LB rule: R_pred {u['R_pred']:+.3f} defer {u['defer']} e {u['e_J'] / 1e3:.1f} kJ")
    log(f"\nsplit-half sign agreement (pv): {agree['agree']:.3f} over {agree['n']} leaf comparisons; "
        f"all 4 KPIs: {agree_all['agree']:.3f} over {agree_all['n']}; pv, leaves supported in both halves: "
        f"{agree_sup['agree']:.3f} over {agree_sup['n']}")
    for p in agree["pairs"]:
        log(f"   {p}")

    verdict = {}
    for b in BETAS:
        rr = out["runs"]["referee_swap0"]["beta"][b]["R_pred"]
        rg = out["runs"]["global_swap0"]["beta"][b]["R_pred"]
        cond = {"R_pred<.35": rr < KILL["R_min"], "margin<.10": rr - rg < KILL["margin_min"],
                "agree<.9": not (agree["agree"] >= KILL["agree_min"])}
        verdict[b] = {"R_ref": rr, "R_global": rg, "margin": rr - rg, "conditions": cond, "kill": any(cond.values())}
        log(f"beta {b:.2f}: R_ref {rr:+.3f} R_global {rg:+.3f} margin {rr - rg:+.3f} kill-conditions {cond} -> "
            f"{'KILL' if any(cond.values()) else 'pass'}{'  [PRIMARY]' if b == PRIMARY_BETA else ''}")
    out["verdict"] = verdict
    out["K_L0"] = "KILL" if verdict[PRIMARY_BETA]["kill"] else "PASS"
    log(f"K-L0 (beta {PRIMARY_BETA}): {out['K_L0']}")
    if "--json" in argv:
        path = argv[argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        log(f"wrote {path}")
    return out


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    main(sys.argv[1:])
