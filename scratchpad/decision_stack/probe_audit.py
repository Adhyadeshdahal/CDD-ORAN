"""Carryover / actuation / placebo AUDIT of the DEV probe campaign (SOL_BUILD_REVIEW.md blocking item 2).

  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py pair --seed S [--scenario base --load medium
        --scored 1800 --flip 0] --out DIR      # base probe episode + same-seed REPLAY with block <flip>'s arm flipped
  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py ref --seed S [...] --out DIR    # no-probe reference
  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py probe --seed S [...] --out DIR   # probe episode only
  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py analyze --dir DIR [--rep 20]
  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py derive --dir DIR     # e6-probe/3 abort table (refs)
  PYTHONPATH=. python scratchpad/decision_stack/probe_audit.py recompute --dir DIR  # offline abort recompute

Replay = identical seed, config and schedule skeleton; only block <flip>'s arm changes (probe -> sham, or sham ->
cio +1 / carrier toggle). Any difference before that block's t0 must be exactly 0 (checked). Measures, per later
block k > flip: config divergence at t0_k (share of knobs whose value differs between the two runs), |delta| of its
pre- and post-window KPIs (own/nbr) relative to |delta| of the flipped block's own post window, and the time profile
of |delta| network PRB utilisation after the flipped block's restore deadline. Replays never enter inference.
analyze also reports blinded first-stage actuation/restoration by family x slot type x slot (probe.first_stage),
the sham-placebo rejection on real references (crt.placebo_rejection, a true sharp null), and the lag-1 CRT on the
probe episodes.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np

from cdd_oran.decision import crt
from cdd_oran.decision import probe as P
from cdd_oran.decision.trace import Trace
from cdd_oran.envs.e6 import config as C


def cfg_of(seed, scenario="base", load="medium", scored=P.PROBE_SCORED_S, warmup=120.0):
    return C.E6Config(seed=int(seed), load=load, mobility="mixed", mix="M4", warmup_s=float(warmup),
                      scored_s=float(scored), scenario=scenario)


def flip_of(block: dict) -> tuple:
    if block["arm"] != "sham":
        return ("sham", 0.0)
    return ("carrier", 1.0) if block["type"] == 1 else ("cio", 1.0)


def run_pair(cfg, flip=0, pcfg=None):
    pcfg = pcfg or P.ProbeConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = P.run_probe_episode(cfg, pcfg)
        if len(base.arrays["blk_t0"]) <= flip:
            raise ValueError("episode has no block to flip")
        a = base.arrays
        blk = {"arm": P.ARMS[int(a["blk_arm"][flip])], "type": int(a["blk_type"][flip])}
        rep = P.run_probe_episode(cfg, pcfg, override={flip: flip_of(blk)})
    return base, rep


def _fast_util(tr):
    a = tr.arrays
    return a["kpm_fast_t1"], np.asarray(a["kpm_fast_prb_util"], float).mean(1)


def contamination(base: Trace, rep: Trace, flip: int = 0) -> dict:
    a = base.arrays
    t0f, obs_end = float(a["blk_t0"][flip]), float(a["blk_obs_end"][flip])
    grace = base.meta["probe"]["config"]["restore_grace_s"]
    kb, kr = base.arrays["kpm_fast_t1"], rep.arrays["kpm_fast_t1"]
    n = min(len(kb), len(kr))
    tb, ub = _fast_util(base)
    _, ur = _fast_util(rep)
    d = np.abs(ub[:n] - ur[:n])
    pre_flip = float(d[tb[:n] <= t0f].max()) if (tb[:n] <= t0f).any() else 0.0
    ob, orp = P.block_outcomes(base), P.block_outcomes(rep)
    keys = sorted(ob["post"])
    eff0 = {k: abs(ob["post"][k][flip] - orp["post"][k][flip]) for k in keys}
    later = []
    for k in range(flip + 1, len(a["blk_t0"])):
        t0 = float(a["blk_t0"][k])
        cb, cr = base.config_at(int(t0)), rep.config_at(int(t0))
        later.append({"blk": k, "gap_s": t0 - (obs_end + grace), "arm": P.ARMS[int(a["blk_arm"][k])],
                      "config_diverged_share": float(np.mean(np.abs(cb - cr) > 1e-9)),
                      "d_pre": {q: float(abs(ob["pre"][q][k] - orp["pre"][q][k])) for q in keys},
                      "d_post": {q: float(abs(ob["post"][q][k] - orp["post"][q][k])) for q in keys}})
    prof = []
    rd = obs_end + grace
    for lo in range(0, 300, 30):
        m = (tb[:n] > rd + lo) & (tb[:n] <= rd + lo + 30)
        if m.any():
            prof.append((lo, float(d[m].mean())))
    m_obs = (tb[:n] > t0f) & (tb[:n] <= obs_end)
    return {"flip_arm": P.ARMS[int(a["blk_arm"][flip])], "flip_to": P.ARMS[int(rep.arrays["blk_arm"][flip])],
            "pre_flip_max_abs_diff": pre_flip, "eff_flip_block": eff0, "later": later,
            "util_absdiff_during_obs": float(d[m_obs].mean()) if m_obs.any() else np.nan,
            "util_absdiff_after_restore_30s_bins": prof}


def _load(pattern):
    return [Trace.from_npz(p) for p in sorted(glob.glob(pattern))]


def analyze(d, n_rep=20):
    out = {}
    pairs = sorted(glob.glob(os.path.join(d, "*_base.npz")))
    cont = []
    for pb in pairs:
        if not os.path.exists(pb.replace("_base.npz", "_replay.npz")):
            continue                            # probe-only re-audit: no replay
        base, rep = Trace.from_npz(pb), Trace.from_npz(pb.replace("_base.npz", "_replay.npz"))
        c = contamination(base, rep, int(rep.meta["probe"].get("flip", 0)))
        c["job"] = os.path.basename(pb)
        cont.append(c)
    out["replays"] = cont
    probes = [Trace.from_npz(p) for p in pairs]
    if probes:
        out["first_stage"] = P.first_stage(probes)
    refs = sorted(glob.glob(os.path.join(d, "*_ref.npz")))
    if refs:
        rt = [Trace.from_npz(p) for p in refs]
        src = [Trace.from_npz(p.replace("_ref.npz", "_base.npz")) for p in refs]
        pdata = crt.build_block_data(rt, schedule_from=src)
        small = crt.CRTConfig(B=199, nc=2, nb=2, min_stratum=3, min_f=2, min_sham=2, min_episodes=1)
        out["placebo"] = crt.placebo_rejection(pdata, n_rep=n_rep, cfg=small,
                                               outcomes=["own_prb_util", "own_embb_thp_p5", "own_energy_w",
                                                         "nbr_prb_util"])
        out["placebo"]["n_blocks"] = pdata.n
    if probes:
        data = crt.build_block_data(probes)
        small = crt.CRTConfig(B=199, nc=2, nb=2, min_stratum=3, min_f=2, min_sham=2, min_episodes=1)
        lag = crt.run_crt(data, outcomes=["own_prb_util", "pre_own_prb_util", "pre_nbr_prb_util"], cfg=small,
                          lag=1)
        out["lag1"] = [{k: r[k] for k in ("family", "outcome", "n_f", "n_sham", "p_mscr", "p_signed", "status")}
                       for r in lag["results"]]
    return out


def main(argv):
    cmd = argv[1]
    a = dict(zip(argv[2::2], argv[3::2], strict=True))
    if cmd == "derive":                    # abort thresholds from the NO-PROBE references only (*_ref.npz)
        refs = [Trace.from_npz(p) for p in sorted(glob.glob(os.path.join(a["--dir"], "*_ref.npz")))]
        tab = P.derive_abort_thresholds(refs)
        print(json.dumps({"table_id": P.ABORT_TABLE_ID, "n_refs": len(refs), "q": P.ABORT_Q, "theta": tab},
                         indent=1))
        return
    if cmd == "recompute":
        print(json.dumps(recompute_aborts(a["--dir"]), indent=1, default=float))
        return
    if cmd == "analyze":
        res = analyze(a["--dir"], int(a.get("--rep", 20)))
        print(json.dumps(res, indent=1, default=float))
        return
    cfg = cfg_of(a["--seed"], a.get("--scenario", "base"), a.get("--load", "medium"),
                 float(a.get("--scored", P.PROBE_SCORED_S)))
    os.makedirs(a["--out"], exist_ok=True)
    stem = os.path.join(a["--out"], f"{cfg.scenario}_{cfg.load}_s{cfg.seed}")
    if cmd == "pair":
        flip = int(a.get("--flip", 0))
        base, rep = run_pair(cfg, flip)
        rep.meta["probe"]["flip"] = flip
        base.to_npz(stem + "_base.npz")
        rep.to_npz(stem + "_replay.npz")
        print(json.dumps(contamination(base, rep, flip), default=float))
    elif cmd == "probe":                   # single probe episode (current ProbeConfig), saved as *_base.npz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            P.run_probe_episode(cfg).to_npz(stem + "_base.npz")
        print("ok", stem)
    elif cmd == "ref":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            P.run_reference(cfg).to_npz(stem + "_ref.npz")
        print("ok", stem)


# ------------------------------------------------------------------------------------------ e6-probe/3 abort recompute
def _table(stats_by_key: dict, q: float = P.ABORT_Q) -> dict:
    """stats_by_key: 'scenario|load' -> list of reference_window_stats outputs -> theta table (+ pooled row)."""
    pooled = {}
    tab = {}
    for key, lst in stats_by_key.items():
        for typ in (0, 1):
            row = {}
            for rule in P.ABORT_RULES:
                v = np.concatenate([w[typ][rule] for w in lst])
                pooled.setdefault((typ, rule), []).append(v)
                row[rule] = max(P.ABORT_FLOOR[rule], float(np.quantile(v, q, method="higher")))
            tab[f"{key}|{typ}"] = row
    for typ in (0, 1):
        tab[f"*|*|{typ}"] = {rule: max(P.ABORT_FLOOR[rule], float(np.quantile(np.concatenate(
            pooled[(typ, rule)]), q, method="higher"))) for rule in P.ABORT_RULES}
    return tab


def _fires(series, t0, obs_end, pre_s, theta):
    sc, _ = P.block_scores(series, t0, obs_end, pre_s)
    best = None
    for j, (rule, (_, k)) in enumerate(P.ABORT_RULES.items()):
        ft = P.fire_time(sc[rule], theta[rule], k)
        if ft is not None and (best is None or ft < best[0]):
            best = (ft, j)
    return best


def recompute_aborts(d: str) -> dict:
    """Offline recompute of e6-probe/3 abort decisions on EXISTING e6-probe/2 audit episodes (logged KPM only).

    Exactness: the old (logged) and new rules act identically on the trajectory until the first TREATED block whose
    abort decision differs (a sham abort restores nothing, so it never changes the trajectory). A block's new
    decision is EXACT iff every report it needs (up to its new fire time, else obs_end) was delivered no later than
    that divergence time; later blocks are INEXACT (need a re-run). Also reports in-sample reference window firing
    rates and leave-one-seed-out (LOSO) sham rates (thresholds re-derived without the episode's own reference)."""
    refs = sorted(glob.glob(os.path.join(d, "*_ref.npz")))
    pc = P.ProbeConfig()
    stats, keys = {}, {}
    for pth in refs:
        tr = Trace.from_npz(pth)
        c = tr.meta["cfg"]
        keys[pth] = f"{c.get('scenario', 'base')}|{c['load']}"
        stats[pth] = P.reference_window_stats(tr, pc)
    by_key = {}
    for pth, k in keys.items():
        by_key.setdefault(k, []).append(stats[pth])
    full = _table(by_key)
    ref_rate = {}
    for key, lst in by_key.items():
        for typ in (0, 1):
            th = full[f"{key}|{typ}"]
            fire = np.zeros(len(lst[0][typ]["ll"]) * 0 + sum(len(w[typ]["ll"]) for w in lst), bool)
            for rule in P.ABORT_RULES:
                fire |= np.concatenate([w[typ][rule] for w in lst]) > th[rule]
            ref_rate[f"{key}|{typ}"] = round(float(fire.mean()), 4)
    rows = []
    for pth in refs:
        base_p = pth.replace("_ref.npz", "_base.npz")
        if not os.path.exists(base_p):
            continue
        loso = _table({k: [stats[p] for p in refs if keys[p] == k and p != pth] or [stats[pth]]
                       for k in set(keys.values())})
        tr = Trace.from_npz(base_p)
        a, pm = tr.arrays, tr.meta["probe"]
        reps = P.trace_reports(tr)
        creg = np.asarray(tr.meta["cell_region"], int)
        c = tr.meta["cfg"]
        key = keys[pth]
        t_div = np.inf
        for slot in sorted(set(a["blk_slot"].tolist())):
            idx = np.nonzero(a["blk_slot"] == slot)[0]
            dec = []
            for i in idx:
                unit = pm["units"][str(int(a["blk_region"][i]))]
                cells = np.nonzero(np.isin(creg, unit))[0]
                ser = P.unit_series(reps, cells)
                t0, oe, typ = float(a["blk_t0"][i]), float(a["blk_obs_end"][i]), int(a["blk_type"][i])
                new = _fires(ser, t0, oe, pc.pre_s, full[f"{key}|{typ}"])
                new_l = _fires(ser, t0, oe, pc.pre_s, loso[f"{key}|{typ}"])
                old_t = float(a["blk_abort_t"][i]) if a["blk_abort"][i] else None
                h = new[0] if new else oe
                if old_t is not None and (new is None or new[0] > old_t):
                    h = np.inf                                      # needs reports after the logged restore
                dec.append((i, new, new_l, old_t, h))
            slot_div = t_div
            for i, new, _, old_t, _ in dec:
                treated = int(a["blk_arm"][i]) != P.ARMS.index("sham")
                differs = (old_t is None) != (new is None) or (old_t is not None and new[0] != old_t)
                if treated and differs:
                    slot_div = min(slot_div, min(old_t if old_t is not None else np.inf,
                                                 new[0] if new else np.inf))
            for i, new, new_l, old_t, h in dec:
                others = min([t_div] + [min(o if o is not None else np.inf, n[0] if n else np.inf)
                                        for j2, n, _, o, _ in dec if j2 != i and
                                        int(a["blk_arm"][j2]) != P.ARMS.index("sham") and
                                        ((o is None) != (n is None) or (o is not None and n[0] != o))])
                rows.append({"ep": os.path.basename(base_p), "stratum": key, "arm": P.ARMS[int(a["blk_arm"][i])],
                             "type": int(a["blk_type"][i]), "old_abort": old_t is not None,
                             "new_abort": new is not None, "new_reason": P.ABORT_REASONS[new[1]] if new else None,
                             "new_abort_loso": new_l is not None, "exact": bool(h <= others),
                             "applied": int(a["blk_n_applied"][i]), "knobs": int(a["blk_n_knobs"][i])})
            t_div = slot_div

    def summ(rs):
        ex = [r for r in rs if r["exact"]]
        return {"blocks": len(rs), "exact": len(ex), "old_abort_rate": round(np.mean([r["old_abort"] for r in rs]), 3)
                if rs else None,
                "new_abort_rate_exact": round(np.mean([r["new_abort"] for r in ex]), 3) if ex else None,
                "new_abort_rate_loso_exact": round(np.mean([r["new_abort_loso"] for r in ex]), 3) if ex else None,
                "new_reasons": {x: sum(r["new_reason"] == x for r in ex) for x in P.ABORT_REASONS},
                "effective_exact": sum((not r["new_abort"]) and (r["arm"] == "sham" or
                                                                 (r["knobs"] > 0 and r["applied"] >= r["knobs"] / 2))
                                       for r in ex)}

    return {"table_in_sample": {k: {r: round(v, 4) for r, v in t.items()} for k, t in full.items()},
            "reference_window_fire_rate_in_sample": ref_rate,
            "by_arm": {arm: summ([r for r in rows if r["arm"] == arm]) for arm in P.ARMS},
            "sham_by_type": {t: summ([r for r in rows if r["arm"] == "sham" and r["type"] == t]) for t in (0, 1)}}


if __name__ == "__main__":
    main(sys.argv)
