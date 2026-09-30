"""Data-efficiency BENCH for E6-P causal discovery methods (MSCR+ research; runner scratchpad/e6_dev/disc_bench_run.py).

1. CACHE (``build_cache``): STREAMS "e6p-disc-rec/1" episode records (one episode at a time; only compact per-unit
   arrays are kept) into one compressed npz per source file. One row per logged unit of a knob family in
   ``crt_units.FAMILIES`` (every unit, whatever its window; per-H validity flags). Columns (schema ``CACHE_SCHEMA``):
     unit table   ep (episode index within the file), c, t0 (rounded, int), mode / applied (index into
                  crt_units.MODES), p, probs (n, 4) = the logged pi0 table of the unit's xApp, step, sgn (sign(step),
                  0 -> +1), family (index into FAMILIES), xapp (str);
     exposure     exp (n, C) bool = the unit's logged obs-only exposure set N(c) (own = {c}, nbr = N(c) - {c},
                  far = not N(c), as gt_p.relation_cells);
     KPI sums     post_H{H} / pre_H{H} (n, K, C) float64: per cell, KPI (crt_units.KPIS order: pv, v, e, rlf, load=ue)
                  summed over rows t0 .. t0+H-1 / t0-H .. t0-1 (crt_units windows), for H in ``HS`` = 30, 60, 90, 150;
                  zero where that window does not fit (t0 + H > T / t0 - H < 0; the loader applies the window rule).
                  float64 + the same numpy slicing as build_unit_data: bit-identical sums (float32 storage of post and
                  pre separately lost ~1e-3 of y through cancellation and moved CRT p-values);
     ctx          ctx (n, J) float64 + ctx_num (n, J) bool (numeric in the record, crt_units._num) over ctx_keys;
     episodes     ep_seed, ep_sub, ep_fold (-1 none), ep_stage, ep_policy, ep_smoke, ep_n_units_logged, ep_T, src.
   Privileged keys (gt_static, gt_labels, lab_outcome) are never read.
2. LOADER (``load_pool`` + ``Pool.unit_data``): concatenates cache files for one (H, H_pre) and rebuilds a
   ``crt_units.UnitData`` for any episode subset, identical (bit for bit, tests) to ``crt_units.build_unit_data`` on
   the same records: episodes sorted by (seed, sub) as the analyzers' read_records, units in record order, the
   window rule of H / H_pre, prev over kept units, z = numeric ctx keys present in the subset + pre_{rel}_{kpi}.
   Existing methods (crt_units_v2.run_crt_units_v2, baselines_disc) run unchanged.
3. BENCH (``bench``): for n in sizes, R seeded disjoint episode subsets of the pool (``draw_subsets``; n = 300 uses
   the four v3 EVAL folds + one random 300 of the other episodes when the pool has them); a METHOD =
   callable(data: UnitData, split: int) -> {(f, rel, k): {"declared", "sign", "p" (optional)}} (edge_score's
   format) is run on each subset and scored against the GT reference (``gt_reference_files``: gt_p.summarize over
   the pooled GT episodes, "dir", frozen rule): chain-set hits (``CHAIN``), premise-edge detection (sleep -> nbr pv
   declared +), indirect recall / precision / F1, overall precision, sign accuracy, #declared, far declared ->
   mean / SD over the R subsets. PLACEBO check (``placebo_check``): the method on each placebo unit table: #declared
   (BY) and the per-hypothesis rejection rate at alpha (share of finite p <= alpha).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np

from .collect_p import dec
from .crt_units import FAMILIES, KPIS, MODES, RELATIONS, SERIES_OF, UnitData, _num
from .edge_score import gt_reference, score_method

CACHE_SCHEMA = "disc-bench-cache/1"
REC_SCHEMA = "e6p-disc-rec/1"
HS = (30, 60, 90, 150)
CHAIN = (("sleep", "nbr", "load", 1), ("sleep", "nbr", "pv", 1), ("sleep", "nbr", "v", 1), ("ptx", "nbr", "load", -1))
PREMISE = ("sleep", "nbr", "pv")
SIZES = (60, 120, 300)
BENCH_TAG = 6690                                  # subset-draw RNG tag (scratch; bench only)


# ------------------------------------------------------------------------------------------------ streaming io
def iter_records(paths, stages=None):
    """Yield parsed episode records (schema e6p-disc-rec/1) one at a time from JSONL ``paths``."""
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if r.get("kind") != "episode" or r.get("schema") != REC_SCHEMA:
                    continue
                if stages is not None and r.get("stage") not in stages:
                    continue
                yield r


def episode_arrays(rec, Hs=HS, kpis=KPIS) -> dict:
    """Compact per-unit arrays of one episode record (module docstring, 1)."""
    nc = int(rec["n_cells"])
    ls = rec["lab_series"]
    D = dec(ls["data"]).astype(np.float64)                                # (T, F, C), as build_unit_data
    fidx = [list(ls["fields"]).index(SERIES_OF[k]) for k in kpis]
    T = D.shape[0]
    table = rec.get("pi0_table")
    us = [u for u in rec["units"] if u["knob"] in FAMILIES]
    n = len(us)
    out = {"c": np.zeros(n, np.int16), "t0": np.zeros(n, np.int32), "mode": np.zeros(n, np.int8),
           "applied": np.zeros(n, np.int8), "p": np.zeros(n), "probs": np.zeros((n, len(MODES))),
           "step": np.zeros(n), "sgn": np.zeros(n), "family": np.zeros(n, np.int8),
           "xapp": np.array([str(u["x"]) for u in us], dtype="<U24"), "exp": np.zeros((n, nc), bool)}
    for H in Hs:
        out[f"post_H{H}"] = np.zeros((n, len(kpis), nc))
        out[f"pre_H{H}"] = np.zeros((n, len(kpis), nc))
    ctx = []
    for i, u in enumerate(us):
        t0 = int(round(float(u["t0"])))
        step = float(u.get("step", u["ctx"].get("step", 0.0)))
        if u.get("probs") is not None:             # per-unit logged row preferred (as crt_units.build_unit_data)
            pr = np.array([float(u["probs"].get(m, 0.0)) for m in MODES])
        elif table is not None:
            tab = table[u["x"]]
            pr = np.array([float(tab.get(m, 0.0)) for m in MODES])
        else:
            pr = np.array([1.0 if m == u["mode"] else 0.0 for m in MODES])
        out["c"][i], out["t0"][i] = int(u["c"]), t0
        out["mode"][i] = MODES.index(u["mode"])
        out["applied"][i] = MODES.index(u.get("applied_mode", u["mode"]))
        out["p"][i], out["probs"][i], out["step"][i] = float(u["p"]), pr, step
        out["sgn"][i] = float(np.sign(step)) or 1.0
        out["family"][i] = FAMILIES.index(u["knob"])
        out["exp"][i, [int(v) for v in u["exp"]]] = True
        for H in Hs:                               # same slicing / summation as build_unit_data (bit-identical)
            if t0 + H <= T and t0 >= 0:
                out[f"post_H{H}"][i] = D[t0:t0 + H][:, fidx, :].sum(0)
            if t0 - H >= 0:
                out[f"pre_H{H}"][i] = D[t0 - H:t0][:, fidx, :].sum(0)
        ctx.append({k: (float(v) if _num(v) else None) for k, v in u["ctx"].items()})
    out["_ctx"] = ctx
    out["_ep"] = {"seed": int(rec["seed"]), "sub": str(rec.get("sub") or ""),
                  "fold": -1 if rec.get("fold") is None else int(rec["fold"]), "stage": str(rec.get("stage")),
                  "policy": str(rec.get("policy")), "smoke": bool(rec.get("smoke")),
                  "n_units_logged": len(rec["units"]), "T": T, "n_cells": nc}
    return out


def build_cache(paths, out_path: str, stages=None, Hs=HS, src: str = "", log=None) -> dict:
    """Stream ``paths`` (JSONL) into one compressed npz at ``out_path`` (module docstring, 1). Episodes are deduped by
    record key (last wins, as read_records). Returns a small summary."""
    t = time.time()
    parts, seen = [], {}
    for rec in iter_records(paths, stages):
        key = json.dumps(rec.get("key")) + str(bool(rec.get("smoke")))
        a = episode_arrays(rec, Hs)
        del rec
        if key in seen:
            parts[seen[key]] = a
        else:
            seen[key] = len(parts)
            parts.append(a)
    if not parts:
        raise ValueError(f"no episode records (stages {stages}) in {paths}")
    ncs = {p["_ep"]["n_cells"] for p in parts}
    if len(ncs) != 1:
        raise ValueError(f"mixed cell counts {ncs}")
    keys = sorted({k for p in parts for c in p["_ctx"] for k in c})
    kx = {k: j for j, k in enumerate(keys)}
    arrs = {}
    for name in [k for k in parts[0] if not k.startswith("_")]:
        arrs[name] = np.concatenate([p[name] for p in parts])
    arrs["ep"] = np.concatenate([np.full(len(p["c"]), e, np.int32) for e, p in enumerate(parts)])
    n = len(arrs["ep"])
    ctx = np.full((n, len(keys)), np.nan)
    num = np.zeros((n, len(keys)), bool)
    i = 0
    for p in parts:
        for c in p["_ctx"]:
            for k, v in c.items():
                if v is not None:
                    ctx[i, kx[k]] = v
                    num[i, kx[k]] = True
            i += 1
    arrs["ctx"], arrs["ctx_num"], arrs["ctx_keys"] = ctx, num, np.array(keys, dtype="<U64")
    for f in ("seed", "sub", "fold", "stage", "policy", "smoke", "n_units_logged", "T"):
        arrs[f"ep_{f}"] = np.array([p["_ep"][f] for p in parts])
    arrs["src"] = np.array(src or os.path.basename(str(paths)))
    arrs["schema"] = np.array(CACHE_SCHEMA)
    arrs["Hs"] = np.array(Hs, np.int32)
    arrs["n_cells"] = np.array(ncs.pop(), np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    s = {"out": out_path, "episodes": len(parts), "units": n, "mb": round(os.path.getsize(out_path) / 2 ** 20, 1),
         "wall_s": round(time.time() - t, 1), "stages": sorted(set(arrs["ep_stage"].tolist())),
         "subs": sorted(set(arrs["ep_sub"].tolist()))}
    if log:
        log(f"cache {s}")
    return s


# ------------------------------------------------------------------------------------------------ loader
@dataclasses.dataclass
class Pool:
    """Concatenated cache files for one (H, H_pre). Episode ids are global (0 .. n_eps-1)."""
    H: int
    H_pre: int
    n_cells: int
    kpis: tuple
    eps: dict                   # seed, sub, fold, stage, policy, smoke, n_units_logged, T, src, file  (n_eps,)
    u: dict                     # per-unit arrays; u["gep"] = global episode id; post / pre (n, K, C) float64
    ctx_keys: list

    @property
    def n_eps(self) -> int:
        return len(self.eps["seed"])

    def episodes(self, mask=None) -> np.ndarray:
        ids = np.arange(self.n_eps)
        return ids if mask is None else ids[np.asarray(mask, bool)]

    def unit_data(self, episodes=None, relations=RELATIONS, sort: bool = True) -> UnitData:
        """UnitData of the given global episode ids (default all), episode index = position after sorting by
        (seed, sub) (``sort``; as the analyzers' read_records) - identical to build_unit_data on those records."""
        E = self.eps
        ids = np.arange(self.n_eps) if episodes is None else np.asarray(episodes, int)
        if sort:
            ids = np.array(sorted(ids.tolist(), key=lambda e: (int(E["seed"][e]), str(E["sub"][e]))), int)
        pos = np.full(self.n_eps, -1)
        pos[ids] = np.arange(len(ids))
        u = self.u
        sel = pos[u["gep"]] >= 0
        rows_all = np.nonzero(sel)[0]
        rows_all = rows_all[np.argsort(pos[u["gep"][rows_all]], kind="stable")]   # episode order, record order
        t0 = u["t0"][rows_all].astype(int)
        T = E["T"][u["gep"][rows_all]]
        keep = (t0 - self.H_pre >= 0) & (t0 + self.H <= T)
        rows = rows_all[keep]
        n, C = len(rows), self.n_cells
        ep = pos[u["gep"][rows]]
        # prev: previous kept unit of the same (episode, c, xApp, knob)
        prev = np.full(n, -1, int)
        last = {}
        c_, x_, f_ = u["c"][rows], u["xapp"][rows], u["family"][rows]
        for i in range(n):
            key = (int(ep[i]), int(c_[i]), str(x_[i]), int(f_[i]))
            prev[i] = last.get(key, -1)
            last[key] = i
        exp = u["exp"][rows]
        own = np.zeros((n, C), bool)
        own[np.arange(n), c_.astype(int)] = True
        rel_all = {"own": own, "nbr": exp & ~own, "far": ~exp}
        masks = {r: rel_all[r] for r in relations}
        post = u["post"][rows].astype(np.float64)
        pre_c = u["pre"][rows].astype(np.float64)
        ycell = {k: post[:, j, :] - pre_c[:, j, :] for j, k in enumerate(self.kpis)}
        precell = {k: pre_c[:, j, :] for j, k in enumerate(self.kpis)}
        y = {(r, k): (ycell[k] * masks[r]).sum(1) for r in relations for k in self.kpis}
        pre = {(r, k): (precell[k] * masks[r]).sum(1) for r in relations for k in self.kpis}
        num = u["ctx_num"][rows]
        cv = u["ctx"][rows]
        z = {f"ctx_{k}": np.where(num[:, j], cv[:, j], np.nan) for j, k in enumerate(self.ctx_keys)
             if num[:, j].any()}
        z.update({f"pre_{r}_{k}": pre[(r, k)] for r in relations for k in self.kpis})
        mode = u["mode"][rows].astype(int)
        probs = u["probs"][rows]
        p_mismatch = int(np.sum(np.abs(probs[np.arange(n), mode] - u["p"][rows]) > 1e-6))
        epmeta = [{"seed": int(E["seed"][e]), "stage": str(E["stage"][e]), "fold": int(E["fold"][e]),
                   "sub": str(E["sub"][e]), "policy": str(E["policy"][e]), "smoke": bool(E["smoke"][e]),
                   "n_units_logged": int(E["n_units_logged"][e]), "src": str(E["src"][e])} for e in ids]
        fold = E["fold"][u["gep"][rows]].astype(int)
        return UnitData(episode=ep.astype(int), seed=E["seed"][u["gep"][rows]].astype(int), fold=fold,
                        family=f_.astype(int), xapp=x_.astype(object), c=c_.astype(int), t0=u["t0"][rows].astype(int),
                        mode=mode, applied=u["applied"][rows].astype(int), p=u["p"][rows].astype(float),
                        probs=probs.astype(float), step=u["step"][rows].astype(float),
                        sgn=u["sgn"][rows].astype(float), y=y, pre=pre, ycell=ycell, rel_mask=masks, z=z, prev=prev,
                        n_cells=C, H=self.H, H_pre=self.H_pre, relations=tuple(relations), kpis=self.kpis,
                        meta={"episodes": epmeta, "dropped": {"window": int((~keep).sum()), "family": None},
                              "p_mismatch": p_mismatch, "n_units": n, "conditioners": list(z),
                              "global_episodes": ids.tolist()})


def load_pool(paths, H: int = 90, H_pre: int | None = None, kpis=KPIS, stages=None) -> Pool:
    """Concatenate cache npz ``paths`` (list or comma string) for one H / H_pre (module docstring, 2). ``stages``:
    keep only episodes of these record stages (e.g. {"eval"}; None = all). An episode (seed, sub, stage, smoke) found
    in several files is kept once (first file)."""
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    H = int(H)
    H_pre = H if H_pre is None else int(H_pre)
    kidx = [list(KPIS).index(k) for k in kpis]
    eps = {k: [] for k in ("seed", "sub", "fold", "stage", "policy", "smoke", "n_units_logged", "T", "src", "file")}
    U = {k: [] for k in ("gep", "c", "t0", "mode", "applied", "p", "probs", "step", "sgn", "family", "xapp", "exp",
                         "post", "pre", "ctx", "ctx_num")}
    ctx_keys, n_cells, off = None, None, 0
    all_keys = []
    loaded = []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        if str(z["schema"]) != CACHE_SCHEMA:
            raise ValueError(f"{path}: schema {z['schema']}")
        if H not in z["Hs"] or H_pre not in z["Hs"]:
            raise ValueError(f"{path}: H {H} / H_pre {H_pre} not cached ({z['Hs'].tolist()})")
        nc = int(z["n_cells"])
        if n_cells is None:
            n_cells = nc
        elif nc != n_cells:
            raise ValueError("cache files with different cell counts cannot be pooled")
        loaded.append(z)
        all_keys.extend(z["ctx_keys"].tolist())
    ctx_keys = sorted(set(all_keys))
    for path, z in zip(paths, loaded, strict=True):
        ne = len(z["ep_seed"])
        for k in ("seed", "sub", "fold", "stage", "policy", "smoke", "n_units_logged", "T"):
            eps[k].append(z[f"ep_{k}"])
        eps["src"].append(np.full(ne, str(z["src"])))
        eps["file"].append(np.full(ne, os.path.basename(path)))
        U["gep"].append(z["ep"].astype(np.int64) + off)
        for k in ("c", "t0", "mode", "applied", "p", "probs", "step", "sgn", "family", "xapp", "exp"):
            U[k].append(z[k])
        post = z[f"post_H{H}"][:, kidx, :]
        pre = z[f"pre_H{H_pre}"][:, kidx, :]
        U["post"].append(post)
        U["pre"].append(pre)
        keys = z["ctx_keys"].tolist()
        n = len(z["ep"])
        cx = np.full((n, len(ctx_keys)), np.nan)
        cn = np.zeros((n, len(ctx_keys)), bool)
        j = [ctx_keys.index(k) for k in keys]
        cx[:, j], cn[:, j] = z["ctx"], z["ctx_num"]
        U["ctx"].append(cx)
        U["ctx_num"].append(cn)
        off += ne
        z.close()
    eps = {k: np.concatenate(v) for k, v in eps.items()}
    U = {k: np.concatenate(v) for k, v in U.items()}
    seen, keep = set(), np.ones(len(eps["seed"]), bool)            # an episode in two files: the first file wins
    for e, key in enumerate(zip(eps["seed"].tolist(), eps["sub"].tolist(), eps["stage"].tolist(),
                                eps["smoke"].tolist(), strict=True)):
        keep[e] = key not in seen and (stages is None or key[2] in stages)
        seen.add(key)
    if not keep.all():
        new = np.cumsum(keep) - 1
        ku = keep[U["gep"]]
        U = {k: v[ku] for k, v in U.items()}
        U["gep"] = new[U["gep"]]
        eps = {k: v[keep] for k, v in eps.items()}
    return Pool(H=H, H_pre=H_pre, n_cells=int(n_cells), kpis=tuple(kpis), eps=eps, u=U, ctx_keys=ctx_keys)


# ------------------------------------------------------------------------------------------------ GT reference
def gt_reference_files(paths) -> dict:
    """gt_p.summarize over ALL GT episodes of ``paths`` pooled (stage "gt"; "dir", frozen rule). Returns {"ref",
    "counts", "n_episodes", "n_labels", "labels_per_family", "nbr_true", "seeds", "cells"}."""
    from .gt_p import PRIMARY_ORIENT, summarize
    eps = []
    for r in iter_records(paths, {"gt"}):
        eps.append({"seed": r["seed"], "n_cells": r["n_cells"], "gt_static": r["gt_static"],
                    "gt_labels": [dict(L, delta=dec(L["delta"])) for L in r["gt_labels"]], "sub": r.get("sub")})
    uniq = {(e["seed"], e.get("sub")): e for e in eps}
    eps = [uniq[k] for k in sorted(uniq, key=lambda k: (k[0], str(k[1])))]
    gs = summarize(eps)
    table = gs["tables"][PRIMARY_ORIENT]
    return {"ref": gt_reference(table), "counts": table["counts"], "n_episodes": len(eps),
            "n_labels": gs["n_labels"], "labels_per_family": gs["labels_per_family"],
            "seeds": [int(e["seed"]) for e in eps], "delta": table["delta"],
            "nbr_true": sorted((c["family"], c["kpi"], c["sign"]) for c in table["cells"]
                               if c["relation"] == "nbr" and c["status"] == "TRUE"),
            "cells": table["cells"]}


def save_ref(g: dict, path: str) -> None:
    d = dict(g, ref={"|".join(h): v for h, v in g["ref"].items()})
    with open(path, "w", newline="\n") as fh:
        json.dump(d, fh, indent=1, default=_js)


def load_ref(path: str) -> dict:
    g = json.load(open(path))
    g["ref"] = {tuple(k.split("|")): v for k, v in g["ref"].items()}
    return g


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ------------------------------------------------------------------------------------------------ subsets
def draw_subsets(pool: Pool, n: int, R: int, seed: int = 0, folds_300: bool = True) -> list:
    """R disjoint episode subsets of size n (global ids), seeded ``default_rng([BENCH_TAG, n, R, seed])``.
    n == 300 and ``folds_300`` and the pool has the four EVAL v3 folds of 300 -> those 4 folds (+ R - 4 random disjoint
    300s of the remaining episodes). Returns [{"name", "episodes", "split"}]; split = the RNG split id for methods
    (v3 fold k -> 1 + k, like the analyzers; random subsets 20 + i)."""
    rng = np.random.default_rng([BENCH_TAG, int(n), int(R), int(seed)])
    E = pool.eps
    out, used = [], np.zeros(pool.n_eps, bool)
    if n == 300 and folds_300:
        v3 = (E["sub"] == "v3") & (E["stage"] == "eval")
        fs = [np.nonzero(v3 & (E["fold"] == k))[0] for k in range(4)]
        if all(len(f) == 300 for f in fs):
            for k, f in enumerate(fs[:R]):
                out.append({"name": f"v3fold{k}", "episodes": f, "split": 1 + k})
                used[f] = True
    rest = rng.permutation(np.nonzero(~used)[0])
    i = 0
    while len(out) < R and (i + 1) * n <= len(rest):
        out.append({"name": f"rand{i}", "episodes": np.sort(rest[i * n:(i + 1) * n]), "split": 20 + i})
        i += 1
    if len(out) < R:
        raise ValueError(f"pool of {pool.n_eps} episodes cannot hold {R} disjoint subsets of {n}")
    return out


# ------------------------------------------------------------------------------------------------ scoring
def subset_metrics(decl: dict, ref: dict) -> dict:
    sc = score_method(decl, ref)
    c_true, hits = 0, 0
    members = {}
    for f, r, k, s in CHAIN:
        g = ref.get((f, r, k), {"status": "INDET", "sign": 0})
        d = decl.get((f, r, k)) or {}
        ok_gt = g["status"] == "TRUE" and int(g["sign"]) == s
        hit = bool(d.get("declared")) and int(d.get("sign", 0)) == s
        c_true += ok_gt
        hits += ok_gt and hit
        members["|".join((f, r, k))] = {"gt_true": ok_gt, "hit": hit, "p": d.get("p"), "z": d.get("z_approx")}
    prem = decl.get(PREMISE) or {}
    ind, ov = sc["indirect"], sc["overall"]
    return {"chain_hits": hits, "chain_true": c_true,
            "premise_hit": float(bool(prem.get("declared")) and int(prem.get("sign", 0)) == 1),
            "ind_recall": ind["recall"], "ind_precision": ind["precision"], "ind_f1": ind["f1"],
            "ind_tp": ind["tp"], "ind_fp": ind["fp"], "ov_precision": ov["precision"], "ov_recall": ov["recall"],
            "ov_f1": ov["f1"], "sign_acc": ov["sign_acc"], "n_declared": sc["n_declared"],
            "far_declared": sc["far_declared"], "chain": members}


METRICS = ("chain_hits", "premise_hit", "ind_recall", "ind_precision", "ind_f1", "ov_precision", "ov_f1", "sign_acc",
           "n_declared", "far_declared")


def _agg(rows: list) -> dict:
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows], float)
        f = v[np.isfinite(v)]
        out[m] = {"mean": float(f.mean()) if len(f) else float("nan"),
                  "sd": float(f.std(ddof=1)) if len(f) > 1 else float("nan"), "n_finite": int(len(f)),
                  "min": float(f.min()) if len(f) else float("nan"), "max": float(f.max()) if len(f) else float("nan")}
    return out


def placebo_check(method, data: UnitData, split: int = 9, alpha: float = 0.05) -> dict:
    """``method`` on a placebo unit table: #declared and the per-hypothesis rejection rate at alpha (finite p)."""
    t = time.time()
    decl = method(data, split)
    ps = [float(v["p"]) for v in decl.values() if v and v.get("p") is not None and np.isfinite(float(v["p"]))]
    return {"n_declared": sum(1 for v in decl.values() if v and v.get("declared")),
            "declared": sorted("|".join(h) for h, v in decl.items() if v and v.get("declared")),
            "n_p": len(ps), "n_reject": int(sum(p <= alpha for p in ps)),
            "rate": float(np.mean([p <= alpha for p in ps])) if ps else float("nan"),
            "min_p": float(min(ps)) if ps else float("nan"), "units": int(data.n), "wall_s": round(time.time() - t, 1)}


def bench(pool: Pool, method, ref: dict, sizes=SIZES, R: dict | int | None = None, seed: int = 0,
          placebos: dict | None = None, log=None, keep_decl: bool = False) -> dict:
    """Run ``method`` on R disjoint subsets per size and score against ``ref`` (module docstring, 3). ``R``: int or
    {n: R}; default {60: 10, 120: 10, 300: 5}. ``placebos``: {name: UnitData}."""
    Rm = {60: 10, 120: 10, 300: 5}
    if isinstance(R, int):
        Rm = {n: R for n in sizes}
    elif isinstance(R, dict):
        Rm.update(R)
    res = {"sizes": {}, "pool_episodes": pool.n_eps, "H": pool.H, "H_pre": pool.H_pre}
    for n in sizes:
        subs = draw_subsets(pool, n, Rm.get(n, 10), seed)
        rows = []
        for s in subs:
            t = time.time()
            data = pool.unit_data(s["episodes"])
            decl = method(data, s["split"])
            m = subset_metrics(decl, ref)
            m.update(name=s["name"], split=s["split"], units=int(data.n), wall_s=round(time.time() - t, 1),
                     subs=sorted(set(pool.eps["sub"][s["episodes"]].tolist())))
            if keep_decl:
                m["declared"] = sorted(("|".join(h), int(v.get("sign", 0))) for h, v in decl.items()
                                       if v and v.get("declared"))
            rows.append(m)
            if log:
                log(f"  n={n} {s['name']:>8}: chain {m['chain_hits']}/{m['chain_true']} premise {m['premise_hit']:.0f} "
                    f"ind R {m['ind_recall']:.2f} P {m['ind_precision']:.2f} F1 {m['ind_f1']:.2f} ovP "
                    f"{m['ov_precision']:.2f} sign {m['sign_acc']:.2f} #dec {m['n_declared']} ({m['wall_s']} s)")
            del data
        res["sizes"][str(n)] = {"R": len(rows), "summary": _agg(rows), "subsets": rows}
        if log:
            a = res["sizes"][str(n)]["summary"]
            log(f"n={n}: " + ", ".join(f"{k} {a[k]['mean']:.3f}+-{a[k]['sd']:.3f}" for k in METRICS))
    if placebos:
        res["placebo"] = {nm: placebo_check(method, d) for nm, d in placebos.items()}
        if log:
            log(f"placebo: { {k: (v['n_declared'], round(v['rate'], 3)) for k, v in res['placebo'].items()} }")
    return res


# ------------------------------------------------------------------------------------------------ methods
def method_mscr_v2(B: int = 9999, **cfg_kw):
    """MSCR-CRT v2 (crt_units_v2.run_crt_units_v2, default config but B) as a bench method."""
    from . import crt_units_v2 as V2
    cfg = V2.UnitCRTConfigV2(B=int(B), **cfg_kw)

    def m(data, split):
        return V2.edges_from_crt_v2(V2.run_crt_units_v2(data, cfg, split))
    m.label = f"mscr_crt_v2(B={B})"
    return m


def method_baseline(name: str, dev: UnitData):
    """DEV-tuned (far-FPR tau, baselines_disc.tune_tau) corr / granger, or untuned granger_by. Declarations carry
    "p" for granger (placebo rejection rate); corr has no p."""
    from . import baselines_disc as BD
    if name == "granger_by":
        def m(data, split):
            sc = BD.granger_pvalues(data)
            d = BD.granger_by(sc)
            for h, v in d.items():
                if isinstance(sc.get(h), dict) and "p" in sc[h]:
                    v["p"] = sc[h]["p"]
            return d
        m.label, m.tau = "granger_by", None
        return m
    fn = BD.SCORERS[name]
    tau = BD.tune_tau(fn(dev), "far_fpr")

    def m(data, split):
        sc = fn(data)
        d = BD.declare(sc, tau["tau"])
        for h, v in d.items():
            if isinstance(sc.get(h), dict) and "p" in sc[h]:
                v["p"] = sc[h]["p"]
        return d
    m.label, m.tau = name, tau
    return m


__all__ = ["BENCH_TAG", "CACHE_SCHEMA", "CHAIN", "HS", "METRICS", "PREMISE", "SIZES", "Pool", "bench", "build_cache",
           "draw_subsets", "episode_arrays", "gt_reference_files", "iter_records", "load_pool", "load_ref",
           "method_baseline", "method_mscr_v2", "placebo_check", "save_ref", "subset_metrics"]
