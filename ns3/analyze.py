"""PMRT on ns-3 (5G-LENA) randomised-arbiter logs.

Usage: uv run python ns3/analyze.py RUNS_DIR OUT_JSON

RUNS_DIR holds one directory per episode, `<stage>_<seed>/` with `series.csv`, `units.csv` and
`topology.csv`; stages are dev, eval, placebo. Knockouts live in `gt_<seed>_<cell>_<t0>_reject/`;
their accept arm is the placebo run of the same seed, which applies accept everywhere on the same
common random numbers.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from oran_causal.core.data import UnitTable
from oran_causal.core.design import v_design
from oran_causal.core.pmrt.units import PmrtConfig, PmrtData, fit_pmrt
from oran_causal.methods.system import pmrt as pm
from oran_causal.sim.system.decision.vocab import LEVEL_ARR, MODES

HS = (30, 60, 90, 150)
H = 90
FIELDS = {"load": "util", "thr": "thr", "viol": "viol", "e": "energy", "delay": "delay"}
KPIS = tuple(FIELDS)
RELS = ("own", "nbr", "far")
FAMS = ("ptx",)
MODE_IX = {m: i for i, m in enumerate(MODES)}


def _load_episode(d: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    s = pd.read_csv(d / "series.csv")
    topo = pd.read_csv(d / "topology.csv")
    cells = len(topo)
    t = s["t"].round().astype(int).to_numpy()
    T = int(t.max())
    data = np.zeros((T, len(KPIS), cells))
    for j, k in enumerate(KPIS):
        data[t - 1, j, s["cell"].to_numpy()] = s[FIELDS[k]].to_numpy()
    u = pd.read_csv(d / "units.csv")
    xy = topo[["x", "y"]].to_numpy()
    return data, u, xy


def _exposure(xy: np.ndarray, radius: float) -> np.ndarray:
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return dist <= radius


def _build(eps: list[tuple[np.ndarray, pd.DataFrame, np.ndarray]], radius: float) -> PmrtData:
    rows = []
    for e, (data, u, xy) in enumerate(eps):
        T = len(data)
        exp = _exposure(xy, radius)
        for r in u.to_dict("records"):
            t0 = round(r["t0"])
            if t0 - H < 0 or t0 + H > T:
                continue
            rows.append((e, t0, int(r["cell"]), r, data, exp))
    rows.sort(key=lambda z: (z[0], z[1], z[2]))
    n = len(rows)
    C = eps[0][0].shape[2]
    K = len(KPIS)
    post = np.full((n, K, C, 4), np.nan, np.float32)
    pre = post.copy()
    mask = {r: np.zeros((n, C), bool) for r in RELS}
    mode = np.zeros(n, int)
    probs = np.zeros((n, 4))
    step = np.zeros(n)
    ep = np.zeros(n, int)
    cc = np.zeros(n, int)
    t0s = np.zeros(n, int)
    xapp = np.empty(n, object)
    ctx = np.zeros((n, 4))
    tlen = np.zeros(n)
    for i, (e, t0, c, r, data, exp) in enumerate(rows):
        T = len(data)
        nan = np.full((K, C), np.nan)
        cp = [data[t0 : t0 + h].sum(0) if t0 + h <= T else nan for h in HS]
        cq = [data[t0 - h : t0].sum(0) if t0 - h >= 0 else nan for h in HS]
        post[i] = np.diff(np.stack([np.zeros((K, C)), *cp], -1), axis=-1)
        pre[i] = np.diff(np.stack([np.zeros((K, C)), *cq], -1), axis=-1)
        own = np.eye(C, dtype=bool)[c]
        mask["own"][i], mask["nbr"][i], mask["far"][i] = own, exp[c] & ~own, ~exp[c]
        mode[i] = MODE_IX[r["mode"]]
        probs[i] = [r["p_accept"], r["p_half"], r["p_reject"], 0.0]
        step[i] = r["prop"] - r["cur"]
        ep[i], cc[i], t0s[i], xapp[i], tlen[i] = e, c, t0, r["xapp"], T
        w = data[t0 - 10 : t0]
        ctx[i] = [r["cur"], w[:, 0, c].mean(), w[:, 2, c].mean(), float(r["xapp"] == "Coverage")]
    p3 = post[..., :3].sum(-1).astype(float)
    q3 = pre[..., :3].sum(-1).astype(float)
    y = {(r, k): ((p3[:, j] - q3[:, j]) * mask[r]).sum(1) for r in RELS for j, k in enumerate(KPIS)}
    pq = {(r, k): (q3[:, j] * mask[r]).sum(1) for r in RELS for j, k in enumerate(KPIS)}
    sgn = np.where(step == 0, 1.0, np.sign(step))
    ud = UnitTable(
        episode=ep,
        seed=ep.copy(),
        fold=np.full(n, -1),
        family=np.zeros(n, int),
        xapp=xapp,
        c=cc,
        t0=t0s,
        mode=mode,
        applied=mode.copy(),
        p=probs[np.arange(n), mode],
        probs=probs,
        step=step,
        sgn=sgn,
        y=y,
        pre=pq,
        ycell={k: p3[:, j] - q3[:, j] for j, k in enumerate(KPIS)},
        rel_mask=mask,
        z={},
        prev=np.full(n, -1),
        n_cells=C,
        H=H,
        H_pre=H,
        relations=RELS,
        kpis=KPIS,
        families=FAMS,
        levels=LEVEL_ARR,
        meta={},
    )
    v = v_design(ud.mode, ud.probs, ud.sgn, pm.LEVEL_V2_ARR)
    hist = np.zeros((n, 2))
    for i in range(n):
        past = np.nonzero((ep == ep[i]) & (t0s < t0s[i]) & (t0s >= t0s[i] - 150))[0]
        for jj in past:
            if cc[jj] == cc[i]:
                hist[i, 0] += v[jj]
            elif mask["nbr"][i, cc[jj]]:
                hist[i, 1] += v[jj]
    return PmrtData(
        ud=ud,
        post=post,
        pre=pre,
        ctx=ctx,
        ctx_keys=["cur", "util10", "viol10", "is_cov"],
        hist=hist,
        tfrac=t0s / tlen,
    )


def _stage(root: Path, name: str) -> list[tuple[np.ndarray, pd.DataFrame, np.ndarray]]:
    dirs = sorted(
        (d for d in root.glob(f"{name}_*") if (d / "units.csv").exists()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return [_load_episode(d) for d in dirs]


def _edges(run: dict, arm: str) -> dict[str, dict]:
    return {"|".join(k): v for k, v in pm.edges_pmrt(run, arm).items()}


def _knockout_diffs(
    root: Path, radius: float
) -> tuple[dict[tuple[str, str], list[float]], np.ndarray]:
    pairs: dict[tuple, dict[str, Path]] = {}
    for d in root.glob("gt_*"):
        if not (d / "series.csv").exists():
            continue
        _, seed, cell, t0, mode = d.name.split("_")
        arms = pairs.setdefault(
            (int(seed), int(cell), int(t0)), {"accept": root / f"placebo_{seed}"}
        )
        arms[mode] = d
    diffs: dict[tuple[str, str], list[float]] = {(r, k): [] for r in RELS for k in KPIS}
    seeds: list[int] = []
    for (seed, cell, t0), m in sorted(pairs.items()):
        if "reject" not in m or not (m["accept"] / "series.csv").exists():
            continue
        da, ua, xy = _load_episode(m["accept"])
        dr, _, _ = _load_episode(m["reject"])
        unit = ua[(ua.cell == cell) & (ua.t0.round() == t0)]
        if unit.empty or t0 + H > min(len(da), len(dr)):
            continue
        sgn = float(np.sign(unit.prop.iloc[0] - unit.cur.iloc[0]) or 1.0)
        exp = _exposure(xy, radius)
        own = np.eye(len(xy), dtype=bool)[cell]
        rm = {"own": own, "nbr": exp[cell] & ~own, "far": ~exp[cell]}
        delta = da[t0 : t0 + H].sum(0) - dr[t0 : t0 + H].sum(0)
        for r in RELS:
            for j, k in enumerate(KPIS):
                diffs[(r, k)].append(sgn * float(delta[j][rm[r]].sum()))
        seeds.append(seed)
    return diffs, np.array(seeds)


def _label(m: float, lo: float, hi: float, delta_k: float) -> str:
    if (lo > 0 or hi < 0) and abs(m) >= delta_k:
        return "TRUE+" if m > 0 else "TRUE-"
    if -delta_k <= lo and hi <= delta_k:
        return "NULL"
    return "INDET"


def _ground_truth(root: Path, radius: float, rng: np.random.Generator) -> dict[str, dict]:
    diffs, seeds_a = _knockout_diffs(root, radius)
    out: dict[str, dict] = {}
    means = {h: float(np.mean(v)) if v else float("nan") for h, v in diffs.items()}
    for (r, k), v in diffs.items():
        a = np.array(v)
        if len(a) < 10 or len(np.unique(seeds_a)) < 3:
            out[f"ptx|{r}|{k}"] = {"label": "INDET", "n": len(a)}
            continue
        us = np.unique(seeds_a)
        boot = []
        for _ in range(4000):
            pick = rng.choice(us, len(us))
            boot.append(np.concatenate([a[seeds_a == s] for s in pick]).mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        delta_k = 0.05 * max(abs(means[(rr, k)]) for rr in RELS)
        m = float(a.mean())
        out[f"ptx|{r}|{k}"] = {
            "label": _label(m, float(lo), float(hi), delta_k),
            "mean": m,
            "ci": [float(lo), float(hi)],
            "delta": delta_k,
            "n": len(a),
            "n_seeds": len(us),
        }
    return out


def _score(decl: dict[str, dict], gt: dict[str, dict]) -> dict[str, float]:
    tp = fp = fn = sign_ok = 0
    for h, g in gt.items():
        d = decl.get(h, {}).get("declared", False)
        if g["label"].startswith("TRUE"):
            if d:
                tp += 1
                sign_ok += int((decl[h]["sign"] > 0) == (g["label"] == "TRUE+"))
            else:
                fn += 1
        elif g["label"] == "NULL" and d:
            fp += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "sign_acc": sign_ok / tp if tp else float("nan"),
    }


def _naive(pd_: PmrtData) -> dict[str, dict]:
    ud = pd_.ud
    x = ud.levels[ud.mode] * ud.sgn
    out = {}
    for (r, k), yv in ud.y.items():
        c = float(np.corrcoef(x, yv)[0, 1]) if np.std(yv) > 0 else float("nan")
        out[f"ptx|{r}|{k}"] = {"corr": c}
    return out


def main() -> None:
    """Fit PMRT on dev, test eval and placebo, label knockouts and score; write OUT_JSON."""
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
    radius = float(sys.argv[4]) if len(sys.argv) > 4 else 1.01 * 300
    cfg = PmrtConfig(B=B)
    res: dict = {"B": B, "radius": radius, "kpi_fields": FIELDS}
    dev = _build(_stage(root, "dev"), radius)
    params = fit_pmrt(dev, pm.LEVEL_V2_ARR, cfg)
    res["n_units"] = {"dev": dev.n}
    for name, split in (("eval", 0), ("placebo", 9)):
        eps = _stage(root, name)
        if not eps:
            continue
        pdx = _build(eps, radius)
        res["n_units"][name] = pdx.n
        res[f"support_{name}"] = {
            m: int((pdx.ud.mode == MODE_IX[m]).sum()) for m in ("accept", "half", "reject")
        }
        run = pm.run_pmrt(pdx, params, cfg, split=split)
        res[f"{name}_edges"] = _edges(run, "best")
        ps = {
            a: [r["arms"][a]["p"] for r in run["results"] if r["status"] == "tested"]
            for a in run["arms"]
        }
        res[f"{name}_rej05"] = {a: int(np.sum(np.array(v) <= 0.05)) for a, v in ps.items()}
        m = len(ps.get("best", []))
        res[f"{name}_m"] = m
        res[f"{name}_k0_tail"] = {
            a: float(binom.sf(n - 1, m, 0.05)) for a, n in res[f"{name}_rej05"].items()
        }
        res[f"{name}_naive"] = _naive(pdx)
    gt = _ground_truth(root, radius, np.random.default_rng([0, 6617]))
    res["gt"] = gt
    if gt and "eval_edges" in res:
        res["score_eval"] = _score(res["eval_edges"], gt)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("eval_naive", "placebo_naive")},
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
