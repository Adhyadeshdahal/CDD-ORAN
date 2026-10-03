"""F4 null level of pcorr_hac (ruling R-32) vs plain pcorr (R-5 t-test), through the adapter, on synthetic nulls.

Synthetic api.Dataset (no study world, no study seeds): actions X (focal), W1, W2 (covariates) and P_placebo (i.i.d.),
one KPI Y, no lagged KPIs (all-NaN, dropped by the helper), so the native R-3 set of X is {W1, W2, P_placebo}:
    W1, W2 ~ AR(1)(phi_w)          X = .5 W1 - .3 W2 + e          Y = .4 W1 + .2 W2 + u
    e, u independent AR(1)(rho) with N(0, 1) innovations (stationary start); X _||_ Y | Z exactly (H0 true).
Scenarios: iid (rho = 0, phi_w = 0), ar1_.5 (rho = phi_w = .5), ar1_.8 (rho = phi_w = .8); n in {500, 1000, 4000};
REPS replications. Rejection rate at .05 / .01 of the null X -> Y (and the i.i.d. P_placebo -> Y) for pcorr_hac
(adapter, arm native) and for plain pcorr on the same rows and Z (t-test, df = n - 2 - |Z|). 95 % Wilson CIs.
Expectation: both hold level for iid; plain pcorr over-rejects when both e and u are serially correlated
(rejection grows with rho), pcorr_hac stays near nominal (finite-sample HAC over-rejection shrinks with n).
R-38: also the fixed-b variant (``inference='fixed_b'``: same statistic, Kiefer-Vogelsang 2005 fixed-b p at
b = (L + 1) / n, computed from the run's per-edge (t, b), identical to the adapter's fixed_b p, test) and the
pre-freeze selection of the set-D member: the variant with the smaller max |rate - .05| (X -> Y, level .05) over the
9 cells (decided on this synthetic grid only; a tie keeps the t variant, the R-32 default).
RNG: SeedSequence([9_200, scenario, n, rep]) (synthetic only; same datasets as the R-32 run).
Output: $JOB_OUT or results/ / pcorr_hac_f4b.json (results/pcorr_hac_f4.json = the R-32 run, t variant only).
Run: uv run python scratchpad/xmethod/pcorr_hac_fidelity.py [REPS] [OUT]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
from scipy import stats

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod.methods._classic_common import cond_set, pcorr  # noqa: E402
from cdd_oran.xmethod.methods._fixedb import fixedb_p  # noqa: E402
from cdd_oran.xmethod.methods.pcorr_hac import PcorrHac  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results" / "pcorr_hac_f4.json"
SCEN = {"iid": (0.0, 0.0), "ar1_.5": (0.5, 0.5), "ar1_.8": (0.8, 0.8)}
NS = (500, 1000, 4000)
TESTS = ("pcorr_hac", "pcorr_hac_fixedb", "pcorr_plain")


def ar1(r: np.random.Generator, n: int, rho: float) -> np.ndarray:
    w = r.normal(size=n)
    out = np.empty(n)
    out[0] = w[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + w[t]
    return out


def make(n: int, rho: float, phi_w: float, seed: list[int]) -> api.Dataset:
    r = np.random.default_rng(np.random.SeedSequence(seed))
    w1, w2 = ar1(r, n, phi_w), ar1(r, n, phi_w)
    x = 0.5 * w1 - 0.3 * w2 + ar1(r, n, rho)
    y = 0.4 * w1 + 0.2 * w2 + ar1(r, n, rho)
    pl = r.uniform(size=n)
    acts = ("X", "W1", "W2", "P_placebo")
    des = tuple(api.Design(kind="none") for _ in acts[:3]) + (api.Design(kind="iid", dist={"name": "uniform",
                                                                                              "lo": 0, "hi": 1}),)
    return api.Dataset(world="SYN-HAC", regime="R1", n=n, seed=int(seed[-1]), action_names=acts, kpi_names=("Y",),
                       X_action=np.column_stack([x, w1, w2, pl]), X_kpi_lag=np.full((n, 1), np.nan),
                       Y=y.reshape(-1, 1), designs=des, candidates=(("X", "Y"), ("P_placebo", "Y")),
                       time_index=np.arange(n), context=None, meta={"synthetic": True})


def plain_p(data: api.Dataset, ai: int) -> float:
    cs = cond_set(data, "action", ai, "native")
    rho = pcorr(data.X_action[:, ai], data.Y[:, 0], cs.Z)
    df = data.n - 2 - cs.Z.shape[1]
    t = abs(rho) * np.sqrt(df / max(1 - rho ** 2, 1e-300))
    return float(2 * stats.t.sf(t, df))


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def main(reps: int, out_path: str) -> None:
    m = PcorrHac()
    rows = []
    for si, (scen, (rho, phi)) in enumerate(SCEN.items()):
        for n in NS:
            t0 = time.process_time()
            P = {t: {"X": [], "P_placebo": []} for t in TESTS}
            lags = []
            for rep in range(reps):
                d = make(n, rho, phi, [9_200, si, n, rep])
                res = m.run(d)                                   # inference 't'; fixed-b p from the same (t, b)
                for e in res.edges:
                    t, _, b = res.notes["t_df_b"][f"{e.source}->Y"]
                    P["pcorr_hac"][e.source].append(e.p)
                    P["pcorr_hac_fixedb"][e.source].append(fixedb_p(t, b))
                    P["pcorr_plain"][e.source].append(plain_p(d, d.action_names.index(e.source)))
                lags.append(res.notes["maxlags"]["X->Y"])
            for src in ("X", "P_placebo"):
                for test in TESTS:
                    a = np.asarray(P[test][src])
                    row = {"scenario": scen, "rho": rho, "n": n, "source": src, "test": test, "reps": reps,
                           "maxlags_median_X": float(np.median(lags))}
                    for lvl in (0.05, 0.01):
                        k = int((a <= lvl).sum())
                        row[f"rej_{lvl}"] = round(k / reps, 4)
                        row[f"ci_{lvl}"] = wilson(k, reps)
                    rows.append(row)
            print(f"{scen:7s} n={n:5d} X: " + " ".join(f"{t} {np.mean(np.asarray(P[t]['X']) <= .05):.3f}" for t in TESTS)
                  + f" | L med {np.median(lags):.0f} ({time.process_time() - t0:.0f} cpu-s)", flush=True)
    # R-38 selection (pre-freeze, synthetic only): the variant with the smaller max |rate - .05| over the grid (X -> Y)
    dev = {t: max(abs(r["rej_0.05"] - 0.05) for r in rows if r["test"] == t and r["source"] == "X")
           for t in ("pcorr_hac", "pcorr_hac_fixedb")}
    sel = min(dev, key=dev.get)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reps": reps, "rows": rows, "max_abs_dev_X_05": dev,
                               "selected_set_D": {"pcorr_hac": "inference t", "pcorr_hac_fixedb": "inference fixed_b"}[sel]},
                              indent=1), encoding="utf-8")
    print("max |rate - .05| (X, .05):", dev, "-> set D member:", sel, "| wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000,
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("JOB_OUT", str(OUT.parent)),
                                                             "pcorr_hac_f4b.json"))
