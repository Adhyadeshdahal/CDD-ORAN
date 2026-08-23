"""2x2 decision gate for the Bayesian-direction question (see
.temp/new_arch/reports/DECISION_bayesian_direction.md).

Reads evaluated run dirs (utilities.json, schema v2) for the four cells
{discovered,oracle} x {quantile,mean}, all residual OFF, and reports the QACM
decision-level metrics that gate whether a structure-learner rewrite is worth it:
  - mean xApp utility
  - satisfaction rate (fraction of xApp-decisions satisfied)
  - lower-tail utility (10th percentile) = the risk view

Utility differences at the DECISION level are what gate the direction — not prediction MSE.
No model code, no GPU. ponytail: reuses the trained variants + the existing mean/quantile
switch; deliberately does not compute per-candidate cost-rank correlation (needs raw candidate
costs the eval doesn't persist) — mean/sat/tail are the decisive signals.
"""
import argparse
import json
import statistics
from pathlib import Path

STRUCTURES = ("discovered", "oracle")
AGGS = ("quantile", "mean")


def qacm_values(data, planner="QACM"):
    """Flatten (utilities, satisfied) for one planner across every step/panel/xApp."""
    utils, sats = [], []
    for step in data.get("steps", []):
        for panel in step.get("panels", []):
            utils.extend(panel.get("planner_utilities", {}).get(planner, []))
            sats.extend(panel.get("planner_satisfied", {}).get(planner, []))
    return utils, sats


def _pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def cell_metrics(run_dirs, planner="QACM"):
    """Aggregate one cell across its run dirs (paired seeds). Returns None if empty."""
    utils, sats = [], []
    for rd in run_dirs:
        p = Path(rd) / "utilities.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        u, s = qacm_values(data, planner)
        utils.extend(u)
        sats.extend(s)
    if not utils:
        return None
    su = sorted(utils)
    return {
        "n_runs": len(run_dirs),
        "n_decisions": len(utils),
        "mean_utility": statistics.fmean(utils),
        "satisfaction_rate": statistics.fmean(sats) if sats else float("nan"),
        "tail_utility_p10": _pct(su, 0.10),
    }


def build(cells, planner="QACM"):
    """cells: dict[(structure,agg)] -> list[run_dir]. Returns metrics table + verdict."""
    m = {}
    for st in STRUCTURES:
        for ag in AGGS:
            rds = cells.get((st, ag), [])
            m[(st, ag)] = cell_metrics(rds, planner) if rds else None

    def util(st, ag):
        c = m.get((st, ag))
        return c["mean_utility"] if c else None

    lines = ["| structure | aggregator | mean_util | sat_rate | tail_p10 | n_dec |",
             "|---|---|---|---|---|---|"]
    for st in STRUCTURES:
        for ag in AGGS:
            c = m[(st, ag)]
            if c:
                lines.append(f"| {st} | {ag} | {c['mean_utility']:.4f} | "
                             f"{c['satisfaction_rate']:.3f} | {c['tail_utility_p10']:.4f} | {c['n_decisions']} |")
            else:
                lines.append(f"| {st} | {ag} | — | — | — | 0 |")

    # gate deltas (primary column = quantile)
    verdict = []
    oq, dq = util("oracle", "quantile"), util("discovered", "quantile")
    if oq is not None and dq is not None:
        gap = oq - dq
        verdict.append(f"oracle − discovered (quantile, utility) = {gap:+.4f}")
        if gap <= 0:
            verdict.append("→ oracle does NOT beat discovered on utility: STOP the structure-learner "
                           "direction (a perfect graph doesn't help mitigation here).")
        else:
            verdict.append("→ oracle beats discovered on utility. Confirm it's material vs paired-seed "
                           "spread; if so, a dynamic-parent-mask Bayesian prototype is warranted.")
    else:
        verdict.append("oracle/discovered quantile cell missing — cannot gate the structure direction.")

    dm_disc = m.get(("discovered", "mean"))
    dq_disc = m.get(("discovered", "quantile"))
    if dm_disc and dq_disc:
        dutil = dm_disc["mean_utility"] - dq_disc["mean_utility"]
        dtail = dm_disc["tail_utility_p10"] - dq_disc["tail_utility_p10"]
        verdict.append(f"mean − quantile (discovered): Δutil={dutil:+.4f}, Δtail_p10={dtail:+.4f}")
        if dutil > 0 and dtail >= 0:
            verdict.append("→ mean improves avg utility without worsening the tail: consider mean / a "
                           "calibrated CVaR objective (still not a full Bayesian rewrite).")
        else:
            verdict.append("→ mean does not clearly help / worsens the tail: KEEP risk-averse quantile.")
    return {"metrics": m, "table": "\n".join(lines), "verdict": verdict}


def _selftest():
    # discovered-quantile: 2 xApp-decisions, both satisfied, utils 0.6/0.8
    d_q = {"steps": [{"panels": [{"planner_utilities": {"QACM": [0.6, 0.8]},
                                  "planner_satisfied": {"QACM": [1, 1]}}]}]}
    u, s = qacm_values(d_q)
    assert u == [0.6, 0.8] and s == [1, 1], (u, s)
    c = cell_metrics_from_data([d_q])
    assert abs(c["mean_utility"] - 0.7) < 1e-9 and c["satisfaction_rate"] == 1.0

    def mk(vals, sats):
        return {"steps": [{"panels": [{"planner_utilities": {"QACM": vals},
                                       "planner_satisfied": {"QACM": sats}}]}]}
    # full 2x2 where oracle < discovered -> STOP
    cells = {("discovered", "quantile"): [mk([0.7, 0.7], [1, 1])],
             ("oracle", "quantile"): [mk([0.5, 0.5], [1, 0])],
             ("discovered", "mean"): [mk([0.75, 0.75], [1, 1])],
             ("oracle", "mean"): [mk([0.5, 0.5], [1, 0])]}
    res = build_from_data(cells)
    assert any("STOP the structure-learner" in v for v in res["verdict"]), res["verdict"]
    print("selftest OK")


# test hooks that take parsed dicts instead of dirs
def cell_metrics_from_data(datas, planner="QACM"):
    utils, sats = [], []
    for data in datas:
        u, s = qacm_values(data, planner)
        utils.extend(u)
        sats.extend(s)
    if not utils:
        return None
    su = sorted(utils)
    return {"n_runs": len(datas), "n_decisions": len(utils),
            "mean_utility": statistics.fmean(utils),
            "satisfaction_rate": statistics.fmean(sats) if sats else float("nan"),
            "tail_utility_p10": _pct(su, 0.10)}


def build_from_data(cells_data, planner="QACM"):
    m = {k: cell_metrics_from_data(v, planner) for k, v in cells_data.items()}

    def util(st, ag):
        c = m.get((st, ag))
        return c["mean_utility"] if c else None
    verdict = []
    oq, dq = util("oracle", "quantile"), util("discovered", "quantile")
    if oq is not None and dq is not None:
        gap = oq - dq
        verdict.append(f"oracle − discovered (quantile) = {gap:+.4f}")
        verdict.append("STOP the structure-learner direction" if gap <= 0
                       else "structure-learner prototype warranted (confirm materiality)")
    return {"metrics": m, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="2x2 Bayesian-direction decision gate (QACM utility).")
    ap.add_argument("--run", action="append", default=[], metavar="STRUCTURE:AGG:DIR",
                    help="repeatable; STRUCTURE in {discovered,oracle}, AGG in {quantile,mean}, DIR=eval run dir")
    ap.add_argument("--planner", default="QACM")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0

    cells = {}
    for spec in a.run:
        st, ag, d = spec.split(":", 2)
        assert st in STRUCTURES and ag in AGGS, f"bad --run {spec}"
        cells.setdefault((st, ag), []).append(d)
    res = build(cells, a.planner)
    out = res["table"] + "\n\n## Verdict\n" + "\n".join(f"- {v}" for v in res["verdict"])
    print(out)
    if a.out_md:
        Path(a.out_md).write_text(f"# 2x2 Bayesian-direction gate ({a.planner})\n\n{out}\n", encoding="utf-8")
    if a.out_json:
        j = {f"{st}|{ag}": res["metrics"][(st, ag)] for st in STRUCTURES for ag in AGGS}
        j["verdict"] = res["verdict"]
        Path(a.out_json).write_text(json.dumps(j, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
