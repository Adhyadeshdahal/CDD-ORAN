"""Per-edge-type recovery metrics for a predicted causal graph.

Splits precision/recall/F1 by parent-column type against ``env.true_adj_matrix``:
- NCP->KPI edges: parent column ``< num_params`` (an NCP source).
- KPI->KPI edges: parent column ``>= num_params`` (a KPI source).

Interventions poke only NCPs, so this split is the key diagnostic for the ENCO
decision: a recall floor made of NCP->KPI misses is something interventional
discovery could lift, while a KPI->KPI floor is not (KPIs are never intervened).

Both inputs are ``(fd, fd)`` graphs (action column already dropped), the same
layout ``threshold_sweep``/``auto_threshold`` produce after binarizing.

Self-check:
    uv run python -m cdd_oran.analysis.recovery_metrics --self-check
"""

import argparse

import numpy as np

from cdd_oran.analysis.threshold_sweep import _prf


def recovery_by_edge_type(
    pred: np.ndarray,
    gt: np.ndarray,
    num_params: int,
    node_names: list[str] | None = None,
) -> dict:
    """Precision/recall/F1 split overall / NCP->KPI / KPI->KPI, plus missed edges.

    ``pred`` and ``gt`` are ``(fd, fd)`` binary graphs (child row, parent col).
    Returns a dict with ``overall``, ``ncp_kpi``, ``kpi_kpi`` PRF blocks and a
    ``missed`` list of ``{child, parent, type[, child_name, parent_name]}`` for
    every true edge the prediction missed.
    """
    pred = np.asarray(pred).astype(int)
    gt = np.asarray(gt).astype(int)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} must match")

    ncp = slice(0, num_params)
    kpi = slice(num_params, gt.shape[1])
    result = {
        "num_params": int(num_params),
        "overall": _prf(pred, gt),
        "ncp_kpi": _prf(pred[:, ncp], gt[:, ncp]),
        "kpi_kpi": _prf(pred[:, kpi], gt[:, kpi]),
        "missed": [],
    }

    miss_rows, miss_cols = np.where((gt == 1) & (pred == 0))
    for child, parent in zip(miss_rows.tolist(), miss_cols.tolist(), strict=True):
        edge = {
            "child": int(child),
            "parent": int(parent),
            "type": "ncp_kpi" if parent < num_params else "kpi_kpi",
        }
        if node_names is not None:
            edge["child_name"] = node_names[child]
            edge["parent_name"] = node_names[parent]
        result["missed"].append(edge)
    return result


def _self_check() -> None:
    """Synthetic graph: 3 NCP->KPI + 1 KPI->KPI true edges, one miss of each type
    and one NCP false positive. Verifies the split counts and the missed list."""
    num_params = 3
    fd = 5  # nodes: 0,1,2 = NCP; 3,4 = KPI
    gt = np.zeros((fd, fd), dtype=int)
    gt[3, 0] = 1  # NCP->KPI
    gt[3, 1] = 1  # NCP->KPI
    gt[4, 0] = 1  # NCP->KPI
    gt[4, 3] = 1  # KPI->KPI (parent col 3 >= num_params)

    pred = np.zeros((fd, fd), dtype=int)
    pred[3, 0] = 1  # TP  (ncp)
    pred[4, 0] = 1  # TP  (ncp)
    pred[3, 2] = 1  # FP  (ncp)
    # misses gt[3,1] (ncp) and gt[4,3] (kpi)

    out = recovery_by_edge_type(pred, gt, num_params)

    assert out["ncp_kpi"]["tp"] == 2 and out["ncp_kpi"]["fp"] == 1
    assert out["ncp_kpi"]["fn"] == 1
    assert abs(out["ncp_kpi"]["recall"] - 2 / 3) < 1e-12, out["ncp_kpi"]["recall"]
    assert out["kpi_kpi"]["tp"] == 0 and out["kpi_kpi"]["fn"] == 1
    assert out["kpi_kpi"]["recall"] == 0.0
    assert out["overall"]["tp"] == 2 and out["overall"]["fn"] == 2

    missed = {(m["child"], m["parent"], m["type"]) for m in out["missed"]}
    assert missed == {(3, 1, "ncp_kpi"), (4, 3, "kpi_kpi")}, missed

    # node_names labelling is passed through when provided.
    named = recovery_by_edge_type(pred, gt, num_params, ["p0", "p1", "p2", "k0", "k1"])
    assert named["missed"][0]["parent_name"] in {"p1", "k0"}

    print(
        "recovery_metrics self-check passed: "
        f"ncp recall={out['ncp_kpi']['recall']:.3f} (2/3), "
        f"kpi recall={out['kpi_kpi']['recall']:.3f} (0), "
        f"missed={len(out['missed'])} edges"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true", help="Run the synthetic check and exit")
    args = p.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    p.error("--self-check is the only mode; import recovery_by_edge_type to use the helper")


if __name__ == "__main__":
    raise SystemExit(main())
