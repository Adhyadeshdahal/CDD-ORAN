"""Threshold sensitivity for a trained CDL causal graph. No retraining.

Loads a trained CDL run, reads the learned CMI matrix (model.get_causal_graph()),
sweeps cmi_threshold over a grid, and scores the binarized graph against the
environment ground truth (env.true_adj_matrix). Reports the F1-maximizing
threshold and whether F1=1.0 is reachable at any threshold.

Usage:
    uv run python -m cdd_oran.analysis.threshold_sweep --run RUN_DIR [--grid 0.02,0.40,20]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.models import get_model


def _prf(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "edges": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def sweep(run_dir: str | Path, grid: np.ndarray) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    env = get_env(cfg)
    model = get_model(cfg, env)
    model.load_model(run_dir / "checkpoint.pt")

    cmi = model.get_causal_graph().cpu().detach().numpy()
    fd = cmi.shape[0]
    gt = env.true_adj_matrix

    rows = []
    for thr in grid:
        binary = cmi >= float(thr)
        pred = binary[:, :-1].copy()
        np.fill_diagonal(pred[:fd, :fd], 0)
        rows.append({"threshold": float(thr), **_prf(pred.astype(int), gt)})

    best = max(rows, key=lambda r: (r["f1"], -r["threshold"]))
    f1_one_reachable = any(abs(r["f1"] - 1.0) < 1e-9 for r in rows)
    return {
        "run_dir": str(run_dir),
        "environment": cfg.environment,
        "config_threshold": cfg.model.cmi_threshold,
        "config_threshold_f1": next(
            (r["f1"] for r in rows if abs(r["threshold"] - cfg.model.cmi_threshold) < 1e-9),
            None,
        ),
        "gt_edge_count": int(gt.sum()),
        "best": best,
        "f1_1p0_reachable": f1_one_reachable,
        "grid": rows,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--grid", default="0.02,0.40,20", help="start,stop,num")
    p.add_argument("--out", help="Write full result JSON here")
    args = p.parse_args(argv)
    start, stop, num = args.grid.split(",")
    grid = np.linspace(float(start), float(stop), int(num))
    result = sweep(args.run, grid)
    print(json.dumps({k: v for k, v in result.items() if k != "grid"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
