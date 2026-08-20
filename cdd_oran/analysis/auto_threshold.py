"""Unsupervised CMI threshold selection for a trained CDL causal graph.

The causal graph is built by thresholding the CMI matrix at a HARDCODED
``cmi_threshold``. At deployment there is no ground truth to pick that number.
This module reads a threshold straight from the CMI value distribution, with no
labels, by three methods:

1. ``largest_gap`` -- sort the candidate CMI values and cut at the midpoint of
   the biggest gap between consecutive values (above a small floor). Signal and
   noise usually separate at one dominant gap.
2. ``otsu`` -- 1-D Otsu between-class variance: the cut that best splits the
   values into two clusters (noise vs edges).
3. ``kmeans2`` -- 1-D 2-means on the values; cut = midpoint of the two cluster
   means.

``report(run_dir)`` scores each auto-threshold graph against the environment
ground truth FOR VALIDATION ONLY (labels are used to score, never to choose),
next to the hardcoded-threshold F1 and the sweep-optimal (oracle) F1.

No retraining, no forward passes -- only the checkpoint's CMI matrix is read.

Usage:
    uv run python -m cdd_oran.analysis.auto_threshold --run RUN_DIR
    uv run python -m cdd_oran.analysis.auto_threshold --self-check
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cdd_oran.analysis.threshold_sweep import _prf, sweep


def candidate_values(cmi: np.ndarray) -> np.ndarray:
    """Flat 1-D array of candidate CMI values: off-diagonal of ``cmi[:, :-1]``.

    Drops the action column (last) and the self-loop diagonal -- the entries the
    graph binarizer never keeps.
    """
    core = cmi[:, :-1]  # (fd, fd): NCP/KPI candidate columns, action column removed
    fd = core.shape[0]
    mask = ~np.eye(fd, dtype=bool)
    return core[mask].astype(float).ravel()


def binary_graph(cmi: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a (fd, fd+1) CMI matrix into a (fd, fd) graph at ``threshold``."""
    pred = (cmi[:, :-1] >= float(threshold)).astype(int)
    np.fill_diagonal(pred, 0)
    return pred


def largest_gap(values: np.ndarray, floor: float = 1e-3) -> float:
    """Cut at the midpoint of the biggest gap between sorted CMI values.

    Only gaps whose lower endpoint is at or above ``floor`` are considered, so a
    huge gap from ~0 up to the first real value does not win. Falls back to the
    global biggest gap if nothing clears the floor.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if v.size < 2:
        return float(v[0]) if v.size else 0.0
    gaps = np.diff(v)
    eligible = v[:-1] >= floor
    if eligible.any():
        idx = int(np.argmax(np.where(eligible, gaps, -np.inf)))
    else:
        idx = int(np.argmax(gaps))
    return float((v[idx] + v[idx + 1]) / 2.0)


def otsu(values: np.ndarray) -> float:
    """1-D Otsu threshold: maximize between-class variance over split points."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n < 2:
        return float(v[0]) if n else 0.0
    csum = np.cumsum(v)
    total = csum[-1]
    best_var = -1.0
    best_thr = float(v[0])
    for k in range(1, n):  # class0 = v[:k], class1 = v[k:]
        if v[k] == v[k - 1]:
            continue  # no distinct cut between equal values
        w0 = k / n
        w1 = 1.0 - w0
        mu0 = csum[k - 1] / k
        mu1 = (total - csum[k - 1]) / (n - k)
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thr = (v[k - 1] + v[k]) / 2.0
    return float(best_thr)


def kmeans2(values: np.ndarray, iters: int = 100) -> float:
    """1-D 2-means; cut = midpoint of the two cluster means.

    Deterministic: centers are initialized to the min and max value.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 2:
        return float(v[0]) if v.size else 0.0
    c0, c1 = float(v.min()), float(v.max())
    if c0 == c1:
        return c0
    for _ in range(iters):
        lo = np.abs(v - c0) <= np.abs(v - c1)
        new0 = v[lo].mean() if lo.any() else c0
        new1 = v[~lo].mean() if (~lo).any() else c1
        if new0 == c0 and new1 == c1:
            break
        c0, c1 = new0, new1
    return float((c0 + c1) / 2.0)


METHODS = {"largest_gap": largest_gap, "otsu": otsu, "kmeans2": kmeans2}


def auto_thresholds(cmi: np.ndarray) -> dict[str, float]:
    """Chosen threshold per method from the CMI value distribution (no labels)."""
    values = candidate_values(cmi)
    return {name: float(fn(values)) for name, fn in METHODS.items()}


def report(run_dir: str | Path) -> dict:
    """Score each auto threshold vs ground truth (validation only) against the
    hardcoded threshold and the sweep-optimal (oracle) threshold."""
    from cdd_oran.config import load_config
    from cdd_oran.envs import get_env
    from cdd_oran.models import get_model

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    env = get_env(cfg)
    model = get_model(cfg, env)
    model.load_model(run_dir / "checkpoint.pt")

    cmi = model.get_causal_graph().cpu().detach().numpy()
    gt = env.true_adj_matrix

    thresholds = auto_thresholds(cmi)
    methods = {
        name: {"threshold": thr, **_prf(binary_graph(cmi, thr), gt)}
        for name, thr in thresholds.items()
    }

    hardcoded_thr = cfg.model.cmi_threshold
    hardcoded = {"threshold": hardcoded_thr, **_prf(binary_graph(cmi, hardcoded_thr), gt)}

    grid = np.linspace(0.0, float(cmi[:, :-1].max()) + 1e-6, 200)
    oracle = sweep(run_dir, grid)["best"]

    return {
        "run_dir": str(run_dir),
        "environment": cfg.environment,
        "gt_edge_count": int(gt.sum()),
        "hardcoded": hardcoded,
        "auto": methods,
        "sweep_optimal": oracle,
    }


def _self_check() -> None:
    """Clear signal/noise gap: largest_gap AND otsu both cut inside the gap."""
    noise = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    signal = np.array([0.80, 0.90, 1.00, 1.10])
    values = np.concatenate([noise, signal])

    gap_lo, gap_hi = 0.06, 0.80
    for name in ("largest_gap", "otsu", "kmeans2"):
        thr = METHODS[name](values)
        assert gap_lo < thr < gap_hi, f"{name} cut {thr:.3f} not inside gap ({gap_lo}, {gap_hi})"

    # Determinism: same input -> same output.
    assert largest_gap(values) == largest_gap(values)
    assert otsu(values) == otsu(values)
    assert kmeans2(values) == kmeans2(values)

    # candidate_values drops the action column and the diagonal.
    cmi = np.array([[9.0, 0.5, 0.3], [0.2, 9.0, 0.4]] , dtype=float)  # (2,3): fd=2, +action col
    vals = candidate_values(cmi)
    assert sorted(vals.tolist()) == [0.2, 0.5], f"candidate extraction wrong: {vals}"

    print(
        "auto_threshold self-check passed: "
        f"largest_gap={largest_gap(values):.3f}, otsu={otsu(values):.3f}, "
        f"kmeans2={kmeans2(values):.3f} (all inside the gap)"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument("--out", help="Write the full report JSON here")
    p.add_argument("--self-check", action="store_true", help="Run the synthetic check and exit")
    args = p.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if not args.run:
        p.error("--run is required unless --self-check is given")
    result = report(args.run)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
