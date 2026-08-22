"""CPU-only counterfactual rollout metrics and post-hoc study helpers.

The metric functions in this module deliberately operate on arrays and saved records.
They do not import a model, planner, environment, or CUDA.  Evaluation-time collection
is kept in :mod:`cdd_oran.experiments.evaluate`; this module is the reusable scoring and
study layer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cdd_oran.analysis.stats import bh_fdr, cliffs_delta, holm_correction


def _as_member_horizon_k(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (m, H, k) or (H, k); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_horizon_k(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (H, k); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _discount_weights(horizon: int, gamma: float) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and greater than zero")
    return np.power(float(gamma), np.arange(horizon, dtype=float))


def counterfactual_action_response_error(
    pred_action: Any,
    pred_reference: Any,
    true_action: Any,
    true_reference: Any,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Compute per-member and aggregate CF_ARRE.

    The four inputs are action/reference *next-KPI trajectories*, not raw action
    encodings.  Predicted arrays are ``(members, horizon, kpis)`` and true arrays are
    ``(horizon, kpis)``.  A two-dimensional predicted array is treated as one member.
    """
    pred_a = _as_member_horizon_k(pred_action, "pred_action")
    pred_r = _as_member_horizon_k(pred_reference, "pred_reference")
    true_a = _as_horizon_k(true_action, "true_action")
    true_r = _as_horizon_k(true_reference, "true_reference")
    if pred_a.shape != pred_r.shape:
        raise ValueError(f"pred_action and pred_reference shapes differ: {pred_a.shape} vs {pred_r.shape}")
    if pred_a.shape[1:] != true_a.shape or true_a.shape != true_r.shape:
        raise ValueError(
            "counterfactual trajectories must share (H, k); "
            f"got predicted {pred_a.shape[1:]}, true_action {true_a.shape}, "
            f"true_reference {true_r.shape}"
        )
    weights = _discount_weights(true_a.shape[0], gamma)
    delta_pred = pred_a - pred_r
    delta_true = true_a - true_r
    per_horizon = np.mean((delta_pred - delta_true[None, ...]) ** 2, axis=2)
    per_member = (per_horizon * weights[None, :]).sum(axis=1) / weights.sum()
    return {
        "per_member": per_member,
        "mean": float(np.mean(per_member)),
        "max_member": float(np.max(per_member)),
        "per_horizon_per_member": per_horizon,
    }


def autoregressive_rollout_rmse(
    predicted: Any, truth: Any, gamma: float = 1.0
) -> dict[str, Any]:
    """Compute discounted per-member CF_RMSE for a closed-loop rollout."""
    pred = _as_member_horizon_k(predicted, "predicted")
    actual = _as_horizon_k(truth, "truth")
    if pred.shape[1:] != actual.shape:
        raise ValueError(f"predicted and truth shapes differ: {pred.shape[1:]} vs {actual.shape}")
    weights = _discount_weights(actual.shape[0], gamma)
    per_horizon = np.mean((pred - actual[None, ...]) ** 2, axis=2)
    per_member = np.sqrt((per_horizon * weights[None, :]).sum(axis=1) / weights.sum())
    return {
        "per_member": per_member,
        "mean": float(np.mean(per_member)),
        "max_member": float(np.max(per_member)),
        "per_horizon_per_member": per_horizon,
    }


def one_step_mse(predicted: Any, truth: Any) -> dict[str, Any]:
    """Compute matched local one-step MSE per member and after member aggregation."""
    pred = _as_member_horizon_k(predicted, "predicted")
    actual = _as_horizon_k(truth, "truth")
    if pred.shape[1:] != actual.shape:
        raise ValueError(f"predicted and truth shapes differ: {pred.shape[1:]} vs {actual.shape}")
    per_member = np.mean((pred - actual[None, ...]) ** 2, axis=(1, 2))
    return {"per_member": per_member, "mean": float(np.mean(per_member))}


def _objective_for_value(
    raw_params: Sequence[float],
    param_id: int,
    value: float,
    utility_fns: Sequence[Any],
    xapps: Sequence[Any],
    weights: np.ndarray,
    scaling_term: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    params = np.asarray(raw_params, dtype=float).copy()
    params[param_id] = value
    utilities = np.asarray([fn(params.tolist()) for fn in utility_fns], dtype=float)
    distances = np.zeros(len(xapps), dtype=float)
    satisfied = np.zeros(len(xapps), dtype=float)
    for i, (xapp, utility) in enumerate(zip(xapps, utilities, strict=True)):
        threshold = (float(xapp.threshold) - float(xapp.mean)) / float(xapp.std)
        if int(xapp.direction) == 0:
            distance = max(threshold - utility, 0.0)
            ok = utility >= threshold
        else:
            distance = max(utility - threshold, 0.0)
            ok = utility <= threshold
        distances[i] = distance
        satisfied[i] = float(ok)
    cost = float(np.sum(weights * distances * float(scaling_term)) - np.sum(satisfied) ** 2)
    return cost, utilities, satisfied


def decision_regret(
    raw_params: Sequence[float],
    param_id: int,
    sweep: Sequence[float],
    utility_fns: Sequence[Any],
    xapps_under_conflict: Sequence[Any],
    weights_per_xapps: Sequence[float],
    planner_value: float | None = None,
    scaling_term: float = 10.0,
    *,
    planner_raw_value: float | None = None,
) -> dict[str, Any]:
    """Evaluate exact planner cost, grid oracle cost, and decision regret.

    ``utility_fns`` is aligned with ``xapps_under_conflict``.  The oracle is restricted
    to the supplied feasible sweep, matching the evaluator's 101-point action grid.
    """
    if planner_value is None:
        planner_value = planner_raw_value
    if planner_value is None:
        raise ValueError("planner_value is required")
    xapps = list(xapps_under_conflict)
    fns = list(utility_fns)
    if not xapps or len(fns) != len(xapps):
        raise ValueError("utility_fns and xapps_under_conflict must be non-empty and aligned")
    candidates = np.asarray(sweep, dtype=float).ravel()
    if candidates.size == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError("sweep must contain finite candidate values")
    weights = np.asarray(weights_per_xapps, dtype=float).ravel()
    if weights.shape != (len(xapps),) or np.any(weights < 0):
        raise ValueError("weights_per_xapps must match xapps_under_conflict and be non-negative")
    if not np.isfinite(scaling_term):
        raise ValueError("scaling_term must be finite")

    candidate_costs = np.asarray(
        [
            _objective_for_value(raw_params, param_id, value, fns, xapps, weights, scaling_term)[0]
            for value in candidates
        ],
        dtype=float,
    )
    oracle_index = int(np.argmin(candidate_costs))
    oracle_value = float(candidates[oracle_index])
    oracle_cost, oracle_utilities, _ = _objective_for_value(
        raw_params, param_id, oracle_value, fns, xapps, weights, scaling_term
    )
    planner_cost, planner_utilities, _ = _objective_for_value(
        raw_params, param_id, float(planner_value), fns, xapps, weights, scaling_term
    )
    regret = max(0.0, float(planner_cost - oracle_cost))
    utility_regrets = np.asarray(
        [
            float(oracle_u - planner_u) if int(xapp.direction) == 0 else float(planner_u - oracle_u)
            for xapp, oracle_u, planner_u in zip(
                xapps, oracle_utilities, planner_utilities, strict=True
            )
        ],
        dtype=float,
    )
    return {
        "oracle_action": oracle_value,
        "oracle_cost": float(oracle_cost),
        "planner_action": float(planner_value),
        "planner_cost": float(planner_cost),
        "decision_regret": regret,
        "oracle_utilities": oracle_utilities,
        "planner_utilities": planner_utilities,
        "utility_regrets": utility_regrets,
        "oracle_grid_points": int(candidates.size),
    }


def _record_seed(record: dict[str, Any]) -> Any:
    for field in ("training_seed", "seed", "mitigation_seed"):
        if field in record:
            return record[field]
    return None


def _finite_mean(records: Sequence[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in records if field in row and np.isfinite(row[field])]
    return float(np.mean(values)) if values else float("nan")


def aggregate_counterfactual_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse conflict/step records to one mean row per environment/planner/seed."""
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record.get("environment"), record.get("planner"), _record_seed(record))
        groups[key].append(record)
    output = []
    for (environment, planner, seed), rows in sorted(
        groups.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])
    ):
        result: dict[str, Any] = {
            "environment": environment,
            "planner": planner,
            "seed": seed,
            "n_records": len(rows),
        }
        for field in ("cf_arre", "cf_rmse", "one_step_mse", "decision_regret"):
            result[field] = _finite_mean(rows, field)
        output.append(result)
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_scalar(values: np.ndarray, fn, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    if values.size < 2:
        point = float(fn(values)) if values.size else float("nan")
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    samples = values[rng.integers(0, values.size, size=(n_boot, values.size))]
    estimates = np.asarray([fn(sample) for sample in samples], dtype=float)
    finite = estimates[np.isfinite(estimates)]
    point = float(fn(values))
    if finite.size == 0:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {"estimate": point, "ci_low": float(low), "ci_high": float(high)}


def _sign_flip_pvalue(differences: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, differences.size))
    permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (n_perm + 1))


def _correlation_row(seed_rows: Sequence[dict[str, Any]], metric: str, regret: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    values = np.asarray([row[metric] for row in seed_rows], dtype=float)
    valid = np.isfinite(values) & np.isfinite(regret)
    values, regrets = values[valid], regret[valid]
    if values.size < 2:
        interval = {"estimate": _spearman(values, regrets), "ci_low": float("nan"), "ci_high": float("nan")}
    else:
        indices = rng.integers(0, values.size, size=(n_boot, values.size))
        boot = np.asarray([_spearman(values[index], regrets[index]) for index in indices])
        finite = boot[np.isfinite(boot)]
        interval = {
            "estimate": _spearman(values, regrets),
            "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
            "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        }
    return {"metric": metric, "rho": interval["estimate"], "bootstrap_ci": interval, "n_seeds": int(values.size)}


def study_correlations(
    records: Iterable[dict[str, Any]],
    *,
    n_boot: int = 2000,
    n_permutations: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Report seed-level metric/regret correlations and paired ranking comparisons."""
    seed_rows = aggregate_counterfactual_records(records)
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["environment"], row["planner"])].append(row)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "replication_unit": "seed-level means; conflicts and horizons are not independent",
        "comparisons": [],
    }
    raw_pvalues = []
    for (environment, planner), rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        regret = np.asarray([row["decision_regret"] for row in rows], dtype=float)
        correlations = [
            _correlation_row(rows, metric, regret, rng, n_boot)
            for metric in ("cf_arre", "cf_rmse", "one_step_mse")
        ]
        by_metric = {row["metric"]: row for row in correlations}
        comparison = {
            "environment": environment,
            "planner": planner,
            "n_seeds": len(rows),
            "correlations": correlations,
            "delta_abs_rho": {},
        }
        baseline = by_metric["one_step_mse"]["rho"]
        for metric in ("cf_arre", "cf_rmse"):
            cf_rho = by_metric[metric]["rho"]
            metric_values = np.asarray([row[metric] for row in rows], dtype=float)
            baseline_values = np.asarray([row["one_step_mse"] for row in rows], dtype=float)
            valid = np.isfinite(metric_values) & np.isfinite(baseline_values) & np.isfinite(regret)
            metric_values, baseline_values, paired_regret = metric_values[valid], baseline_values[valid], regret[valid]
            if paired_regret.size < 2 or not np.isfinite(cf_rho) or not np.isfinite(baseline):
                deltas = np.array([], dtype=float)
                delta_ci = {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
                p_value = float("nan")
                cliff = (float("nan"), "undefined")
            else:
                indices = rng.integers(0, paired_regret.size, size=(n_boot, paired_regret.size))
                delta_samples = []
                for index in indices:
                    delta_samples.append(
                        abs(_spearman(metric_values[index], paired_regret[index]))
                        - abs(_spearman(baseline_values[index], paired_regret[index]))
                    )
                deltas = np.asarray(delta_samples, dtype=float)
                point_delta = abs(_spearman(metric_values, paired_regret)) - abs(_spearman(baseline_values, paired_regret))
                finite = deltas[np.isfinite(deltas)]
                delta_ci = {
                    "estimate": float(point_delta),
                    "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
                    "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
                }
                loss_cf = np.abs(_average_ranks(metric_values) - _average_ranks(paired_regret))
                loss_base = np.abs(_average_ranks(baseline_values) - _average_ranks(paired_regret))
                p_value = _sign_flip_pvalue(loss_base - loss_cf, rng, n_permutations)
                cliff = cliffs_delta(loss_base, loss_cf)
            comparison["delta_abs_rho"][metric] = {
                "vs": "one_step_mse",
                "paired_bootstrap_ci": delta_ci,
                "sign_flip_p_value": p_value,
                "cliffs_delta_on_rank_loss": cliff[0],
                "cliffs_label": cliff[1],
            }
            raw_pvalues.append(p_value)
        result["comparisons"].append(comparison)
    finite_p = np.asarray([p if np.isfinite(p) else 1.0 for p in raw_pvalues], dtype=float)
    holm = holm_correction(finite_p)
    bh = bh_fdr(finite_p)
    position = 0
    for comparison in result["comparisons"]:
        for metric in ("cf_arre", "cf_rmse"):
            entry = comparison["delta_abs_rho"][metric]
            entry["holm_p_value"] = float(holm[position]) if raw_pvalues[position] == raw_pvalues[position] else float("nan")
            entry["bh_fdr_p_value"] = float(bh[position]) if raw_pvalues[position] == raw_pvalues[position] else float("nan")
            position += 1
    return result


def _load_records(paths: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(payload)
        else:
            records.extend(payload.get("records", []))
    return records


def _markdown_report(study: dict[str, Any]) -> str:
    lines = [
        "# Counterfactual regret study",
        "",
        f"Replication unit: {study['replication_unit']}",
        "",
        "| Environment | Planner | Seeds | Metric | Spearman rho | CI |",
        "|---|---|---:|---|---:|---|",
    ]
    for comparison in study["comparisons"]:
        for correlation in comparison["correlations"]:
            ci = correlation["bootstrap_ci"]
            lines.append(
                f"| {comparison['environment']} | {comparison['planner']} | {comparison['n_seeds']} | "
                f"{correlation['metric']} | {correlation['rho']:.6g} | "
                f"[{ci['ci_low']:.6g}, {ci['ci_high']:.6g}] |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="counterfactuals.json file(s)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    study = study_correlations(_load_records(args.input), seed=args.seed)
    Path(args.out_json).write_text(json.dumps(study, indent=2, default=_json_default), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(_markdown_report(study), encoding="utf-8")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
