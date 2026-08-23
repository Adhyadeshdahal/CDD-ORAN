"""Phase 4 ablation driver, case-bank validator, and seed-level statistics.

The normal entry point is intentionally orchestration-only: it resolves recipes and
validates reusable artifacts before a later orchestrator launches train/evaluate jobs.
``--dry-run`` and ``--self-check`` are CPU-only and never import a model or start a job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdd_oran.analysis.stats import (  # noqa: E402
    bh_fdr,
    cliffs_delta,
    compare_seeds,
    holm_correction,
)

DEFAULT_ENVS = ("EnvironmentI", "EnvironmentII")
DEFAULT_SEEDS = tuple(range(45, 55))
DEFAULT_VARIANTS = (
    "mlp",
    "hard_mask",
    "oracle",
    "corrupted_oracle",
    "soft_mask",
    "graph_reg",
    "bootstrap_ensemble",
    "fixed_graph_ensemble",
    "full",
)
PLANNERS = ("QACM", "ModelBasedCEM", "ModelBasedMPPI", "ModelBasedMCTS")
CASE_SEED = 20260824
CORRUPTION_EDGES = {
    "EnvironmentI": ("KPI1<-P2", "KPI2<-P3"),
    "EnvironmentII": ("KPI1<-P2", "KPI2<-P3", "KPI41<-P6", "KPI5<-P8"),
}


@dataclass(frozen=True)
class VariantRecipe:
    variant: str
    config: str
    model_kind: str
    train_required: bool
    graph_source: str
    overrides: dict[str, Any]
    reuse_from: str | None = None
    corrupt_edges: tuple[str, ...] = ()
    posterior_artifact: str | None = None
    enumeration_artifact: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PlanRow:
    environment: str
    seed: int
    variant: str
    action: str
    source_variant: str | None
    recipe: VariantRecipe


def _env_token(environment: str) -> str:
    if environment not in DEFAULT_ENVS:
        raise ValueError(f"unsupported P4 environment: {environment}")
    return "i" if environment == "EnvironmentI" else "ii"


def _artifact_path(root: Path, kind: str, environment: str, seed: int) -> str:
    return str(root / kind / environment / str(seed) / f"{kind}.npz")


def variant_recipe(environment: str, variant: str, artifact_root: str | Path = "artifacts/p4", seed: int = 45) -> VariantRecipe:
    """Return the normative, explicit recipe for one environment/seed/variant."""
    token = _env_token(environment)
    root = Path(artifact_root)
    cdl = f"configs/env_{token}_cdl.yaml"
    mlp = f"configs/env_{token}_mlp.yaml"
    posterior = _artifact_path(root, "posterior", environment, seed)
    enumeration = _artifact_path(root, "enum", environment, seed)
    common_planner = {
        "planner.n_horizon": 1,
        "planner.cem.n_candidate": 256,
        "planner.cem.n_top": 128,
        "planner.cem.n_iter": 20,
        "planner.mppi.n_samples": 2000,
        "planner.mppi.temperature": 0.6,
        "planner.mppi.noise_sigma": 0.1,
        "planner.mcts.n_simulations": 3000,
        "planner.mcts.ucb_c": 1.5,
    }

    recipes = {
        "mlp": VariantRecipe(
            variant="mlp", config=mlp, model_kind="mlp", train_required=True,
            graph_source="source-hard", overrides={
                **common_planner, "model.generative_fc_dims": [64, 64],
                "model.feature_fc_dims": [64, 64],
            }, reuse_from="source-hard",
            notes="matched dense 64-64 MLP; graph is supplied only for the current MLP API",
        ),
        # TODO(orchestrator): the hard_mask conference architecture was deleted (feat/v2).
        # This baseline recipe's "model.dynamics_mode" override no longer resolves against a
        # config field. Decide whether to drop this baseline or reconstruct it from git history.
        "hard_mask": VariantRecipe(
            variant="hard_mask", config=cdl, model_kind="cdl", train_required=True,
            graph_source="causal", overrides={
                **common_planner, "model.dynamics_mode": "hard_mask",
                "model.predict_members": 1, "model.posterior_artifact": None,
                "model.enumeration_graph": None,
            }, notes="DELETED conference architecture; recipe kept for the orchestrator to resolve",
        ),
        "oracle": VariantRecipe(
            variant="oracle", config=cdl, model_kind="cdl", train_required=False,
            graph_source="oracle", overrides={**common_planner, "graph_override": "oracle"},
            reuse_from="hard_mask", notes="true environment graph; conflict bank remains fixed",
        ),
        "corrupted_oracle": VariantRecipe(
            variant="corrupted_oracle", config=cdl, model_kind="cdl", train_required=False,
            graph_source="oracle", overrides={**common_planner, "graph_override": "oracle"},
            reuse_from="hard_mask", corrupt_edges=CORRUPTION_EDGES[environment],
            notes="immutable direct actuator-edge corruption; prediction mask only",
        ),
        # TODO(orchestrator): "soft_mask" was never an implemented dynamics_mode value; with
        # the mode field removed this override does not resolve. Resolve or drop this baseline.
        "soft_mask": VariantRecipe(
            variant="soft_mask", config=cdl, model_kind="cdl", train_required=True,
            graph_source="causal", overrides={
                **common_planner, "model.dynamics_mode": "soft_mask",
                "model.posterior_artifact": posterior, "model.predict_members": 1,
                "model.residual_bound": 0.0, "model.residual_l1": 0.0,
                "model.residual_l2": 0.0,
            }, posterior_artifact=posterior,
            notes="trained continuous posterior marginal weights; no binary threshold",
        ),
        "graph_reg": VariantRecipe(
            variant="graph_reg", config=mlp, model_kind="mlp", train_required=True,
            graph_source="source-hard", overrides={
                **common_planner, "model.generative_fc_dims": [64, 64],
                "model.feature_fc_dims": [64, 64], "graph_reg_lambda": 1e-2,
                "graph_reg_source": posterior,
            }, reuse_from="source-hard", posterior_artifact=posterior,
            notes="dense predictor with fixed non-edge Jacobian penalty",
        ),
        "bootstrap_ensemble": VariantRecipe(
            variant="bootstrap_ensemble", config=cdl,
            model_kind="cdl", train_required=True, graph_source="enumeration",
            overrides={
                **common_planner,
                "model.posterior_artifact": posterior, "model.enumeration_graph": enumeration,
                "model.residual_bound": 0.0, "model.residual_l1": 0.0,
                "model.residual_l2": 0.0, "model.predict_members": 8,
                "planner.ensemble_aggregator": "mean", "planner.ensemble_quantile": 0.9,
                "planner.disagreement_penalty": 0.0, "planner.ood_threshold": None,
            }, posterior_artifact=posterior, enumeration_artifact=enumeration,
            notes="posterior structures only; measured residual must remain separate",
        ),
        "fixed_graph_ensemble": VariantRecipe(
            variant="fixed_graph_ensemble", config=cdl, model_kind="cdl", train_required=True,
            graph_source="causal", overrides={
                **common_planner, "fixed_graph_members": 8,
                "fixed_graph_member_seed_stride": 1000,
                "planner.ensemble_aggregator": "mean",
                "planner.disagreement_penalty": 0.0, "planner.ood_threshold": None,
            }, notes="eight independent hard-mask dynamics members; outer seed is replication unit",
        ),
        "full": VariantRecipe(
            variant="full", config=cdl,
            model_kind="cdl", train_required=True, graph_source="enumeration",
            overrides={
                **common_planner,
                "model.posterior_artifact": posterior, "model.enumeration_graph": enumeration,
                "model.predict_members": 8, "planner.ensemble_aggregator": "quantile",
                "planner.ensemble_quantile": 0.9, "planner.disagreement_penalty": 0.0,
                "planner.ood_threshold": None,
            }, posterior_artifact=posterior, enumeration_artifact=enumeration,
            notes="primary calibrated structure + bounded residual + robust return model",
        ),
    }
    if variant not in recipes:
        raise ValueError(f"unknown P4 variant {variant!r}; choose from {DEFAULT_VARIANTS}")
    return recipes[variant]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def build_oracle_graph(env, include_action: bool = True):
    """Return the true graph in the predictor layout ``[child, parent]``."""
    import torch

    graph = torch.as_tensor(np.asarray(env.true_adj_matrix), dtype=torch.bool)
    if include_action:
        graph = torch.cat([graph, torch.ones((graph.shape[0], 1), dtype=torch.bool)], dim=1)
    return graph


def build_corrupted_oracle_graph(env, edge_specs: Sequence[str]):
    """Build an oracle graph and clear only the named child/parent cells."""
    from cdd_oran.analysis.graph_baselines import remove_edges

    graph, removed = remove_edges(build_oracle_graph(env), edge_specs, env)
    return graph, removed


def soft_mask_weights(posterior: Any, state_dim: int | None = None) -> np.ndarray:
    """Normalize continuous edge probabilities and force self/action slots on."""
    weights = np.asarray(posterior, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"posterior weights must be 2-D, got {weights.shape}")
    if state_dim is None:
        state_dim = weights.shape[0]
    if weights.shape == (state_dim, state_dim):
        weights = np.concatenate([weights, np.ones((state_dim, 1))], axis=1)
    if weights.shape != (state_dim, state_dim + 1):
        raise ValueError(f"posterior weights must be ({state_dim}, {state_dim}) or ({state_dim}, {state_dim + 1})")
    weights = np.clip(weights, 0.0, 1.0)
    weights[:, -1] = 1.0
    rows = np.arange(state_dim)
    weights[rows, rows] = 1.0
    return weights


def graph_regularizer_exclusion_mask(weights: Any) -> np.ndarray:
    """Return the fixed non-edge penalty mask, excluding self and action slots."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] not in (weights.shape[0], weights.shape[0] + 1):
        raise ValueError(f"weights must be square or square-plus-action, got {weights.shape}")
    mask = 1.0 - np.clip(weights, 0.0, 1.0)
    if mask.shape[1] == mask.shape[0] + 1:
        mask[:, -1] = 0.0
    rows = np.arange(mask.shape[0])
    mask[rows, rows] = 0.0
    return mask


def graph_regularizer(sensitivity: Any, weights: Any) -> float:
    sensitivity = np.asarray(sensitivity, dtype=float)
    mask = graph_regularizer_exclusion_mask(weights)
    if sensitivity.shape != mask.shape:
        raise ValueError(f"sensitivity and graph mask shapes differ: {sensitivity.shape} vs {mask.shape}")
    return float(np.mean((sensitivity * mask) ** 2))


def validate_member_shape(member_returns: Any, members: int | None = None) -> np.ndarray:
    values = np.asarray(member_returns, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(f"member returns must be (m,) or (m, batch), got {values.shape}")
    if members is not None and values.shape[0] != members:
        raise ValueError(f"expected {members} members, got {values.shape[0]}")
    return values


def aggregate_member_returns(member_returns: Any, method: str = "mean", quantile: float = 0.9) -> np.ndarray:
    values = validate_member_shape(member_returns)
    if method == "mean":
        return values.mean(axis=0)
    if method == "quantile":
        return np.quantile(values, quantile, axis=0)
    if method == "kappa":
        return values.mean(axis=0) + values.std(axis=0)
    raise ValueError(f"unknown member return aggregator {method!r}")


def planner_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    fields = ("environment", "variant", "seed", "planner", "case_id", "conflict_id")
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"raw record missing pairing fields: {missing}")
    return tuple(record[field] for field in fields)


def validate_planner_coverage(records: Iterable[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for record in records:
        grouped[(record.get("environment"), record.get("variant"), record.get("seed"), record.get("case_id"), record.get("conflict_id"))].add(record.get("planner"))
    for key, planners in grouped.items():
        missing = set(PLANNERS) - planners
        if missing:
            raise ValueError(f"incomplete four-planner output for {key}: missing {sorted(missing)}")


def merge_raw_records(existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge resumable raw rows and reject duplicate logical units."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in list(existing) + list(incoming):
        key = planner_record_key(record)
        if key in merged:
            raise ValueError(f"duplicate raw record key: {key}")
        merged[key] = record
    return [merged[key] for key in sorted(merged, key=lambda value: tuple(str(part) for part in value))]


def _case_config(environment: str, case_seed: int):
    from dataclasses import replace

    from cdd_oran.config import load_config

    config = load_config(f"configs/env_{_env_token(environment)}_cdl.yaml")
    return replace(config, seed=case_seed, mitigation_seed=case_seed, param_ranges="ood", evaluation_param_ranges="ood", device="cpu")


def generate_case_bank(environment: str, case_seed: int = CASE_SEED, n_cases: int = 30) -> dict[str, Any]:
    """Generate deterministic true-graph case/conflict keys and normalized states."""
    from cdd_oran.conflicts import denormalize_params, detect_conflict_edges, state_to_tensor
    from cdd_oran.envs import get_env
    from cdd_oran.utils.seeding import seed_everything

    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    cfg = _case_config(environment, case_seed)
    seed_everything(case_seed, deterministic=False)
    env = get_env(cfg)
    cases = []
    true_graph = np.asarray(env.true_adj_matrix, dtype=np.int8)
    for case_id in range(n_cases):
        env.reset()
        observed = state_to_tensor(env.get_state()).numpy().astype(np.float32)
        raw_params = denormalize_params(observed[: env.num_params], env).astype(float)
        edges = detect_conflict_edges(true_graph, env)
        conflicts = []
        for conflict_id, edge in enumerate(edges):
            param_id = int(edge["param_id"])
            low, high = env.params[param_id].get_threshold()
            conflicts.append({
                "conflict_id": conflict_id,
                "primary_xapp_id": int(edge["primary_xapp_id"]),
                "param_id": param_id,
                "conflict_xapp_ids": [int(value) for value in edge["conflict_xapp_ids"]],
                "action_low": float(low),
                "action_high": float(high),
            })
        cases.append({
            "case_id": case_id,
            "state": observed.tolist(),
            "raw_params": raw_params.tolist(),
            "conflicts": conflicts,
        })
    payload = {
        "schema_version": 1,
        "environment": environment,
        "case_seed": int(case_seed),
        "n_cases": int(n_cases),
        "evaluation_param_ranges": "ood",
        "state_dim": int(env.get_state_dim()),
        "param_ranges": [list(param.get_threshold()) for param in env.params],
        "true_graph": true_graph.tolist(),
        "cases": cases,
    }
    payload["digest"] = _digest(payload)
    return payload


def write_case_bank(bank: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bank)
    expected = payload.pop("digest", None)
    actual = _digest(payload)
    if expected != actual:
        raise ValueError(f"case bank digest mismatch before write: {expected} != {actual}")
    payload["digest"] = actual
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    states = np.asarray([case["state"] for case in payload["cases"]], dtype=np.float32)
    raw = np.asarray([case["raw_params"] for case in payload["cases"]], dtype=np.float32)
    np.savez_compressed(path.with_suffix(".npz"), states=states, raw_params=raw, digest=actual)
    return path


def validate_case_bank(path: str | Path, environment: str | None = None, case_seed: int | None = None) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.get("digest")
    unsigned = dict(payload)
    unsigned.pop("digest", None)
    if digest != _digest(unsigned):
        raise ValueError(f"case bank digest mismatch: {path}")
    if environment is not None and payload.get("environment") != environment:
        raise ValueError(f"case bank environment mismatch: expected {environment}, got {payload.get('environment')}")
    if case_seed is not None and int(payload.get("case_seed")) != int(case_seed):
        raise ValueError("case bank seed mismatch")
    if int(payload.get("n_cases", 0)) != len(payload.get("cases", [])):
        raise ValueError("case bank n_cases does not match cases")
    keys = []
    for case in payload["cases"]:
        for conflict in case["conflicts"]:
            keys.append((case["case_id"], conflict["conflict_id"]))
    if len(keys) != len(set(keys)):
        raise ValueError("case bank contains duplicate case/conflict keys")
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as arrays:
            if str(arrays["digest"].item()) != digest:
                raise ValueError("case bank NPZ digest mismatch")
            if arrays["states"].shape[0] != len(payload["cases"]):
                raise ValueError("case bank NPZ state count mismatch")
    return payload


def _raw_utility(record: dict[str, Any]) -> float:
    if "utility" in record:
        return float(record["utility"])
    values = record.get("utility_by_xapp")
    if isinstance(values, dict):
        values = list(values.values())
    if not values:
        raise ValueError("raw record lacks utility or utility_by_xapp")
    return float(np.mean(np.asarray(values, dtype=float)))


def _raw_satisfaction(record: dict[str, Any]) -> float:
    values = record.get("satisfaction", record.get("satisfied_by_xapp"))
    if values is None:
        return float("nan")
    if isinstance(values, dict):
        values = list(values.values())
    return float(np.mean(np.asarray(values, dtype=float)))


def aggregate_seed_results(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        planner_record_key(record)
        groups[(record["environment"], record["variant"], int(record["seed"]), record["planner"])].append(record)
    rows = []
    for (environment, variant, seed, planner), group in sorted(groups.items()):
        case_keys = { (row["case_id"], row["conflict_id"]) for row in group }
        rows.append({
            "environment": environment,
            "variant": variant,
            "seed": seed,
            "planner": planner,
            "utility": float(np.mean([_raw_utility(row) for row in group])),
            "satisfaction": float(np.nanmean([_raw_satisfaction(row) for row in group])),
            "n_cases_conflicts": len(case_keys),
            "case_keys": sorted([list(key) for key in case_keys]),
        })
    return rows


def _paired_values(rows: Sequence[dict[str, Any]], variant_a: str, variant_b: str, field: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    a = {(row["seed"]): row for row in rows if row["variant"] == variant_a}
    b = {(row["seed"]): row for row in rows if row["variant"] == variant_b}
    if set(a) != set(b):
        raise ValueError(f"unpaired seeds for {variant_a} vs {variant_b}: {sorted(set(a) ^ set(b))}")
    seeds = sorted(a)
    return np.asarray([a[seed][field] for seed in seeds]), np.asarray([b[seed][field] for seed in seeds]), seeds


def stats_report(records: Iterable[dict[str, Any]], full_variant: str = "full") -> dict[str, Any]:
    """Compute full-vs-each-baseline statistics per environment and planner."""
    records = list(records)
    rows = aggregate_seed_results(records)
    validate_planner_coverage(records)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["planner"])].append(row)
    comparisons = []
    all_pvalues = []
    pending = []
    for (environment, planner), group in sorted(grouped.items()):
        variants = sorted({row["variant"] for row in group} - {full_variant})
        for baseline in variants:
            full, base, seeds = _paired_values(group, full_variant, baseline, "utility")
            sat_full, sat_base, sat_seeds = _paired_values(group, full_variant, baseline, "satisfaction")
            if seeds != sat_seeds:
                raise ValueError("utility and satisfaction seed pairing differs")
            utility_stats = compare_seeds(full, base, seed=0)
            satisfaction_stats = compare_seeds(sat_full, sat_base, seed=0)
            item: dict[str, Any] = {
                "environment": environment,
                "planner": planner,
                "full_variant": full_variant,
                "baseline_variant": baseline,
                "seeds": seeds,
                "utility": utility_stats,
                "satisfaction": satisfaction_stats,
            }
            pending.append(item)
            all_pvalues.append(float(utility_stats["wilcoxon"]["p_value"]))
            comparisons.append(item)
    # Confirmatory Holm family: all baselines for each environment/planner.
    for environment, planner in sorted(grouped):
        family = [item for item in pending if item["environment"] == environment and item["planner"] == planner]
        adjusted = holm_correction([item["utility"]["wilcoxon"]["p_value"] for item in family])
        for item, value in zip(family, adjusted, strict=True):
            item["utility"]["holm_p_value"] = float(value)
    bh_values = bh_fdr(all_pvalues)
    for item, value in zip(pending, bh_values, strict=True):
        item["utility"]["bh_fdr_p_value"] = float(value)
    return {
        "replication_unit": "independent paired training seeds",
        "full_variant": full_variant,
        "comparisons": comparisons,
        "n_comparisons": len(comparisons),
        "correction_families": {
            "holm": "all full-vs-baseline comparisons within each (environment, planner)",
            "bh": "all full-vs-baseline comparisons across both environments and four planners",
        },
    }


def write_stats_report(records: Iterable[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = stats_report(records)
    (output / "stats.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    for (environment, planner), comparisons in _group_comparisons(report["comparisons"]):
        lines = [
            f"# {environment} — {planner}", "",
            "Replication unit: independent paired training seeds.", "",
            "| Full | Baseline | n | Δ utility | 95% CI | Holm p | BH p | Δ satisfaction |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for item in comparisons:
            utility = item["utility"]
            sat = item["satisfaction"]
            lines.append(
                f"| {item['full_variant']} | {item['baseline_variant']} | {utility['n_seeds']} | "
                f"{utility['mean_diff']:.6g} | [{utility['ci_low']:.6g}, {utility['ci_high']:.6g}] | "
                f"{utility['holm_p_value']:.6g} | {utility['bh_fdr_p_value']:.6g} | {sat['mean_diff']:.6g} |"
            )
        (tables / f"{environment}_{planner}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _group_comparisons(comparisons):
    grouped = defaultdict(list)
    for item in comparisons:
        grouped[(item["environment"], item["planner"])].append(item)
    return sorted(grouped.items())


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("records", payload.get("raw_records", []))


def _load_reuse_map(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("runs", payload)


def _reuse_key(environment: str, seed: int, variant: str) -> str:
    return f"{environment}|{seed}|{variant}"


def plan_rows(environments: Sequence[str], seeds: Sequence[int], variants: Sequence[str], artifact_root: str | Path, reuse_map: dict[str, Any]) -> list[PlanRow]:
    rows = []
    for environment in environments:
        for seed in seeds:
            for variant in variants:
                recipe = variant_recipe(environment, variant, artifact_root, seed)
                source = recipe.reuse_from
                if recipe.train_required and reuse_map.get(_reuse_key(environment, seed, variant)):
                    action = "REUSE existing"
                elif recipe.train_required:
                    action = "TRAIN + EVAL"
                elif source:
                    action = f"REUSE {source}"
                else:
                    action = "EVAL reuse"
                rows.append(PlanRow(environment, seed, variant, action, source, recipe))
    return rows


def print_dry_run(rows: Sequence[PlanRow], case_bank_template: str, output: str, reuse_map: dict[str, Any]) -> None:
    print("P4 ablation dry-run (no training/evaluation launched)")
    print(f"Plan rows: {len(rows)} ({len(set(row.environment for row in rows))} environments × {len(set(row.seed for row in rows))} seeds × {len(set(row.variant for row in rows))} variants)")
    print(f"Case banks: {case_bank_template}; output: {output}")
    print("\nResolved plan:")
    for row in rows:
        recipe = row.recipe
        print(f"{row.environment:14} seed={row.seed:3d} {row.variant:22} {row.action:18} graph={recipe.graph_source:11} config={recipe.config}")
    print("\nGPU cost ordering (cheapest decisive subset first):")
    print("1. CPU case-bank/schema validation and dry-run manifest")
    print("2. Core: matched MLP vs hard-mask CDL (EnvironmentI, then EnvironmentII; all ten seeds; all four planners)")
    print("3. Oracle and corrupted-oracle evaluations reusing hard-mask checkpoints")
    print("4. Soft-mask and graph-regularized dense variants")
    print("5. Posterior validation, bootstrap ensemble, and full model")
    print("6. Fixed-graph dynamics ensemble last (8 nested members; 160 extra training runs)")
    print(f"Reuse-map entries loaded: {len(reuse_map)}")


def self_check() -> None:
    high = np.asarray([1.0, 1.1, 1.2, 1.3])
    low = np.asarray([0.0, 0.1, 0.2, 0.3])
    result = compare_seeds(high, low, seed=3)
    assert result["n_seeds"] == 4
    assert result["mean_diff"] == 1.0
    assert result["cliffs_delta"] == 1.0
    assert np.allclose(holm_correction([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
    assert np.allclose(bh_fdr([0.01, 0.02, 0.5]), [0.03, 0.03, 0.5])
    assert cliffs_delta(high, low)[0] == 1.0
    try:
        _paired_values(
            [
                {"variant": "full", "seed": 1, "utility": 1},
                {"variant": "hard_mask", "seed": 2, "utility": 1},
            ],
            "full", "hard_mask", "utility",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired seed arrays must be rejected")
    print("ablation_sweep self-check passed: compare_seeds/Holm/BH/Cliff/unpaired guard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--stats-input", help="raw JSON/JSONL records for CPU statistics")
    parser.add_argument("--stats-out", default="results/p4")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--case-bank", default="artifacts/p4/case_banks/{env}.json")
    parser.add_argument("--artifact-root", default="artifacts/p4")
    parser.add_argument("--output", default="results/p4")
    parser.add_argument("--reuse-map")
    parser.add_argument("--case-seed", type=int, default=CASE_SEED)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--train-missing", action="store_true", help="allow a later orchestrator to launch missing jobs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.stats_input:
        write_stats_report(_load_records(args.stats_input), args.stats_out)
        print(f"stats written to {Path(args.stats_out) / 'stats.json'}")
        return 0
    environments = tuple(value.strip() for value in args.envs.split(",") if value.strip())
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("--seeds must contain unique non-empty integers")
    for environment in environments:
        for variant in variants:
            variant_recipe(environment, variant, args.artifact_root, seeds[0])
    reuse_map = _load_reuse_map(args.reuse_map)
    rows = plan_rows(environments, seeds, variants, args.artifact_root, reuse_map)
    if args.dry_run:
        print_dry_run(rows, args.case_bank, args.output, reuse_map)
        return 0
    # The orchestrator may opt into job launch after validating this manifest. This guard
    # prevents an accidental GPU sweep from an otherwise CPU-only driver invocation.
    if not args.train_missing:
        raise ValueError("non-dry-run requires --train-missing and an external job launcher")
    raise RuntimeError("job launch is intentionally delegated; use the printed manifest with the orchestrator")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ablation_sweep: {error}", file=sys.stderr)
        raise SystemExit(1) from error
