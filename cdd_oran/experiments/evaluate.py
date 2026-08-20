"""
evaluate.py  –  xApp Utility Curves with Algorithm Parameter Selections
------------------------------------------------------------------------
Pipeline
  1. For N environment steps:
     a. Get current state from env
     b. Use learned CDL module to detect conflict edges (kpi_node → param_id)
     c. For each conflict edge, run every algorithm → get action → compute utility
     d. Log mean utility per algorithm per step to TensorBoard (all on ONE graph)
     e. Step the environment forward
  2. TensorBoard logs:
     - Mean utility per algorithm per environment step (single graph, different colours)
     - Conflict count per step
"""

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cdd_oran.analysis.graph_baselines import build_override_graph, collect_transitions
from cdd_oran.config import DEFAULT_CONFIG, ExperimentConfig
from cdd_oran.conflicts import (
    compute_utility,
    denormalize_params,
    detect_conflict_edges,
    state_to_tensor,
)
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.planners import get_planners
from cdd_oran.utils.runs import read_metrics, write_metrics
from cdd_oran.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def main(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    run_dir=None,
    graph_run=None,
    graph_cfg=None,
    graph_override="causal",
):
    if run_dir is None:
        raise ValueError("An experiment run directory is required")

    cfg = replace(
        cfg,
        seed=cfg.mitigation_seed,
        param_ranges=cfg.evaluation_param_ranges,
    )
    seed_everything(cfg.seed, cfg.deterministic)
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "checkpoint.pt"

    env = get_env(cfg)
    act_dim = env.get_action_dim()
    model = get_model(cfg, env)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_model(checkpoint_path)

    if cfg.model_kind == "cdl":
        cdl = model
    else:
        if graph_run is None or graph_cfg is None:
            raise ValueError("MLP evaluation requires --graph-run pointing to a trained CDL run")
        if graph_cfg.model_kind != "cdl":
            raise ValueError("--graph-run must contain a CDL configuration")
        if graph_cfg.environment != cfg.environment:
            raise ValueError("--graph-run must use the same environment")
        cdl = get_model(replace(graph_cfg, model_kind="cdl"), env)
        graph_checkpoint = Path(graph_run) / "checkpoint.pt"
        if not graph_checkpoint.exists():
            raise FileNotFoundError(f"Graph checkpoint not found: {graph_checkpoint}")
        cdl.load_model(graph_checkpoint)

    if graph_override != "causal":
        transitions = None
        if graph_override == "correlation":
            transitions = collect_transitions(get_env(cfg), cfg.model.batch_size, act_dim)
        override = build_override_graph(graph_override, cdl, env, transitions)
        action_col = torch.ones((override.shape[0], 1), dtype=torch.bool, device=override.device)
        full_graph = torch.cat([override, action_col], dim=1)
        cdl.get_binary_graph = lambda threshold=None, g=full_graph: g
        logger.info("Graph override active: %s (%d edges)", graph_override, int(override.sum()))

    algorithms = get_planners(cfg, model, env)
    utility_fns = env.get_utility_fns()

    KPI_THRESHOLDS, MEAN_STD_KPIS = env.get_thresholds_stds()
    algorithm_names = [algo.name for algo in algorithms]
    utilities_data = {
        "version": 1,
        "environment": cfg.environment,
        "model_kind": cfg.model_kind,
        "algorithm_names": algorithm_names,
        "param_thresholds": [list(param.get_threshold()) for param in env.params],
        "kpi_thresholds": KPI_THRESHOLDS,
        "mean_std_kpis": [list(mean_std) for mean_std in MEAN_STD_KPIS],
        "steps": [],
    }
    sweep_points = 101

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=run_dir / "tensorboard" / f"evaluate-{timestamp}")

    logger.info("Running evaluation for %d environment steps", cfg.num_steps)

    env.reset()
    all_utilities: dict[str, list[float]] = {algo.name: [] for algo in algorithms}

    for global_step in range(cfg.num_steps):
        state_dict = env.get_state()
        state_t = state_to_tensor(state_dict).to(cfg.device)
        raw_params = denormalize_params(state_t[: env.num_params].cpu().numpy(), env)

        causal_graph = cdl.get_binary_graph()[:, :-1].cpu().detach().numpy()
        edges = detect_conflict_edges(causal_graph, env)

        num_conflicts = len(edges)
        writer.add_scalar("conflicts/count", num_conflicts, global_step)

        if num_conflicts == 0:
            logger.info("Step %d: no conflicts detected", global_step)
        else:
            logger.info("Step %d: %d conflict(s) detected", global_step, num_conflicts)

        step_utilities: dict[str, list[float]] = {algo.name: [] for algo in algorithms}
        step_panels = []

        for edge in edges:
            param_id = edge["param_id"]
            primary_xapp_id = edge["primary_xapp_id"]
            xapps_in_conflict = edge["xapps_in_conflict"]
            conflict_xapp_ids = edge["conflict_xapp_ids"]
            param_threshold = env.params[param_id].get_threshold()
            sweep = np.linspace(*param_threshold, num=sweep_points)
            curves = [
                {
                    "xapp_id": xid,
                    "values": [
                        compute_utility(utility_fns[xid], raw_params, param_id, value)
                        for value in sweep
                    ],
                }
                for xid in conflict_xapp_ids
                if xid < len(utility_fns)
            ]
            panel = {
                "step": global_step,
                "param_id": param_id,
                "primary_xapp_id": primary_xapp_id,
                "sweep": sweep.tolist(),
                "curves": curves,
                "algo_names": algorithm_names,
                "algo_actions": [],
                "planner_utilities": {},
            }

            num_xapps = len(xapps_in_conflict)
            # Under review: this currently normalizes every conflict weight back to one.
            weights = np.ones(num_xapps)
            for idx, xapp in enumerate(xapps_in_conflict):
                if xapp == env.xapps[primary_xapp_id]:
                    weights[idx] *= 1.0
            weights = weights / weights.sum() * num_xapps

            for algo in algorithms:
                state_for_algo = state_t.clone().detach()

                action = algo.act(
                    current_state=state_for_algo,
                    conflict_param_index=param_id,
                    xapps_under_conflict=xapps_in_conflict,
                    weights_per_xapps=weights.tolist(),
                    scaling_term=10,
                )
                raw_val = env.action_to_param(action)[1]
                panel["algo_actions"].append(float(raw_val))

                planner_values = []
                for xid in conflict_xapp_ids:
                    if xid >= len(utility_fns):
                        continue
                    u = compute_utility(utility_fns[xid], raw_params, param_id, raw_val)
                    step_utilities[algo.name].append(u)
                    planner_values.append(u)
                panel["planner_utilities"][algo.name] = planner_values

                if primary_xapp_id < len(utility_fns):
                    logger.info(
                        "%s x%d->p%d action=%.4f utility(primary)=%.4f",
                        algo.name,
                        primary_xapp_id,
                        param_id,
                        raw_val,
                        compute_utility(
                            utility_fns[primary_xapp_id], raw_params, param_id, raw_val
                        ),
                    )

                else:
                    logger.info(
                        "%s x%d->p%d action=%.4f", algo.name, primary_xapp_id, param_id, raw_val
                    )

            step_panels.append(panel)

        scalars_for_step = {}
        for algo in algorithms:
            vals = step_utilities[algo.name]
            if vals:
                mean_u = float(np.mean(vals))
                scalars_for_step[algo.name] = mean_u
                all_utilities[algo.name].extend(vals)
                logger.info("Mean utility %s: %.4f", algo.name, mean_u)

        if scalars_for_step:
            writer.add_scalars("mean_utility/all_algorithms", scalars_for_step, global_step)

        utilities_data["steps"].append({"step": global_step, "panels": step_panels})

        neutral_action = np.zeros(act_dim)
        # Intentional for independent conflict evaluation; action proposals are not applied.
        env.step(neutral_action)

        writer.flush()

    writer.close()
    metrics = read_metrics(run_dir)
    metrics["evaluation"] = {
        "planner_mean_utilities": {
            name: float(np.mean(values)) if values else None
            for name, values in all_utilities.items()
        }
    }
    write_metrics(run_dir, metrics)
    (run_dir / "utilities.json").write_text(json.dumps(utilities_data, indent=2))
    logger.info("Evaluation metrics saved to %s", run_dir / "metrics.json")
    return 0
