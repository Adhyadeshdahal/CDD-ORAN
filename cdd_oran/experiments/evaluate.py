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

import hashlib
import json
import logging
import random
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cdd_oran.analysis.graph_baselines import (
    build_override_graph,
    collect_transitions,
    remove_edges,
)
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


@contextmanager
def preserve_probe_rng(model):
    """Run the residual probe without perturbing any RNG the planners depend on.

    Snapshots python/numpy/torch GLOBAL RNG *and* the posterior sampler's PRIVATE numpy
    Generator (``model.sampler.rng``), restoring all of them on exit. The probe draws a
    structure (advancing the sampler) and reseeds/consumes env randomness; without this the
    planners would see a different structure sequence with vs without the diagnostic
    (review v2 BLOCKER A)."""
    global_snapshot = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    sampler = getattr(model, "sampler", None)
    sampler_state = (
        sampler.rng.bit_generator.state
        if sampler is not None and hasattr(sampler, "rng")
        else None
    )
    try:
        yield
    finally:
        random.setstate(global_snapshot[0])
        np.random.set_state(global_snapshot[1])
        torch.set_rng_state(global_snapshot[2])
        if global_snapshot[3] is not None:
            torch.cuda.set_rng_state_all(global_snapshot[3])
        if sampler_state is not None:
            sampler.rng.bit_generator.state = sampler_state


def _planner_seed(base_seed, global_step, conflict_index, planner_name):
    """Stable per-(step, conflict, planner) seed for common random numbers (CRN).

    Reseeding the global torch/numpy RNG with this before each ``planner.act`` call
    guarantees the SAME conflict gets the SAME random draws at every corruption level
    k: because the conflict population is fixed (see below) and each unit of planning
    work is seeded from its own coordinates rather than from the shared global stream,
    dropping/re-ordering conflicts can no longer reassign samples to later conflicts.
    Utility differences across k then reflect the corrupted world-model mask alone.
    """
    key = f"{base_seed}|{global_step}|{conflict_index}|{planner_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def main(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    run_dir=None,
    graph_run=None,
    graph_cfg=None,
    graph_override="causal",
    corrupt_edges=None,
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
    sampler = None
    run_manifest = None
    if cfg.model_kind == "cdl" and cfg.model.dynamics_mode == "structure_conditioned":
        from cdd_oran.models import (
            make_structure_sampler,
            verify_checkpoint_manifest,
            verify_run_artifacts,
        )

        # Self-contained run (review v2 MAJOR): read + hash-verify the STAGED artifacts and
        # build from those copies, never the external source paths. Fail loudly if missing/changed.
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        run_manifest, resolved = verify_run_artifacts(
            run_dir, env=env, environment=cfg.environment
        )
        # Inspect the checkpoint binding before constructing or sampling the posterior.
        verify_checkpoint_manifest(checkpoint_path, run_manifest)
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                posterior_artifact=resolved["posterior"],
                enumeration_graph=resolved["enumeration_graph"],
            ),
        )
        sampler = make_structure_sampler(cfg, seed=cfg.seed, env=env)
    model = get_model(cfg, env, sampler=sampler)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_model(checkpoint_path)

    if run_manifest is not None:
        # Keep a post-load exact check as a defense-in-depth assertion. The pre-construction
        # check above is the safety gate that prevents a mismatched sampler from being used.
        if getattr(model, "artifact_manifest", None) != run_manifest:
            raise ValueError(
                "checkpoint artifact manifest does not exactly match the staged run manifest"
            )

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

    # Opt-in corruption (Phase 0 mechanism test). Scope: this is an INFERENCE-TIME
    # structural omission in the CURRENT CDL. The predictor was trained on the FULL
    # feature set with random one-source dropout, and the learned graph is applied
    # only at prediction (cdl.py train_step/predict_next_state); removing an edge here
    # blinds the world model's PREDICTION mask to that parent. It is NOT evidence about
    # a predictor structurally retrained never to see the parent -- that
    # "retrained-without-edge" variant is an optional future check, not built here.
    #
    # Separation of the graph's two uses (the paper's own design principle): conflict
    # ENUMERATION stays on the BASE (pre-corruption) graph, captured here and held
    # FIXED for every corruption level k, so the planner is always asked to mitigate
    # the SAME conflicts. Only predict_next_state sees the corrupted mask. When
    # corruption is off, the plain --graph-override path is unchanged.
    enum_graph = None  # numpy [state_dim, state_dim] used for conflict enumeration
    if corrupt_edges:
        base_graph = cdl.get_binary_graph().clone()
        enum_graph = base_graph[:, :-1].cpu().detach().numpy()
        corrupted, removed = remove_edges(base_graph, corrupt_edges, env)
        cdl.get_binary_graph = lambda threshold=None, g=corrupted: g
        logger.info(
            "Graph corruption active (world-model mask only; conflicts fixed on base "
            "graph): removed %d edge(s): %s",
            len(removed),
            ", ".join(spec for spec, _, _ in removed),
        )

    # Phase 2 residual diagnostics on a FIXED probe bank (structure_conditioned only),
    # so runs are comparable and the report never depends on planner-induced states.
    residual_report = None
    if getattr(model, "dynamics_mode", "hard_mask") == "structure_conditioned":
        # Compute the probe under SAVED/RESTORED RNG state (review MAJOR #7): the probe
        # reseeds and consumes env randomness, so snapshot python/numpy/torch RNG first and
        # restore it afterward -- the real evaluation trajectory (env resets under CRN) is
        # then byte-identical to a run without the diagnostic.
        # preserve_probe_rng also snapshots the posterior sampler's PRIVATE numpy Generator
        # (review v2 BLOCKER A): residual_diagnostics draws a structure and would otherwise
        # advance that generator, shifting the planners' structure sequence.
        with preserve_probe_rng(model):
            seed_everything(cfg.seed, cfg.deterministic)
            probe = collect_transitions(get_env(cfg), cfg.model.batch_size, act_dim)
            s_probe = probe[:, 0].to(cfg.device)
            a_probe = torch.zeros(s_probe.shape[0], act_dim, device=cfg.device)
            diag = model.residual_diagnostics(s_probe, a_probe)
            residual_report = {
                "fraction_per_kpi": [float(x) for x in diag["fraction_per_kpi"]],
                "fraction_aggregate": float(diag["fraction_aggregate"]),
                "fraction_max_member": float(diag["fraction_max_member"]),
                "abs_mean_residual": [float(x) for x in diag["abs_mean_residual"]],
                "signed_mean_residual": [float(x) for x in diag["signed_mean_residual"]],
                "alert": bool(diag["alert"]),
                "alert_fraction": diag["alert_fraction"],
            }
        if diag["alert"]:
            logger.warning(
                "Residual contribution fraction %.3f exceeds alert threshold %.3f "
                "(possible MLP collapse)",
                residual_report["fraction_aggregate"],
                diag["alert_fraction"],
            )

    algorithms = get_planners(cfg, model, env)
    utility_fns = env.get_utility_fns()

    KPI_THRESHOLDS, MEAN_STD_KPIS = env.get_thresholds_stds()
    algorithm_names = [algo.name for algo in algorithms]
    utilities_data = {
        # v2 adds per-panel `conflict_xapp_ids` and per-planner `planner_satisfied`
        # (0/1 per conflicting xApp) so satisfaction is readable with no env access.
        # All v1 fields are preserved; old readers ignore the new keys.
        "version": 2,
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

        # Fixed conflict population: enumerate from the BASE graph when corrupting,
        # so no conflict is ever dropped and k = all-edges is still a valid row.
        if enum_graph is not None:
            causal_graph = enum_graph
        else:
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

        for conflict_index, edge in enumerate(edges):
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
                # v2: xApp ids aligned to each planner_utilities/planner_satisfied list.
                "conflict_xapp_ids": [xid for xid in conflict_xapp_ids if xid < len(utility_fns)],
                "planner_satisfied": {},
            }

            num_xapps = len(xapps_in_conflict)
            # Under review: this currently normalizes every conflict weight back to one.
            weights = np.ones(num_xapps)
            for idx, xapp in enumerate(xapps_in_conflict):
                if xapp == env.xapps[primary_xapp_id]:
                    weights[idx] *= 1.0
            weights = weights / weights.sum() * num_xapps

            for algo in algorithms:
                # Common random numbers: reseed the global torch/numpy RNG from this
                # unit of work's coordinates so the SAME conflict draws the SAME
                # samples at every k (planners use the global stream, see cem/mppi/mcts).
                planner_seed = _planner_seed(cfg.seed, global_step, conflict_index, algo.name)
                torch.manual_seed(planner_seed)
                np.random.seed(planner_seed)
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
                satisfied_values = []
                for xid in conflict_xapp_ids:
                    if xid >= len(utility_fns):
                        continue
                    u = compute_utility(utility_fns[xid], raw_params, param_id, raw_val)
                    step_utilities[algo.name].append(u)
                    planner_values.append(u)
                    # Same satisfied rule as cost.weighted_distance, persisted so
                    # satisfaction analysis needs no env: dir 0 satisfied iff
                    # u >= norm_threshold, dir 1 iff u <= norm_threshold.
                    xapp = env.xapps[xid]
                    norm_threshold = (xapp.threshold - xapp.mean) / xapp.std
                    satisfied = (
                        u >= norm_threshold if xapp.direction == 0 else u <= norm_threshold
                    )
                    satisfied_values.append(int(satisfied))
                panel["planner_utilities"][algo.name] = planner_values
                panel["planner_satisfied"][algo.name] = satisfied_values

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
    if residual_report is not None:
        metrics["evaluation"]["residual"] = residual_report
        utilities_data["residual"] = residual_report
    write_metrics(run_dir, metrics)
    (run_dir / "utilities.json").write_text(json.dumps(utilities_data, indent=2))
    logger.info("Evaluation metrics saved to %s", run_dir / "metrics.json")
    return 0
