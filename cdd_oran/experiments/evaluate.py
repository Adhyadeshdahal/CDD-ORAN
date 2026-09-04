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

import copy
import hashlib
import json
import logging
import random
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cdd_oran.analysis.counterfactual_metrics import (
    autoregressive_rollout_rmse,
    counterfactual_action_response_error,
    decision_regret,
    one_step_mse,
)
from cdd_oran.analysis.graph_baselines import (
    build_override_graph,
    collect_transitions,
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
        if sampler is not None and sampler_state is not None:
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


def _counterfactual_seed(base_seed, global_step, conflict_index, planner_name, horizon, role):
    """Stable seed namespace for opt-in P5 collection; never shares planner seeds."""
    key = (
        f"{base_seed}|{global_step}|{conflict_index}|{planner_name}|"
        f"counterfactual|{horizon}|{role}"
    ).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _latent_kpis(env):
    return np.asarray([kpi.compute_utility_value() for kpi in env.kpis], dtype=np.float32)


def _latent_state_tensor(env, device):
    params = []
    for param in env.params:
        low, high = param.get_threshold()
        value = param.get_param()
        params.append((value - low) / (high - low) if high != low else 1.0)
    return torch.as_tensor(np.asarray(params + _latent_kpis(env)), dtype=torch.float32, device=device)


def _reference_action(env, param_id, raw_value):
    """Encode the nearest valid action to the current raw parameter value."""
    best = None
    max_index = int(env.action_space[param_id + 2])
    for bin_id in range(int(env.num_bins)):
        for index in range(max_index + 1):
            candidate = [int(param_id), bin_id, index]
            distance = abs(float(env.action_to_param(candidate)[1]) - float(raw_value))
            key = (distance, bin_id, index)
            if best is None or key < best[0]:
                best = (key, candidate)
    if best is None:
        raise ValueError(f"no valid action found for parameter {param_id}")
    return best[1]


def _prediction_mean(model, state, action, structures=None):
    states = state.reshape(1, -1) if state.ndim == 1 else state
    predictions = []
    for member, member_state in enumerate(states):
        action_t = torch.as_tensor(action, dtype=torch.float32, device=member_state.device).reshape(1, -1)
        if structures is None:
            dist = model.predict_next_state(member_state.reshape(1, -1), action_t)
        else:
            dist = model.predict_next_state(
                member_state.reshape(1, -1), action_t, structures=structures
            )
        mean = dist.mean if hasattr(dist, "mean") else dist
        mean = torch.as_tensor(mean, dtype=torch.float32)
        if mean.ndim == 2:
            member_mean = mean[0]
        elif mean.ndim == 3 and mean.shape[1] == 1:
            member_mean = mean[min(member, mean.shape[0] - 1), 0]
        else:
            raise ValueError(
                "model prediction must be (1, k) or (m, 1, k); "
                f"got {tuple(mean.shape)}"
            )
        predictions.append(member_mean.detach().cpu().numpy())
    return np.asarray(predictions, dtype=float)


def collect_counterfactual_trajectory(
    model,
    planner,
    env,
    state_t,
    first_action,
    param_id,
    xapps_under_conflict,
    weights_per_xapps,
    scaling_term,
    raw_params,
    utility_fns,
    global_step,
    conflict_index,
    base_seed,
    *,
    planner_name=None,
    horizon=3,
    gamma=1.0,
):
    """Collect one opt-in, isolated P5 trajectory and score it per member.

    # ponytail: the collector intentionally uses latent targets and a deep-copied env;
    # this keeps the normal noisy evaluator state and its neutral-action transition
    # untouched while making the post-hoc target reproducible.
    """
    if horizon <= 0:
        raise ValueError("counterfactual horizon must be positive")
    planner_name = planner_name or getattr(planner, "name", planner.__class__.__name__)
    true_env = copy.deepcopy(env)
    device = state_t.device
    observed_state = state_t.detach().clone()

    structures = None
    if hasattr(model, "_sample_structures"):
        count = int(getattr(model, "predict_members", 1))
        structures = model._sample_structures(count, device)
        member_structures = structures.detach().cpu().numpy()
    else:
        member_structures = []
        count = 1
    pred_states = observed_state.reshape(1, -1).repeat(count, 1)

    action_rows = []
    reference_rows = []
    true_action_kpis = []
    true_reference_kpis = []
    predicted_action_kpis = []
    predicted_reference_kpis = []
    one_step_predictions = []
    oracle_actions = []
    oracle_costs = []
    planner_costs = []
    regrets = []
    utility_regrets = []
    current_true_state = observed_state
    action = list(first_action)

    # Utility functions are full-environment indexed in evaluate.py; the pure helper
    # consumes only the xApps participating in this planner's objective.
    env_xapp_indices = [env.xapps.index(xapp) for xapp in xapps_under_conflict]
    aligned_utility_fns = [utility_fns[index] for index in env_xapp_indices]

    for step in range(horizon):
        if step > 0:
            p5_seed = _counterfactual_seed(
                base_seed, global_step, conflict_index, planner_name, step, "action"
            )
            torch.manual_seed(p5_seed)
            np.random.seed(p5_seed)
            action = planner.act(
                current_state=current_true_state.clone().detach(),
                conflict_param_index=param_id,
                xapps_under_conflict=xapps_under_conflict,
                weights_per_xapps=list(weights_per_xapps),
                scaling_term=scaling_term,
            )
        action = [int(value) for value in action]
        raw_current = float(true_env.params[param_id].get_param())
        raw_params_for_regret = [param.get_param() for param in true_env.params]
        reference = _reference_action(true_env, param_id, raw_current)

        reference_env = copy.deepcopy(true_env)
        reference_env.step(reference)
        reference_kpi = _latent_kpis(reference_env)
        true_env.step(action)
        action_kpi = _latent_kpis(true_env)

        pred_action = _prediction_mean(model, pred_states, action, structures)
        pred_reference = _prediction_mean(model, pred_states, reference, structures)
        one_step = _prediction_mean(model, current_true_state, action, structures)

        action_rows.append(action)
        reference_rows.append(reference)
        true_action_kpis.append(action_kpi)
        true_reference_kpis.append(reference_kpi)
        predicted_action_kpis.append(pred_action)
        predicted_reference_kpis.append(pred_reference)
        one_step_predictions.append(one_step)

        current_regret = decision_regret(
            raw_params_for_regret,
            param_id,
            np.linspace(*env.params[param_id].get_threshold(), num=101),
            aligned_utility_fns,
            xapps_under_conflict,
            weights_per_xapps,
            planner_value=float(env.action_to_param(action)[1]),
            scaling_term=scaling_term,
        )
        oracle_actions.append(current_regret["oracle_action"])
        oracle_costs.append(current_regret["oracle_cost"])
        planner_costs.append(current_regret["planner_cost"])
        regrets.append(current_regret["decision_regret"])
        utility_regrets.append(current_regret["utility_regrets"])

        # Feed each model member its own prediction. Never average transitions.
        next_pred_states = pred_states.clone()
        next_pred_states[:, env.num_params :] = torch.as_tensor(
            pred_action, dtype=torch.float32, device=device
        )
        pred_states = next_pred_states
        current_true_state = _latent_state_tensor(true_env, device)

    arre = counterfactual_action_response_error(
        np.asarray(predicted_action_kpis).transpose(1, 0, 2),
        np.asarray(predicted_reference_kpis).transpose(1, 0, 2),
        np.asarray(true_action_kpis),
        np.asarray(true_reference_kpis),
        gamma=gamma,
    )
    rmse = autoregressive_rollout_rmse(
        np.asarray(predicted_action_kpis).transpose(1, 0, 2),
        np.asarray(true_action_kpis),
        gamma=gamma,
    )
    mse = one_step_mse(
        np.asarray(one_step_predictions).transpose(1, 0, 2),
        np.asarray(true_action_kpis),
    )
    return {
        "initial_state_observed": observed_state.detach().cpu().numpy(),
        "initial_params_raw": np.asarray(raw_params, dtype=float),
        "actions": np.asarray(action_rows, dtype=int),
        "reference_actions": np.asarray(reference_rows, dtype=int),
        "true_kpis": np.asarray(true_action_kpis),
        "true_reference_kpis": np.asarray(true_reference_kpis),
        "predicted_kpis": np.asarray(predicted_action_kpis).transpose(1, 0, 2),
        "predicted_reference_kpis": np.asarray(predicted_reference_kpis).transpose(1, 0, 2),
        "member_structures": member_structures,
        "one_step_mse_per_member": mse["per_member"],
        "cf_arre_per_member": arre["per_member"],
        "cf_rmse_per_member": rmse["per_member"],
        "one_step_mse": mse["mean"],
        "cf_arre": arre["mean"],
        "cf_arre_max_member": arre["max_member"],
        "cf_rmse": rmse["mean"],
        "cf_rmse_max_member": rmse["max_member"],
        "oracle_action": oracle_actions,
        "oracle_cost": oracle_costs,
        "planner_cost": planner_costs,
        "decision_regret": float(np.mean(regrets)),
        "decision_regret_per_horizon": np.asarray(regrets),
        "utility_regrets": np.asarray(utility_regrets),
        "oracle_grid_points": 101,
        "horizon": int(horizon),
        "gamma": float(gamma),
    }


def main(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    run_dir=None,
    graph_run=None,
    graph_cfg=None,
    graph_override="causal",
    counterfactual_horizon=0,
    counterfactual_gamma=1.0,
    enum_source="self",
    enum_graph=None,
):
    if run_dir is None:
        raise ValueError("An experiment run directory is required")
    if enum_source != "self" and graph_override != "causal":
        raise ValueError(
            "--enum-source and --graph-override both replace the enumeration graph; "
            "use at most one"
        )

    training_seed = cfg.seed
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
    # Oracle structure is rebuilt from the env's true adjacency inside get_model; it staged no
    # artifacts at train time, so there is nothing to verify or load here.
    if (
        cfg.model_kind == "cdl"
        and cfg.model.structure_source != "oracle"
        and cfg.model.posterior_artifact
    ):
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
        logger.info(
            "Graph override active: %s (%d edges) -- overrides the conflict-enumeration graph",
            graph_override,
            int(override.sum()),
        )

    # Matched-decision-set gate: install a canonical enumeration graph used identically for
    # every structure so both are scored on the SAME conflict set. Only conflict enumeration
    # reads get_binary_graph; the planner's prediction sampler is untouched.
    if enum_source != "self":
        from cdd_oran.models import (
            build_oracle_enumeration_graph,
            load_enumeration_graph,
        )

        if enum_source == "oracle":
            enum = build_oracle_enumeration_graph(env)
        else:
            if not enum_graph:
                raise ValueError("--enum-source=path requires --enum-graph")
            enum, _ = load_enumeration_graph(enum_graph)
        enum = enum.to(cdl.get_binary_graph().device)
        cdl.get_binary_graph = lambda threshold=None, g=enum: g
        logger.info(
            "Enumeration graph overridden: source=%s (%d state edges) -- matched decision "
            "set; prediction sampler untouched",
            enum_source,
            int(enum[:, :-1].sum()),
        )

    # Residual diagnostics on a FIXED probe bank (CDL world model only), so runs are
    # comparable and the report never depends on planner-induced states.
    residual_report = None
    if hasattr(model, "residual_diagnostics"):
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
        # Provenance for the matched-decision-set gate: readers/gate_2x2 can assert both
        # cells enumerated on the same source before comparing utilities.
        "enum_source": enum_source,
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
    counterfactual_records = []

    for global_step in range(cfg.num_steps):
        state_dict = env.get_state()
        state_t = state_to_tensor(state_dict).to(cfg.device)
        raw_params = denormalize_params(state_t[: env.num_params].cpu().numpy(), env)

        # Conflicts are enumerated from the (frozen) enumeration graph, or a --graph-override
        # replacement when one is active.
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

                if counterfactual_horizon:
                    with preserve_probe_rng(model):
                        counterfactual = collect_counterfactual_trajectory(
                            model=model,
                            planner=algo,
                            env=env,
                            state_t=state_t,
                            first_action=action,
                            param_id=param_id,
                            xapps_under_conflict=xapps_in_conflict,
                            weights_per_xapps=weights.tolist(),
                            scaling_term=10,
                            raw_params=raw_params,
                            utility_fns=utility_fns,
                            global_step=global_step,
                            conflict_index=conflict_index,
                            base_seed=cfg.seed,
                            planner_name=algo.name,
                            horizon=int(counterfactual_horizon),
                            gamma=float(counterfactual_gamma),
                        )
                    counterfactual.update(
                        {
                            "environment": cfg.environment,
                            "training_seed": int(training_seed),
                            "mitigation_seed": int(cfg.mitigation_seed),
                            "planner": algo.name,
                            "global_step": int(global_step),
                            "conflict_index": int(conflict_index),
                            "param_id": int(param_id),
                            "metric_version": "p5-v1",
                            "targets": "latent_noiseless_kpi_utility",
                            "pairing_key": [
                                cfg.environment,
                                int(cfg.mitigation_seed),
                                int(global_step),
                                int(conflict_index),
                            ],
                        }
                    )
                    counterfactual_records.append(counterfactual)

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
    evaluation: dict[str, Any] = {
        "planner_mean_utilities": {
            name: float(np.mean(values)) if values else None
            for name, values in all_utilities.items()
        }
    }
    metrics["evaluation"] = evaluation
    if residual_report is not None:
        evaluation["residual"] = residual_report
        utilities_data["residual"] = residual_report
    if counterfactual_horizon:
        evaluation["counterfactual"] = {
            "enabled": True,
            "horizon": int(counterfactual_horizon),
            "gamma": float(counterfactual_gamma),
            "n_records": len(counterfactual_records),
            "metric_version": "p5-v1",
            "targets": "latent_noiseless_kpi_utility",
        }
        counterfactual_payload = {
            "version": 1,
            "metadata": {
                "environment": cfg.environment,
                "training_seed": int(training_seed),
                "mitigation_seed": int(cfg.mitigation_seed),
                "horizon": int(counterfactual_horizon),
                "gamma": float(counterfactual_gamma),
                "state_bank_id": None,
                "metric_version": "p5-v1",
                "targets": "latent_noiseless_kpi_utility",
            },
            "records": counterfactual_records,
        }
        (run_dir / "counterfactuals.json").write_text(
            json.dumps(counterfactual_payload, indent=2, default=_json_default), encoding="utf-8"
        )
    write_metrics(run_dir, metrics)
    (run_dir / "utilities.json").write_text(json.dumps(utilities_data, indent=2))
    logger.info("Evaluation metrics saved to %s", run_dir / "metrics.json")
    return 0


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")
