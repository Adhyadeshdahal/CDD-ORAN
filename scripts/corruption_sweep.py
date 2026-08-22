"""Controlled graph-corruption sweep (Phase 0 mechanism test).

Given a trained CDL run whose recovered graph equals the oracle (F1=1.0) and an
ORDERED list of action-relevant NCP->KPI edges, remove the first ``k`` edges for
k = 0, 1, 2, ... and evaluate at FULL planner budget. Collects
``evaluation.planner_mean_utilities`` per k and writes a degradation curve.

Design (post-review v2):
  * Conflict ENUMERATION is fixed on the BASE (pre-corruption) graph inside
    ``evaluate`` -- the conflict population is IDENTICAL for every k, so a removed
    edge is never dropped from the evaluated set; only the world model's PREDICTION
    mask is blinded to it. k = all-edges is a valid row, not ``None``.
  * ``evaluate`` uses common random numbers (per-(step, conflict, planner) reseeding)
    so the SAME conflict draws the SAME samples at every k.

Scope of the intervention: this is an INFERENCE-TIME structural omission in the
CURRENT CDL. The predictor was trained on the full feature set with random
one-source dropout and applies the learned graph only at prediction, so this tests
whether the (now blind) model can still mitigate the SAME conflict -- it is NOT
evidence about a predictor structurally retrained never to see the parent. A
"retrained-without-edge" variant is a possible future check; it is not built here.

The banked run is NEVER mutated: evaluation runs in a disposable copy of the run
dir under ``.temp/`` and the driver asserts the base run dir is byte-identical
(file set + mtimes + sizes) after the sweep.

Outputs:
  - ``.temp/results/p0_corruption_envI.md``           rows = k, cols = planners + pooled
  - ``.temp/results/p0_corruption_envI.json``          utility vs k, per planner
  - ``.temp/results/p0_corruption_envI_edgedeltas.json`` per-edge world-model deltas

This driver RUNS the planners; on a GPU box that is the full-budget sweep. Use
``--dry-run`` to print the plan without executing anything. The edge-delta diagnostic
always runs on CPU (no planning).

Usage:
    uv run python scripts/corruption_sweep.py \
        --run runs/EnvironmentI/cdl/20260822-023837-14f229f1 \
        --edges "KPI1<-P2,KPI2<-P3,KPI3<-P4,KPI4<-P7" \
        --mcts 3000 --mppi 2000 --cem-iter 20
"""

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdd_oran.analysis.graph_baselines import remove_edges  # noqa: E402
from cdd_oran.config import load_config  # noqa: E402
from cdd_oran.conflicts import state_to_tensor  # noqa: E402
from cdd_oran.envs import get_env  # noqa: E402
from cdd_oran.experiments.evaluate import main as evaluate_main  # noqa: E402
from cdd_oran.models import get_model  # noqa: E402
from cdd_oran.utils.runs import read_metrics  # noqa: E402
from cdd_oran.utils.seeding import seed_everything  # noqa: E402


def _budget_cfg(cfg, args):
    """Apply per-planner budget overrides; unspecified planners keep the run config."""
    cem = cfg.planner.cem
    mppi = cfg.planner.mppi
    mcts = cfg.planner.mcts
    if args.cem_cand is not None:
        cem = replace(cem, n_candidate=args.cem_cand)
    if args.cem_iter is not None:
        cem = replace(cem, n_iter=args.cem_iter)
    if args.mppi is not None:
        mppi = replace(mppi, n_samples=args.mppi)
    if args.mcts is not None:
        mcts = replace(mcts, n_simulations=args.mcts)
    planner = replace(cfg.planner, cem=cem, mppi=mppi, mcts=mcts)
    cfg = replace(cfg, planner=planner)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    return cfg


def _pooled(utilities):
    """Pooled utility = unweighted mean over the per-planner mean utilities."""
    vals = [v for v in utilities.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def _snapshot(run_dir):
    """(rel_path -> (mtime_ns, size)) for every file under ``run_dir``.

    Used to prove the banked run is byte-identical before and after the sweep.
    """
    run_dir = Path(run_dir)
    return {
        str(p.relative_to(run_dir)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in run_dir.rglob("*")
        if p.is_file()
    }


# --------------------------------------------------------------------------- #
# Fix 5: per-edge world-model delta diagnostic (CPU, no planning).            #
# --------------------------------------------------------------------------- #
def _eval_states(cfg):
    """The FIXED evaluation states the sweep sees: seed with the mitigation seed,
    reset, and roll the env forward with a neutral action, exactly as ``evaluate``.
    """
    cfg = replace(cfg, seed=cfg.mitigation_seed, param_ranges=cfg.evaluation_param_ranges)
    seed_everything(cfg.seed, cfg.deterministic)
    env = get_env(cfg)
    act_dim = env.get_action_dim()
    env.reset()
    states = []
    neutral = torch.zeros(act_dim)
    n = min(cfg.num_steps, 32)  # a handful of fixed states is enough to detect a bite
    for _ in range(n):
        states.append(state_to_tensor(env.get_state()))
        env.step(neutral.numpy())
    return torch.stack(states), act_dim


def _predict_and_pool(cdl, s, a, graph):
    """Predicted next-state mean + max-pooled masked features under ``graph``.

    Mirrors ``CDL.predict_next_state`` (cdl.py) exactly, but also returns the pooled
    feature tensor that feeds the predictor head, so we can tell a correctly removed
    but numerically inactive edge (delta ~ 0 under max-pool) from a harness failure.
    """
    s = s.to(cdl.device).float()
    a = a.to(cdl.device).float()
    with torch.no_grad():
        cdl.models.eval()
        parameters = cdl._stacked_parameters()
        feats = cdl._batched_forward(
            parameters, s, a, features=None, features_in_dim=None, return_features=True
        )
        targets = torch.arange(cdl.kpi_start, cdl.state_dim, device=cdl.device)
        parameters = {name: value[targets] for name, value in parameters.items()}
        feats = feats[targets]
        graph_mask = graph[targets].clone()
        graph_mask[torch.arange(len(targets), device=cdl.device), targets] = True
        graph_mask[:, -1] = True
        masked_feats = feats.masked_fill(~graph_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
        pooled = masked_feats.max(dim=-2).values  # (num_targets, bs, feature_dim)
        mu, _ = cdl._batched_forward(parameters, features=masked_feats, features_in_dim=0)
    mu = mu.squeeze(-1).transpose(0, 1)  # (bs, num_targets)
    pooled = pooled.transpose(0, 1)  # (bs, num_targets, feature_dim)
    return mu, pooled


def edge_delta_diagnostic(run_dir, base_cfg, edges):
    """For each edge, the change in predicted next-state and pooled features on the
    FIXED evaluation states, with vs without that single edge (base graph as ref)."""
    cfg = replace(base_cfg, device="cpu")
    states, act_dim = _eval_states(cfg)

    env = get_env(replace(cfg, seed=cfg.mitigation_seed, param_ranges=cfg.evaluation_param_ranges))
    cdl = get_model(cfg, env)
    cdl.load_model(Path(run_dir) / "checkpoint.pt")

    base_graph = cdl.get_binary_graph().clone()
    s = states
    a = torch.zeros(states.shape[0], act_dim)  # neutral action probe

    mu_base, pooled_base = _predict_and_pool(cdl, s, a, base_graph)

    results = []
    for spec in edges:
        corrupted, removed = remove_edges(base_graph, [spec], env)
        _, child, parent = removed[0]
        present = bool(base_graph[child, parent])
        mu_c, pooled_c = _predict_and_pool(cdl, s, a, corrupted)
        d_mu = (mu_c - mu_base).abs()
        d_pool = (pooled_c - pooled_base).abs()
        child_tgt = child - cdl.kpi_start  # index into the KPI target axis
        results.append(
            {
                "edge": spec,
                "child_row": int(child),
                "parent_col": int(parent),
                "present_in_base_graph": present,
                "pred_mean_delta_max": float(d_mu.max()),
                "pred_mean_delta_mean": float(d_mu.mean()),
                "child_kpi_pred_delta_max": float(d_mu[:, child_tgt].max()),
                "pooled_feature_delta_max": float(d_pool.max()),
                "pooled_feature_delta_mean": float(d_pool.mean()),
                # A correctly removed edge that is numerically inactive under max-pool
                # shows ~0 here even though it IS absent from the graph.
                "bites": bool(d_mu.max() > 1e-6),
            }
        )
    return {
        "run": str(run_dir),
        "environment": base_cfg.environment,
        "n_eval_states": int(states.shape[0]),
        "action_probe": "neutral (zeros)",
        "note": (
            "Inference-time structural omission in the current CDL. delta ~ 0 means the "
            "edge was removed from the graph but is numerically inactive under the "
            "max-pool, NOT that the harness failed to remove it (see present_in_base_graph)."
        ),
        "edges": results,
    }


def _write_table(path, edges, planner_names, rows):
    header = ["k", "removed_edge", *planner_names, "pooled"]
    lines = ["# Env I controlled graph-corruption sweep", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        util = row["planner_mean_utilities"]
        cells = [str(row["k"]), row["removed_edge"] or "(none)"]
        for name in planner_names:
            value = util.get(name)
            cells.append(f"{value:.6f}" if value is not None else "n/a")
        pooled = row["pooled"]
        cells.append(f"{pooled:.6f}" if pooled is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def run_sweep(args):
    run_dir = Path(args.run)
    edges = [e.strip() for e in args.edges.split(",") if e.strip()]
    base_cfg = _budget_cfg(load_config(run_dir / "config.yaml"), args)

    if args.dry_run:
        print(f"[dry-run] run: {run_dir}")
        print(f"[dry-run] ordered edges ({len(edges)}): {edges}")
        print(
            "[dry-run] budget: "
            f"MCTS n_simulations={base_cfg.planner.mcts.n_simulations}, "
            f"MPPI n_samples={base_cfg.planner.mppi.n_samples}, "
            f"CEM n_candidate={base_cfg.planner.cem.n_candidate} "
            f"n_iter={base_cfg.planner.cem.n_iter}"
        )
        print(f"[dry-run] would evaluate k = 0..{len(edges)} on device={base_cfg.device}")
        print("[dry-run] evaluation runs in a temp copy; banked run is left untouched")
        print(f"[dry-run] outputs: {args.out_table}, {args.out_json}, {args.out_edgedeltas}")
        return 0

    # Fix 5: prove each removed edge actually bites (CPU, no planning) before the sweep.
    deltas = edge_delta_diagnostic(run_dir, base_cfg, edges)
    Path(args.out_edgedeltas).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_edgedeltas).write_text(json.dumps(deltas, indent=2) + "\n")
    print(f"[diag] edge deltas -> {args.out_edgedeltas}")
    for e in deltas["edges"]:
        print(
            f"[diag]   {e['edge']}: pred_delta_max={e['pred_mean_delta_max']:.3e} "
            f"bites={e['bites']}"
        )

    # Fix 3: never mutate the banked run. Evaluate in a disposable copy under .temp/,
    # read the metrics back, discard. Assert the base run dir is byte-identical after.
    base_snapshot = _snapshot(run_dir)
    scratch_root = Path(".temp") / "corruption_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"{datetime.now():%Y%m%d-%H%M%S}-", dir=scratch_root))
    for name in ("checkpoint.pt", "config.yaml"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, scratch / name)

    rows = []
    planner_names = None
    try:
        for k in range(len(edges) + 1):
            corrupt = edges[:k] or None
            removed_edge = edges[k - 1] if k > 0 else None
            print(f"[k={k}] removing {corrupt or '(none)'} ...", flush=True)
            rc = evaluate_main(base_cfg, run_dir=str(scratch), corrupt_edges=corrupt)
            if rc != 0:
                print(f"[FAIL] evaluate exited {rc} at k={k}")
                return 1
            utilities = read_metrics(scratch)["evaluation"]["planner_mean_utilities"]
            if planner_names is None:
                planner_names = list(utilities)
            rows.append(
                {
                    "k": k,
                    "removed_edge": removed_edge,
                    "removed_edges": edges[:k],
                    "planner_mean_utilities": utilities,
                    "pooled": _pooled(utilities),
                }
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    after_snapshot = _snapshot(run_dir)
    if after_snapshot != base_snapshot:
        changed = sorted(set(after_snapshot) ^ set(base_snapshot)) or [
            n for n in base_snapshot if after_snapshot.get(n) != base_snapshot[n]
        ]
        raise AssertionError(f"banked run dir was mutated during the sweep: {changed}")
    print(f"[ok] banked run {run_dir} unchanged ({len(base_snapshot)} files)")

    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    _write_table(args.out_table, edges, planner_names, rows)
    Path(args.out_json).write_text(
        json.dumps(
            {
                "run": str(run_dir),
                "environment": base_cfg.environment,
                "ordered_edges": edges,
                "budget": {
                    "mcts_n_simulations": base_cfg.planner.mcts.n_simulations,
                    "mppi_n_samples": base_cfg.planner.mppi.n_samples,
                    "cem_n_candidate": base_cfg.planner.cem.n_candidate,
                    "cem_n_iter": base_cfg.planner.cem.n_iter,
                },
                "planner_names": planner_names,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] table -> {args.out_table}")
    print(f"[done] curve -> {args.out_json}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Trained CDL run directory (base graph = oracle)")
    p.add_argument(
        "--edges",
        required=True,
        help="Ordered comma-separated edges to remove one at a time, e.g. 'KPI1<-P2,KPI2<-P3'",
    )
    p.add_argument("--mcts", type=int, default=None, help="MCTS n_simulations (default: run config)")
    p.add_argument("--mppi", type=int, default=None, help="MPPI n_samples (default: run config)")
    p.add_argument("--cem-cand", type=int, default=None, help="CEM n_candidate (default: run config)")
    p.add_argument("--cem-iter", type=int, default=None, help="CEM n_iter (default: run config)")
    p.add_argument("--device", default=None, help="Force device (cuda/cpu); default: run config")
    p.add_argument("--out-table", default=".temp/results/p0_corruption_envI.md")
    p.add_argument("--out-json", default=".temp/results/p0_corruption_envI.json")
    p.add_argument(
        "--out-edgedeltas", default=".temp/results/p0_corruption_envI_edgedeltas.json"
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan, execute nothing")
    return p


def main(argv=None):
    return run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
