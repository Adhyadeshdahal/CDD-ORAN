"""One-command recovery gate: does a recall floor persist at FULL budget, and is
it NCP->KPI (ENCO could help) or KPI->KPI (ENCO cannot)?

For ONE env (default EnvironmentI) and ONE seed:
  1. train CDL at FULL budget (Env I 20000 steps; --steps overrides),
  2. with NO retrain, read the frozen CMI matrix and score it at three threshold
     regimes -- config threshold, sweep-optimal, and each auto-threshold method --
     plus the bootstrap edge-stability graph, each SPLIT into NCP->KPI vs KPI->KPI
     recall via cdd_oran.analysis.recovery_metrics,
  3. print a verdict block with the missed-edge list and a one-line read on whether
     a floor persists and of which type.

Reuses the existing CLI (cdd_oran.cli) and analysis modules (threshold_sweep,
auto_threshold, edge_stability); it does not reimplement training or analysis.

Usage:
    uv run python scripts/recovery_gate.py --dry-run
    uv run python scripts/recovery_gate.py                 # trains full budget, then scores
    uv run python scripts/recovery_gate.py --run RUN_DIR   # score an existing run, no train
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_journal_suite import (  # noqa: E402
    CONFIG_CDL,
    FULL_STEPS,
    _cli_command,
    _train_args,
    discover_run,
)

from cdd_oran.analysis import auto_threshold, edge_stability, threshold_sweep  # noqa: E402
from cdd_oran.analysis.recovery_metrics import recovery_by_edge_type  # noqa: E402


def _train_command(env, seed, steps, device):
    overrides = [f"train.total_steps={steps}"]
    return _cli_command(_train_args(CONFIG_CDL[env], seed, overrides, device))


def _load_graph(run_dir):
    """Read the frozen CMI matrix + ground truth (no retrain, no forward passes)."""
    from cdd_oran.config import load_config
    from cdd_oran.envs import get_env
    from cdd_oran.models import get_model

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    env = get_env(cfg)
    model = get_model(cfg, env)
    model.load_model(run_dir / "checkpoint.pt")
    cmi = model.get_causal_graph().cpu().detach().numpy()
    return cfg, env, model, cmi


def _score(cmi, gt, num_params, threshold, node_names):
    pred = auto_threshold.binary_graph(cmi, threshold)  # (fd, fd), diagonal zeroed
    return recovery_by_edge_type(pred, gt, num_params, node_names)


def _fmt_block(name, threshold, block):
    o, n, k = block["overall"], block["ncp_kpi"], block["kpi_kpi"]
    return (
        f"  {name:<16} thr={threshold:<8.4f} "
        f"overall R={o['recall']:.3f} (tp{o['tp']}/fn{o['fn']}, P={o['precision']:.3f}) | "
        f"NCP->KPI R={n['recall']:.3f} (tp{n['tp']}/fn{n['fn']}) | "
        f"KPI->KPI R={k['recall']:.3f} (tp{k['tp']}/fn{k['fn']})"
    )


def score_run(run_dir, args):
    cfg, env, model, cmi = _load_graph(run_dir)
    gt = env.true_adj_matrix
    num_params = env.num_params
    node_names = getattr(model, "node_names", None)

    start, stop, num = args.grid.split(",")
    grid = np.linspace(float(start), float(stop), int(num))
    sweep_result = threshold_sweep.sweep(run_dir, grid)
    sweep_thr = sweep_result["best"]["threshold"]
    auto_thrs = auto_threshold.auto_thresholds(cmi)

    regimes = [("config", cfg.model.cmi_threshold), ("sweep-optimal", sweep_thr)]
    regimes += [(f"auto:{name}", thr) for name, thr in auto_thrs.items()]

    print(f"\n=== RECOVERY GATE: {cfg.environment} seed={cfg.seed} ===")
    print(f"run: {run_dir}")
    print(f"ground-truth edges: {int(gt.sum())} "
          f"(NCP->KPI cols <{num_params}, KPI->KPI cols >={num_params})")
    print("\nthreshold regimes (recall split by edge type):")

    scored = {}
    for name, thr in regimes:
        block = _score(cmi, gt, num_params, thr, node_names)
        scored[name] = block
        print(_fmt_block(name, thr, block))

    stab = edge_stability.edge_stability(
        run_dir, B=args.B, n_transitions=args.n_transitions, pi=args.pi,
        seed=args.seed, device=args.device,
    )
    freq = np.asarray(stab["frequency_matrix"])
    stab_pred = (freq >= args.pi).astype(int)
    np.fill_diagonal(stab_pred, 0)
    stab_block = recovery_by_edge_type(stab_pred, gt, num_params, node_names)
    scored[f"stability@{args.pi}"] = stab_block
    print(_fmt_block(f"stability@{args.pi}", args.pi, stab_block))

    # Best achievable recall per edge type across all regimes -> the "floor".
    best_ncp = max(b["ncp_kpi"]["recall"] for b in scored.values())
    best_kpi = max(b["kpi_kpi"]["recall"] for b in scored.values())
    best_overall_regime = max(scored.items(), key=lambda kv: kv[1]["overall"]["recall"])

    print("\n--- missed true edges (best-overall-recall regime "
          f"'{best_overall_regime[0]}') ---")
    missed = best_overall_regime[1]["missed"]
    if not missed:
        print("  none -- full recall reached")
    for m in missed:
        label = ""
        if "child_name" in m:
            label = f"  {m['parent_name']} -> {m['child_name']}"
        print(f"  {m['type']:<8} parent col {m['parent']} -> child row {m['child']}{label}")

    print("\n--- VERDICT ---")
    print(f"  best NCP->KPI recall (any regime): {best_ncp:.3f}")
    print(f"  best KPI->KPI recall (any regime): {best_kpi:.3f}")
    if best_ncp >= 1.0 - 1e-9 and best_kpi >= 1.0 - 1e-9:
        print("  NO floor: full budget recovers every edge. ENCO not needed.")
    elif best_ncp < 1.0 - 1e-9:
        print("  NCP->KPI floor PERSISTS at full budget -- interventional discovery "
              "(ENCO) could lift it. BUILD SIMPLIFIED is justified.")
        if best_kpi < 1.0 - 1e-9:
            print("  (KPI->KPI edges also missed, but ENCO cannot help those: KPIs "
                  "are never intervened.)")
    else:
        print("  Only KPI->KPI edges missed. ENCO CANNOT help (KPIs never "
              "intervened). Not worth a second model for this floor.")
    return scored


def _print_plan(args, steps):
    print(f"Recovery gate plan [{args.env} seed={args.seed}]")
    if args.run:
        print(f"  score existing run: {args.run} (NO training)")
    else:
        print(f"  1. train CDL FULL budget: {steps} steps")
    print("  2. score frozen CMI (no retrain) at: config / sweep-optimal / "
          "auto-threshold(largest_gap,otsu,kmeans2) / edge-stability")
    print("  3. split recall NCP->KPI vs KPI->KPI + missed-edge list + verdict")


def _dry_run(args, steps):
    print("\nDry run: no commands are executed.")
    if not args.run:
        print(" ".join(_train_command(args.env, args.seed, steps, args.device)))
    run_ref = args.run or f"<runs/{args.env}/cdl seed={args.seed} (from step 1)>"
    print(" ".join(
        ["uv", "run", "python", "-m", "cdd_oran.analysis.threshold_sweep",
         "--run", run_ref, "--grid", args.grid]
    ))
    print(" ".join(
        ["uv", "run", "python", "-m", "cdd_oran.analysis.auto_threshold", "--run", run_ref]
    ))
    print(" ".join(
        ["uv", "run", "python", "-m", "cdd_oran.analysis.edge_stability",
         "--run", run_ref, "--B", str(args.B), "--n-transitions", str(args.n_transitions),
         "--pi", str(args.pi)]
    ))
    print("in-process: cdd_oran.analysis.recovery_metrics.recovery_by_edge_type "
          "on each thresholded graph (NCP->KPI vs KPI->KPI split + missed edges)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="EnvironmentI", choices=list(CONFIG_CDL))
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--steps", type=int, default=None, help="Full-budget train steps (default: env FULL_STEPS)")
    p.add_argument("--run", default=None, help="Score this existing run; skip training")
    p.add_argument("--dry-run", action="store_true", help="Print the plan + commands, run nothing")
    p.add_argument("--grid", default="0.0,0.40,41", help="threshold_sweep grid start,stop,num")
    p.add_argument("--B", type=int, default=50, help="edge_stability bootstrap resamples")
    p.add_argument("--n-transitions", type=int, default=2048, help="edge_stability pool size")
    p.add_argument("--pi", type=float, default=0.5, help="edge_stability selection-frequency cut")
    p.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    steps = args.steps if args.steps is not None else FULL_STEPS[args.env]

    _print_plan(args, steps)

    if args.dry_run:
        _dry_run(args, steps)
        return 0

    run_dir = args.run
    if run_dir is None:
        train_argv = _train_command(args.env, args.seed, steps, args.device)
        print("\n[train] " + " ".join(train_argv))
        import subprocess

        start = time.time()
        result = subprocess.run(train_argv, check=False)
        if result.returncode != 0:
            print(f"[FAIL] training exited {result.returncode}")
            return 1
        run_dir = discover_run(args.env, "cdl", args.seed, after=start)
        if run_dir is None:
            print("[FAIL] could not locate the trained run")
            return 1
        print(f"[train] run: {run_dir}")

    score_run(run_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
