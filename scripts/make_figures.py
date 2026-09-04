"""Regenerate the paper figures from committed run directories (read-only, no training).

Audit item 4: the named figures (`combined_envs.png`, `cmi_<Env>.png`, `mlp_<Env>.png`)
had no generator. This script rebuilds them from run dirs so a reviewer can reproduce
them. It reuses the matplotlib style and `algo_style` from `cdd_oran/viz/panels.py`
(importing that module applies its rcParams) WITHOUT modifying any viz module.

Figures:
  1. Per-step planner-utility line figure from a run's utilities.json:
     `cmi_<Environment>.png` (CDL) / `mlp_<Environment>.png` (MLP). x = env step,
     y = mean planner utility per step (averaged across that step's panels), one line
     per planner.
  2. `combined_envs.png`: causal-recovery recall and F1 vs training step for Env I and
     Env II, read from each run's TensorBoard scalars graph_eval/recall & graph_eval/f1.
     A panel whose run lacks those scalars is skipped with a message, not a crash.

Usage:
  uv run python scripts/make_figures.py --run <dir> --kind cdl
  uv run python scripts/make_figures.py --curve-runs EnvI=<dir>,EnvII=<dir>
  uv run python scripts/make_figures.py --run <dir> --kind mlp --out figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing panels applies its rcParams (shared style) and gives us the planner styles.
from cdd_oran.viz.panels import algo_style  # noqa: E402

RECALL_COLOR = "#1565C0"
F1_COLOR = "#D32F2F"


def make_utility_figure(run_dir, kind, out_dir):
    """Per-step planner-utility line figure from utilities.json. Returns the path, or
    None (with a message) if the run has no utilities.json (e.g. a train-only run)."""
    utilities_path = Path(run_dir) / "utilities.json"
    if not utilities_path.exists():
        print(
            f"[skip] {utilities_path} not found. This looks like a train-only run; run "
            f"`cdd_oran.cli evaluate --run {run_dir}` first to produce utilities.json, "
            f"then re-run for the {('cmi' if kind == 'cdl' else 'mlp')} utility figure."
        )
        return None

    data = json.loads(utilities_path.read_text())
    steps = data.get("steps", [])
    planners = list(data.get("algorithm_names", []))
    environment = data.get("environment", "unknown")

    step_x = []
    series = {planner: [] for planner in planners}
    for step in steps:
        step_x.append(step.get("step"))
        pooled = {planner: [] for planner in planners}
        for panel in step.get("panels", []):
            for planner, values in panel.get("planner_utilities", {}).items():
                pooled.setdefault(planner, []).extend(values)
        for planner in planners:
            vals = pooled.get(planner) or []
            series[planner].append(float(np.mean(vals)) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    for planner in planners:
        style = algo_style(planner)
        ax.plot(
            step_x, series[planner],
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            markersize=5, linewidth=1.6, label=style["label"],
        )
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Mean planner utility")
    ax.set_title(f"Per-step planner utility: {environment} / {data.get('model_kind', kind)}")
    if planners:
        ax.legend(frameon=False, fontsize=8)

    prefix = "cmi" if kind == "cdl" else "mlp"
    out_path = Path(out_dir) / f"{prefix}_{environment}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _load_scalar(tb_dir, tag):
    """Return ([steps], [values]) for a TensorBoard scalar tag, or None if absent."""
    tb_dir = Path(tb_dir)
    if not tb_dir.exists():
        return None
    accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def make_curve_figure(curve_runs, out_dir):
    """combined_envs.png: recall & F1 vs training step, one panel per (env, run).

    A run whose tensorboard lacks graph_eval/recall & graph_eval/f1 is skipped with a
    message; the figure is written only if at least one panel had data."""
    labels = list(curve_runs)
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), squeeze=False)
    drawn = 0
    for index, label in enumerate(labels):
        ax = axes[0][index]
        tb_dir = Path(curve_runs[label]) / "tensorboard"
        recall = _load_scalar(tb_dir, "graph_eval/recall")
        f1 = _load_scalar(tb_dir, "graph_eval/f1")
        if recall is None and f1 is None:
            print(
                f"[skip] {label}: no graph_eval/recall or graph_eval/f1 scalars under "
                f"{tb_dir}; skipping this panel."
            )
            ax.set_visible(False)
            continue
        drawn += 1
        if recall is not None:
            ax.plot(recall[0], recall[1], color=RECALL_COLOR, marker="o", markersize=4,
                    linewidth=1.6, label="recall")
        if f1 is not None:
            ax.plot(f1[0], f1[1], color=F1_COLOR, marker="s", markersize=4,
                    linewidth=1.6, label="F1")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Causal recovery: {label}")
        ax.legend(frameon=False, fontsize=8)

    if drawn == 0:
        print("[skip] combined_envs.png: no panel had graph_eval scalars; not writing.")
        plt.close(fig)
        return None

    out_path = Path(out_dir) / "combined_envs.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _parse_curve_runs(spec):
    runs = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--curve-runs entry must be Label=dir, got {pair!r}")
        label, path = pair.split("=", 1)
        runs[label.strip()] = path.strip()
    if not runs:
        raise SystemExit("--curve-runs is empty")
    return runs


def build_parser():
    parser = argparse.ArgumentParser(description="Regenerate paper figures from run dirs")
    parser.add_argument("--run", default=None, help="Run dir for the per-step utility figure")
    parser.add_argument("--kind", choices=["cdl", "mlp"], help="Model kind of --run")
    parser.add_argument(
        "--curve-runs", default=None,
        help="Comma list Label=dir for the combined recovery-curve figure "
        "(e.g. EnvI=runs/.../cdl/x,EnvII=runs/.../cdl/y)",
    )
    parser.add_argument("--out", default="figures", help="Output directory (default: figures/)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.run and not args.curve_runs:
        raise SystemExit("nothing to do: pass --run (+ --kind) and/or --curve-runs")
    plt.switch_backend("Agg")

    if args.run:
        if not args.kind:
            raise SystemExit("--run requires --kind {cdl,mlp}")
        make_utility_figure(args.run, args.kind, args.out)
    if args.curve_runs:
        make_curve_figure(_parse_curve_runs(args.curve_runs), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
