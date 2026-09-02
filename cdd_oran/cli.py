import argparse
import logging
from dataclasses import replace
from pathlib import Path

from cdd_oran.config import load_config
from cdd_oran.experiments.sweep import run_aggregate, run_sweep
from cdd_oran.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _add_log_level(parser):
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")


def _load_run_config(run_dir, config_path):
    return load_config(config_path or Path(run_dir) / "config.yaml")


def build_parser():
    parser = argparse.ArgumentParser(description="CDD O-RAN experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a world model")
    train.add_argument("--config", default="configs/env_i_mlp.yaml", help="YAML experiment config")
    train.add_argument("--seed", type=int, help="Override the config seed")
    train.add_argument("--resume", action="store_true", help="Resume the checkpoint in --run")
    train.add_argument("--run", help="Existing run directory to resume")
    train.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config path",
    )
    _add_log_level(train)

    evaluate = commands.add_parser("evaluate", help="Evaluate planners using a trained run")
    evaluate.add_argument("--run", required=True, help="Training run containing checkpoint.pt")
    evaluate.add_argument("--config", help="Override config path; defaults to RUN/config.yaml")
    evaluate.add_argument(
        "--graph-run",
        help="Required for MLP runs: trained CDL run supplying the causal graph checkpoint",
    )
    evaluate.add_argument(
        "--graph-override",
        choices=("causal", "full", "correlation"),
        default="causal",
        help=(
            "Ablate the conflict-enumeration graph structure: causal (default), full (dense), "
            "correlation"
        ),
    )
    evaluate.add_argument(
        "--enum-source",
        choices=("self", "oracle", "path"),
        default="self",
        help=(
            "Enumeration graph for conflict detection: self (structure_source, default), "
            "oracle (env.true_adj_matrix -- canonical matched decision set), path (--enum-graph)"
        ),
    )
    evaluate.add_argument(
        "--enum-graph",
        help="Enumeration-graph JSON (fd, fd+1) when --enum-source=path",
    )
    _add_log_level(evaluate)

    discover = commands.add_parser(
        "discover",
        help="Discover causal structure -> posterior + frozen enumeration graph",
        description=(
            "Discover causal structure from an environment. --calibration-run is REQUIRED to "
            "produce a downstream-sampleable CALIBRATED posterior (<env>_posterior.json): its "
            "labels must come from a DISTINCT held-out environment, never the target's own true "
            "adjacency. Without --calibration-run the output is only an UNCALIBRATED raw "
            "diagnostic (<env>_raw_posterior.json) that robust structure sampling / staging "
            "reject; the target's true adjacency is not read on this path. True adjacency is "
            "allowed for post-hoc recovery scoring only (e.g. cdd_oran.analysis.auto_threshold)."
        ),
    )
    discover.add_argument("--config", required=True, help="YAML experiment config (model_kind=cdl)")
    discover.add_argument("--seed", type=int, help="Override the config seed")
    discover.add_argument(
        "--out", default="artifacts", help="Directory for <env>_posterior.json / <env>_enum.json"
    )
    discover.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config path",
    )
    discover.add_argument("--train-steps", type=int, default=4000, help="CMI-head gradient steps")
    discover.add_argument("--pool-size", type=int, default=4096, help="Transition pool size")
    discover.add_argument("--cmi-batches", type=int, default=64, help="Batches to converge mask_CMI")
    discover.add_argument("--B", type=int, default=50, help="Posterior bootstrap resamples")
    discover.add_argument("--n-transitions", type=int, default=2048, help="Bootstrap transition pool")
    discover.add_argument(
        "--calibration-run",
        help=(
            "DISTINCT held-out environment's run dir supplying calibration labels. Required for a "
            "calibrated, downstream-sampleable posterior; omit it for a raw uncalibrated "
            "diagnostic only. Must NOT be the target environment (same-target calibration leaks "
            "the target's true adjacency and is rejected)."
        ),
    )
    _add_log_level(discover)

    viz = commands.add_parser("viz", help="Visualize an experiment run")
    viz.add_argument("--run", required=True, help="Experiment run directory")
    viz.add_argument(
        "--figure",
        choices=("all", "causal-graph", "cmi-heatmap", "panels"),
        default="all",
    )
    viz.add_argument("--out", help="Output image path, or directory with --figure all")
    viz.add_argument("--show", action="store_true", help="Display the figure after saving")
    _add_log_level(viz)

    sweep = commands.add_parser("sweep", help="Train and evaluate one run per seed")
    sweep.add_argument("--config", required=True, help="YAML experiment config")
    sweep.add_argument("--seeds", required=True, help="Comma-separated training seeds")
    sweep.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config path",
    )
    sweep.add_argument("--tag", help="Sweep manifest name under sweeps/")
    sweep.add_argument(
        "--graph-sweep",
        help="Required for MLP: manifest containing one trained CDL run for each seed",
    )
    _add_log_level(sweep)

    aggregate_parser = commands.add_parser(
        "aggregate", help="Aggregate utilities across sweep seeds"
    )
    aggregate_parser.add_argument("--sweep", action="append", required=True, help="Sweep directory")
    aggregate_parser.add_argument("--out", help="Output directory (default: results/<sweep-tag>)")
    _add_log_level(aggregate_parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "train":
            if args.resume and not args.run:
                raise ValueError("--resume requires --run")
            cfg = load_config(args.config, args.set)
            if args.seed is not None:
                cfg = replace(cfg, seed=args.seed)
            from cdd_oran.experiments.train import main as train

            train(cfg, resume=args.resume, run_dir=Path(args.run) if args.run else None)
            return 0

        if args.command == "evaluate":
            cfg = _load_run_config(args.run, args.config)
            graph_cfg = _load_run_config(args.graph_run, None) if args.graph_run else None
            from cdd_oran.experiments.evaluate import main as evaluate

            return evaluate(
                cfg,
                run_dir=args.run,
                graph_run=args.graph_run,
                graph_cfg=graph_cfg,
                graph_override=args.graph_override,
                enum_source=args.enum_source,
                enum_graph=args.enum_graph,
            )

        if args.command == "discover":
            cfg = load_config(args.config, args.set)
            if args.seed is not None:
                cfg = replace(cfg, seed=args.seed)
            from cdd_oran.experiments.discover import main as discover_main

            result = discover_main(
                cfg,
                out_dir=args.out,
                train_steps=args.train_steps,
                pool_size=args.pool_size,
                cmi_batches=args.cmi_batches,
                B=args.B,
                n_transitions=args.n_transitions,
                calibration_run=args.calibration_run,
            )
            if result["calibrated"]:
                logger.info(
                    "Discovery complete: calibrated_posterior=%s enum=%s (%d state edges)",
                    result["calibrated_posterior"],
                    result["enumeration_graph"],
                    result["state_edges"],
                )
            else:
                logger.info(
                    "Discovery complete: raw_posterior=%s (UNCALIBRATED, not downstream-"
                    "sampleable; pass --calibration-run) enum=%s (%d state edges)",
                    result["raw_posterior"],
                    result["enumeration_graph"],
                    result["state_edges"],
                )
            return 0

        if args.command == "sweep":
            run_sweep(args.config, args.seeds, args.set, args.tag, args.graph_sweep)
            return 0

        if args.command == "aggregate":
            output_dir = run_aggregate(args.sweep, args.out)
            logger.info("Aggregate written to %s", output_dir)
            return 0

        cfg = _load_run_config(args.run, None)
        from cdd_oran.experiments.viz import main as visualize

        return visualize(cfg, run_dir=args.run, figure=args.figure, out=args.out, show=args.show)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
